"""Model downloads for Rulith Inference (design: tmp/downloads/DESIGN.md).

Two halves. The worker the manager starts for one job, `python tools/downloader.py <job folder>`; and the helpers
the manager uses to plan a job: the built-in catalog (model_catalog.json), repository listings, drives and folders.

A job is a folder config/jan/downloads/<id>/:
- job.json: what to fetch: files with their URLs, size and sha256, and the target folder. Written once by the
  manager from the catalog or from its own repository listing, never from what a client sent.
- state.json: the worker's progress (atomic writes, a few a second).
- control.json: the manager's requests to a running worker ({"stop": true}).

Each file is fetched into <folder>/<name>.part with several ranged connections, hashed while its finished prefix
grows, and renamed to <name> once the sha256 matches. Only new files are written: a file that already exists is never
opened for writing, and a .part file is only ever one this downloader created. Standard library only.
"""
import ctypes
import datetime as dt
import hashlib
import html
import html.parser
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CATALOG = Path(__file__).with_name('model_catalog.json')
CHUNK = 64 << 20                 # one ranged request, at most
MIN_CHUNK = 4 << 20              # small files: chunks of size / (2 x connections), not below this
READ = 1 << 20                   # one read from the socket
CONNECTIONS = 8                  # per file
MAX_CONNECTIONS = 16
WINDOW = 2                       # chunks handed out at most CONNECTIONS * WINDOW past the first unfinished one
RETRIES = 5                      # per chunk, with backoff, once the source has delivered
FIRST_TRIES = 2                  # a source that never delivered is given up after this many failures
TIMEOUT = 30
STATE_EVERY = 0.5
BUSY_WAIT = 2.0                  # seconds a job file another program holds open is waited for (retry_busy)
SPACE_MARGIN = 2 << 30
HOSTS = ('huggingface.co', 'hf.co', 'hf-mirror.com', 'github.com', 'githubusercontent.com', 'gh-proxy.com')
SCHEMES = ('https',)
UA = 'RulithInference-downloader/1'
# tests only: plain http to these hosts as well (a local test server)
if os.environ.get('RULITH_DL_TEST_HOSTS'):
    HOSTS = HOSTS + tuple(os.environ['RULITH_DL_TEST_HOSTS'].split(','))
    SCHEMES = ('https', 'http')


class DownloadError(Exception):
    def __init__(self, code, **params):
        super().__init__(code)
        self.code, self.params = code, params


class _Stopped(Exception):
    pass


class _Released(Exception):
    pass


class _NoRanges(Exception):
    pass


class _SourceFailed(Exception):
    def __init__(self, error):
        super().__init__(error.code)
        self.error = error


# ---- network ----

def allowed(url):
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or '').lower()
    return u.scheme in SCHEMES and any(host == h or host.endswith('.' + h) for h in HOSTS)


def host_of(url):
    return urllib.parse.urlsplit(url).hostname or ''


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Redirects only to the allowed hosts (the sources and their CDNs)."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed(newurl): raise DownloadError('redirect_refused', host=host_of(newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# The system proxy (urllib reads the environment, else the Windows setting) for the sources abroad; the mirrors in
# China are reached directly, as a proxy there is often far slower than the mirror itself.
DIRECT = ('hf-mirror.com', 'gh-proxy.com', 'modelscope.cn')


def system_proxies():
    """HTTP(S)_PROXY from the environment, else the Windows proxy setting (Clash, v2rayN and the like set it).
    urllib's own lookup reads the registry only when the environment has no *_PROXY variable at all, so a lone
    NO_PROXY hid the Windows proxy and huggingface.co was tried directly."""
    env = urllib.request.getproxies_environment()
    if env.get('https') or env.get('http'): return env
    reg = urllib.request.getproxies_registry() if os.name == 'nt' else {}
    return {**reg, **({'no': env['no']} if 'no' in env else {})}


OPENER = urllib.request.build_opener(_Redirects(), urllib.request.ProxyHandler(system_proxies()))
OPENER_DIRECT = urllib.request.build_opener(_Redirects(), urllib.request.ProxyHandler({}))


def opener_for(url):
    host = host_of(url).lower()
    return OPENER_DIRECT if any(host == h or host.endswith('.' + h) for h in DIRECT) else OPENER


def get(url, headers=None, timeout=TIMEOUT):
    if not allowed(url): raise DownloadError('host_refused', host=host_of(url))
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'identity', **(headers or {})})
    try:
        return opener_for(url).open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        e.close()
        raise DownloadError('http', status=e.code, host=host_of(url))
    except DownloadError:
        raise
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise DownloadError('network', reason=str(getattr(e, 'reason', e)), host=host_of(url))


# listings, searches and model cards: a source that does not answer in this time is given up for the other one
META_TIMEOUT = 10


def get_json(url, timeout=META_TIMEOUT):
    with get(url, {'Accept': 'application/json'}, timeout) as r:
        return json.loads(r.read(64 << 20)), r.headers


def retryable(e):
    """A chunk error worth another try at the same source."""
    if e.code in ('network', 'short_read'): return True
    return e.code == 'http' and (e.params.get('status') == 429 or e.params.get('status', 0) >= 500)


# ---- catalog, sources, repositories ----

def load_catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))


SPEC = re.compile(r'([\w.-]+)/([\w.-]+)@([\w.-]+):([\w./+-]+)')
PAIR = {'hf': 'gh', 'hfm': 'ghp', 'gh': 'hf', 'ghp': 'hfm'}     # the same region's source for the other kind


def source_url(source, spec):
    """'owner/repo@rev:path' on a source of kind hf (resolve URL) or gh (release asset)."""
    m = SPEC.fullmatch(spec)
    if not m or '..' in m[4].split('/'): raise DownloadError('catalog_invalid', spec=spec)
    owner, repo, rev, path = m.groups()
    q = urllib.parse.quote
    if source['kind'] == 'hf': return f"{source['base']}/{owner}/{repo}/resolve/{rev}/{q(path)}"
    if source['kind'] == 'gh': return f"{source['base']}/{owner}/{repo}/releases/download/{q(rev)}/{q(path)}"
    raise DownloadError('catalog_invalid', spec=spec)


def source_order(sources, prefer):
    """prefer first, then its region's source of the other kind, then the rest in catalog order"""
    head = [k for k in (prefer, PAIR.get(prefer)) if k in sources]
    return head + [k for k in sources if k not in head]


def entry_urls(catalog, entry, prefer):
    sources = catalog['sources']
    return [source_url(sources[k], entry[sources[k]['kind']]) for k in source_order(sources, prefer)
            if sources[k]['kind'] in entry]


def safe_name(name):
    """a file name the job may create: a base name, ending in .gguf"""
    base = re.sub(r'[\x00-\x1f<>:"|?*\\/]', '_', str(name).rsplit('/', 1)[-1]).strip(' .')
    if not base.lower().endswith('.gguf') or base.lower().startswith(('con.', 'prn.', 'aux.', 'nul.')):
        raise DownloadError('name_invalid', name=str(name))
    return base


def safe_dir(text):
    return re.sub(r'[^\w.-]', '_', str(text)).strip('.') or '_'


REPO_URL = re.compile(r'^https://(huggingface\.co|hf-mirror\.com)/(?!api/|datasets/|spaces/)([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]*?)'
                      r'(?:\.git)?(?:/(?:tree|resolve|blob)/([\w.-]+)(?:/.*)?)?/?$')
REPO_ID = re.compile(r'^([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]*)$')


def parse_repo(text, source=None):
    """A repository link or 'owner/repo' -> (source key, owner, repo, revision)."""
    t = str(text).strip()
    m = REPO_URL.match(t)
    if m:
        return ('hf' if m[1] == 'huggingface.co' else 'hfm'), m[2], m[3], m[4] or 'main'
    m = REPO_ID.match(t)
    if m:
        return (source or 'hf'), m[1], m[2], 'main'
    raise DownloadError('repo_invalid', text=t[:200])


SHARD = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.I)
QUANT = re.compile(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', re.I)
EXL3_BPW = re.compile(r'EXL3[-_](\d+(?:\.\d+)?)bpw', re.I)


def quant_of(name):
    """the quantization a file name gives: UD-IQ4_XS -> IQ4_XS, ...-EXL3-4.05bpw -> EXL3 4.05 bpw"""
    e = EXL3_BPW.search(name)
    if e: return f'EXL3 {e[1]} bpw'
    q = QUANT.search(name)
    return q[1] if q else ''


def file_role(name):
    low = name.lower()
    return 'projection' if low.startswith('mmproj') else 'draft' if low.startswith('mtp-') else 'model'


def group_files(files):
    """[{path, size, sha256}] -> groups: the shards of one model together, each projector and draft alone"""
    groups = {}
    for f in files:
        name = f['path'].rsplit('/', 1)[-1]
        s = SHARD.search(name)
        key = f['path'][:len(f['path']) - len(name)] + (name[:s.start()] if s else name[:-5])
        g = groups.setdefault(key, {'key': key, 'role': file_role(name), 'files': [], 'shards': int(s[2]) if s else 1})
        g['files'].append({**f, 'name': name})
    out = []
    for g in groups.values():
        g['files'].sort(key=lambda f: f['name'])
        out.append({**g, 'name': g['key'].rsplit('/', 1)[-1], 'quant': quant_of(g['files'][0]['name']),
                    'size': sum(f['size'] for f in g['files']), 'complete': len(g['files']) == g['shards']})
    return sorted(out, key=lambda g: (g['role'] != 'model', g['key'].lower()))


def list_repo(catalog, source, owner, repo, rev='main'):
    """The .gguf files of a Hugging Face repository (through huggingface.co or hf-mirror.com), at the commit the
    listing saw."""
    src = catalog['sources'].get(source)
    if not src or src['kind'] != 'hf': raise DownloadError('source_unsupported', source=source)
    base, q = src['base'], urllib.parse.quote
    try:
        info, _ = get_json(f'{base}/api/models/{owner}/{repo}/revision/{q(rev, safe="")}')
    except DownloadError as e:
        if e.code == 'http' and e.params.get('status') in (401, 403, 404):
            raise DownloadError('repo_unavailable', status=e.params['status'], repo=f'{owner}/{repo}')
        raise
    commit = info.get('sha')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise DownloadError('repo_unavailable', status=0, repo=f'{owner}/{repo}')
    if info.get('gated') or info.get('private'):
        raise DownloadError('repo_unavailable', status=403, repo=f'{owner}/{repo}')
    files, url = [], f'{base}/api/models/{owner}/{repo}/tree/{commit}?recursive=true'
    for _ in range(50):                              # pages of the tree listing (Link: <...>; rel="next")
        page, headers = get_json(url)
        for e in page if isinstance(page, list) else []:
            if e.get('type') != 'file' or not str(e.get('path', '')).lower().endswith('.gguf'): continue
            lfs = e.get('lfs') or {}
            files.append({'path': e['path'], 'size': int(lfs.get('size', e.get('size', 0))),
                          'sha256': lfs.get('oid') if re.fullmatch(r'[0-9a-f]{64}', str(lfs.get('oid', ''))) else None})
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get('Link', '') or '')
        if not m: break
        url = urllib.parse.urljoin(base, m[1])
    if not files: raise DownloadError('repo_no_gguf', repo=f'{owner}/{repo}')
    return {'source': source, 'owner': owner, 'repo': repo, 'revision': rev, 'commit': commit,
            'groups': group_files(files), 'info': info_of(info)}


def info_of(info):
    """what the page shows above a repository: its counts, last change, task, license, base model and the GGUF
    header's architecture, parameter count and context length"""
    g = info.get('gguf') if isinstance(info.get('gguf'), dict) else {}
    card = info.get('cardData') if isinstance(info.get('cardData'), dict) else {}
    num = lambda v: int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    text = lambda v: v if isinstance(v, str) else None
    base = card.get('base_model')
    if isinstance(base, list): base = base[0] if base else None
    return {'downloads': num(info.get('downloads')) or 0, 'likes': num(info.get('likes')) or 0,
            'updated': text(info.get('lastModified')), 'task': text(info.get('pipeline_tag')),
            'license': text(card.get('license_name')) or text(card.get('license')), 'base_model': text(base),
            'architecture': text(g.get('architecture')), 'params': num(g.get('total')),
            'context': num(g.get('context_length'))}


def repo_info(catalog, source, owner, repo):
    """info_of a repository's main branch (a recommended model's header)"""
    src = catalog['sources'].get(source)
    if not src or src['kind'] != 'hf': raise DownloadError('source_unsupported', source=source)
    try:
        info, _ = get_json(f"{src['base']}/api/models/{owner}/{repo}/revision/main")
    except DownloadError as e:
        if e.code == 'http' and e.params.get('status') in (401, 403, 404):
            raise DownloadError('repo_unavailable', status=e.params['status'], repo=f'{owner}/{repo}')
        raise
    return info_of(info)


SEARCH_FIELDS = ('downloads', 'likes', 'lastModified', 'pipeline_tag', 'gguf')


def search_repos(catalog, source, query, limit=40, author=None):
    """GGUF repositories matching `query` (or all of `author`'s), most downloaded first: id, counts, last change,
    task, and the GGUF header's architecture and parameter count the hub reads from the files"""
    src = catalog['sources'].get(source)
    if not src or src['kind'] != 'hf': raise DownloadError('source_unsupported', source=source)
    q = urllib.parse.quote
    what = f'author={q(author)}' if author else f'search={q(str(query).strip()[:200])}'
    url = (f"{src['base']}/api/models?{what}&filter=gguf&sort=downloads&direction=-1"
           f"&limit={int(limit)}" + ''.join(f'&expand%5B%5D={f}' for f in SEARCH_FIELDS))
    found, _ = get_json(url)
    out = []
    for m in found if isinstance(found, list) else []:
        ident = str(m.get('id', ''))
        if not REPO_ID.match(ident): continue
        g = m.get('gguf') or {}
        out.append({'id': ident, 'owner': ident.split('/')[0], 'repo': ident.split('/')[1],
                    'downloads': int(m.get('downloads') or 0), 'likes': int(m.get('likes') or 0),
                    'updated': m.get('lastModified'), 'task': m.get('pipeline_tag'),
                    'architecture': g.get('architecture'), 'params': g.get('total')})
    return out


def readme(catalog, source, owner, repo, commit):
    """a repository's README.md at the listing's commit, without its YAML header; '' when it has none"""
    src = catalog['sources'].get(source)
    if not src or src['kind'] != 'hf': raise DownloadError('source_unsupported', source=source)
    try:
        with get(source_url(src, f'{owner}/{repo}@{commit}:README.md'), None, META_TIMEOUT) as r:
            text = r.read(512 << 10).decode('utf-8', errors='replace')
    except DownloadError as e:
        if e.code == 'http' and e.params.get('status') == 404: return ''
        raise
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end > 0: text = text[end + 4:]
    return markdown_only(text).lstrip('\n')


class _Table(html.parser.HTMLParser):
    """the rows of an HTML table as lists of cell texts"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.rows.append([])
        elif tag in ('td', 'th'): self.cell = []
        elif tag in ('br', 'div', 'p', 'li') and self.cell is not None: self.cell.append(' ')   # blocks inside a cell

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            if not self.rows: self.rows.append([])
            self.rows[-1].append(' '.join(''.join(self.cell).split()).replace('|', '\\|'))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)


def _md_table(block):
    t = _Table()
    t.feed(block)
    rows = [r for r in t.rows if r]
    if not rows: return ''
    width = max(len(r) for r in rows)
    rows = [r + [''] * (width - len(r)) for r in rows]
    lines = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '---|' * width] + ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
    return '\n\n' + '\n'.join(lines) + '\n\n'


def markdown_only(text):
    """A model card's Markdown with its raw HTML made safe to show as Markdown: style, script and comments dropped,
    tables turned into Markdown tables, line breaks kept, every other tag removed (its text stays). The page renders
    Markdown only, never HTML from a repository."""
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'<(style|script|svg|iframe)\b.*?</\1\s*>', '', text, flags=re.S | re.I)
    text = re.sub(r'<table\b.*?</table\s*>', lambda m: _md_table(m[0]), text, flags=re.S | re.I)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(r'</?[a-zA-Z][^<>]*>', '', text)
    text = html.unescape(text) if '&' in text else text
    return re.sub(r'\n{3,}', '\n\n', text)


def repo_urls(catalog, prefer, owner, repo, commit, path):
    """the file at the pinned commit on prefer, then on the other sources of kind hf"""
    sources = catalog['sources']
    keys = [k for k in source_order(sources, prefer) if sources[k]['kind'] == 'hf']
    return [source_url(sources[k], f'{owner}/{repo}@{commit}:{path}') for k in keys]


# ---- drives and folders ----

def fixed_drives():
    if os.name != 'nt': return ['/']
    k = ctypes.windll.kernel32
    mask = k.GetLogicalDrives()
    return [f'{chr(65 + i)}:\\' for i in range(26) if mask >> i & 1 and k.GetDriveTypeW(f'{chr(65 + i)}:\\') == 3]


def free_bytes(path):
    p = Path(path)
    while not p.exists() and p != p.parent: p = p.parent
    return shutil.disk_usage(p).free


def drives_view():
    out = []
    for d in fixed_drives():
        try:
            u = shutil.disk_usage(d)
            out.append({'drive': d, 'free': u.free, 'total': u.total})
        except OSError:
            pass
    return out


def models_root():
    """<the fixed drive with the most free space>\\Rulith Models: the default download root"""
    drives = drives_view()
    if not drives: raise DownloadError('folder_invalid', folder='')
    return str(Path(max(drives, key=lambda d: d['free'])['drive']) / 'Rulith Models')


def is_fixed(path):
    if os.name != 'nt': return True
    return Path(path).drive.upper() + '\\' in fixed_drives()


# ---- the worker ----

def now():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def retry_busy(fn):
    """fn(), again while it fails with PermissionError, for up to BUSY_WAIT seconds. On Windows a file cannot be
    replaced while another handle has it open, and opening one while it is being replaced fails as well: the manager
    reading state.json for the Downloads page as the worker saves it, or a virus scanner or the search indexer
    looking at either. Both pass in milliseconds; a file refused for longer is a real error (GitHub issue #23)."""
    end = time.monotonic() + BUSY_WAIT
    while True:
        try:
            return fn()
        except PermissionError:
            if time.monotonic() >= end: raise
            time.sleep(0.02)


def read_json(path, default=None):
    """default for a missing or broken file; one that stays locked raises, so that a worker never starts from an
    empty state.json and fetches the finished chunks again"""
    try:
        return json.loads(retry_busy(lambda: Path(path).read_text(encoding='utf-8')))
    except PermissionError:
        raise
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    text = json.dumps(data, ensure_ascii=False)

    def write():
        temp.write_text(text, encoding='utf-8')
        os.replace(temp, path)
    retry_busy(write)


def _kernel32():
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID,
                                  wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    k.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    k.SetEndOfFile.argtypes = [wintypes.HANDLE]
    return k


def set_sparse(f, on):
    """NTFS zero-fills the gap when a write lands past a file's valid data length, synchronously; a sparse file
    leaves the gap unallocated. Set while downloading, cleared once every byte is written. Best effort."""
    if os.name != 'nt': return
    try:
        import msvcrt
        out = ctypes.c_ulong()
        flag = ctypes.c_ubyte(1 if on else 0)                     # FILE_SET_SPARSE_BUFFER
        _kernel32().DeviceIoControl(msvcrt.get_osfhandle(f.fileno()), 0x000900C4, ctypes.byref(flag), 1, None, 0,
                                    ctypes.byref(out), None)
    except (OSError, AttributeError):
        pass


def set_size(f, size):
    """Extend f to size without writing (SetEndOfFile). truncate() on Windows goes through the CRT's _chsize_s, which
    extends a file by writing zeros and would allocate the whole sparse file up front."""
    if os.name != 'nt':
        f.truncate(size)
        return
    import msvcrt
    k, h = _kernel32(), msvcrt.get_osfhandle(f.fileno())
    if not k.SetFilePointerEx(h, size, None, 0) or not k.SetEndOfFile(h):
        raise DownloadError('disk_error', reason=f'SetEndOfFile failed ({ctypes.get_last_error()})')


class Job:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.job = read_json(self.dir / 'job.json')
        if not self.job: raise DownloadError('job_invalid', folder=str(folder))
        self.state = read_json(self.dir / 'state.json', {}) or {}
        self.state.setdefault('files', {})
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.written = 0.0
        self.log_file = open(self.dir / 'worker.log', 'a', encoding='utf-8')

    def log(self, text):
        self.log_file.write(f'{now()} {text}\n')
        self.log_file.flush()

    def save(self, force=False):
        t = time.monotonic()
        if not force and t - self.written < STATE_EVERY: return
        self.written = t
        self.state['updated_at'] = now()
        with self.lock:
            try:
                write_json(self.dir / 'state.json', self.state)
            except PermissionError as e:
                if force: raise
                self.log(f'progress not saved this time: {e!r}')      # the next update saves it; the download goes on

    def check_control(self):
        c = read_json(self.dir / 'control.json', {}) or {}
        if c.get('stop'): self.stop.set()
        if self.stop.is_set(): raise _Stopped()

    def run(self):
        st = self.state
        st.update(status='running', pid=os.getpid(), error=None, started_at=now(), speed=0)
        self.save(True)
        self.log(f'start: {len(self.job["files"])} files into {self.job["folder"]}')
        try:
            folder = Path(self.job['folder'])
            folder.mkdir(parents=True, exist_ok=True)
            for spec in self.job['files']:
                self.check_control()
                self.fetch_file(folder, spec)
            st['status'] = 'done'
            self.log('done')
        except _Stopped:
            st['status'] = 'stopped'
            self.log('stopped')
        except DownloadError as e:
            st['status'], st['error'] = 'failed', {'code': e.code, 'params': e.params}
            self.log(f'failed: {e.code} {e.params}')
        except Exception as e:                       # noqa: BLE001 - reported, not swallowed
            st['status'], st['error'] = 'failed', {'code': 'internal', 'params': {'reason': repr(e)}}
            self.log(f'failed: {e!r}')
        finally:
            st['speed'] = 0
            st['finished_at'] = now()
            self.save(True)
            self.log_file.close()
        return st['status']

    def fetch_file(self, folder, spec):
        name, size, want = safe_name(spec['name']), int(spec['size']), spec.get('sha256')
        final, part = folder / name, folder / (name + '.part')
        fs = self.state['files'].setdefault(name, {'status': 'pending', 'done': [], 'source': 0})
        fs.update(size=size, bytes=fs.get('bytes', 0))
        if final.exists():
            # never touched: a file of that name is used when its size matches, else the job stops
            if final.is_file() and final.stat().st_size == size:
                fs.update(status='done', bytes=size, done=[], present=fs.get('status') != 'done')
                return
            raise DownloadError('name_taken', file=name)
        urls = spec['urls']
        while True:
            if fs['source'] >= len(urls): raise DownloadError('sources_failed', file=name, last=fs.get('last_error'))
            try:
                digest = self.download(part, size, urls[fs['source']], fs)
            except _SourceFailed as e:
                fs['last_error'] = {'code': e.error.code, 'params': e.error.params}
                self.log(f'{name}: source {fs["source"]} failed: {e.error.code} {e.error.params}')
                fs['source'] += 1
                continue
            if want and digest != want:
                self.log(f'{name}: sha256 {digest} != {want} from source {fs["source"]}')
                part.unlink()
                fs.update(done=[], bytes=0, last_error={'code': 'hash_mismatch', 'params': {}})
                fs['source'] += 1
                if fs['source'] >= len(urls): raise DownloadError('hash_mismatch', file=name)
                continue
            break
        with open(part, 'r+b') as f:
            set_sparse(f, False)
        os.rename(part, final)                       # fails rather than replace a file that appeared meanwhile
        fs.update(status='done', done=[], bytes=size, verified=bool(want))
        self.log(f'{name}: done ({"sha256 ok" if want else "size only"})')
        self.save(True)

    def download(self, part, size, url, fs):
        """Fetch every missing chunk of part from url, hashing the finished prefix as it grows; returns the
        hex sha256 of the whole file."""
        # a mirror limits each connection, so a small file is cut finer to use them all; the size a .part was
        # started with stays with it (its finished chunks are counted in it)
        conns = max(1, min(MAX_CONNECTIONS, int(self.job.get('connections', CONNECTIONS))))
        if not part.exists() or not fs.get('chunk'):
            fs['chunk'] = max(min(CHUNK, MIN_CHUNK), min(CHUNK, -(-size // (2 * conns))))
        chunk = int(fs['chunk'])
        n = max(1, math.ceil(size / chunk))
        if part.exists() and part.stat().st_size != size: part.unlink()
        if not part.exists():
            with open(part, 'xb') as f:
                set_sparse(f, True)
                set_size(f, size)
            fs['done'] = []
        done = set(fs['done'])
        fs['status'] = 'downloading'
        hasher, hashed = hashlib.sha256(), 0
        reader = open(part, 'rb')
        try:
            def hash_prefix():
                nonlocal hashed
                while hashed in done:
                    reader.seek(hashed * chunk)
                    left = min(chunk, size - hashed * chunk)
                    while left:
                        b = reader.read(min(READ * 8, left))
                        if not b: raise DownloadError('short_file', file=part.name)
                        hasher.update(b)
                        left -= len(b)
                    hashed += 1
            hash_prefix()                            # a resume re-reads what was already there
            if len(done) < n:
                try:
                    self.fetch_chunks(part, size, n, chunk, url, done, fs, hash_prefix)
                except _NoRanges:
                    self.log(f'{part.name}: the source sends whole files only; starting the file over')
                    done.clear()
                    fs['done'] = []
                    hasher, hashed = hashlib.sha256(), 0
                    self.fetch_chunks(part, size, n, chunk, url, done, fs, hash_prefix, single=True)
            hash_prefix()
            if hashed != n: raise DownloadError('internal', reason='hash prefix incomplete')
            return hasher.hexdigest()
        finally:
            reader.close()

    def fetch_chunks(self, part, size, n, chunk, url, done, fs, hash_prefix, single=False):
        conns = max(1, min(MAX_CONNECTIONS, int(self.job.get('connections', CONNECTIONS))))
        window = conns * WINDOW
        cond = threading.Condition()
        inflight = {}                                # chunk -> bytes received so far
        errors = []
        delivered = [False]
        single = [single or n == 1]
        abort = threading.Event()

        def next_chunk():
            with cond:
                while True:
                    if abort.is_set() or errors or self.stop.is_set(): return None
                    low = next((i for i in range(n) if i not in done), None)
                    if low is None: return None
                    if single[0]:                    # one stream from offset 0 for the whole file
                        if 0 in done or 0 in inflight: return None
                        inflight[0] = 0
                        return 0
                    for i in range(low, min(n, low + window)):
                        if i not in done and i not in inflight:
                            inflight[i] = 0
                            return i
                    cond.wait(0.5)

        def fetch(i):
            a, b = i * chunk, min(size, (i + 1) * chunk)
            failures = 0
            while True:
                try:
                    end = self.fetch_range(url, part, a, b, size, lambda k: inflight.__setitem__(i, inflight[i] + k))
                    delivered[0] = True
                    if a == 0 and end == size: single[0] = True
                    return list(range(i, math.ceil(end / chunk)))
                except DownloadError as e:
                    inflight[i] = 0
                    if e.code == 'no_ranges':        # the source sends whole files only: chunk 0 streams it
                        single[0] = True
                        raise _Released()
                    failures += 1
                    if not retryable(e) or failures >= (RETRIES if delivered[0] else FIRST_TRIES): raise
                    self.log(f'{part.name}: chunk {i} try {failures}: {e.code} {e.params}')
                    if self.stop.wait(min(30, 2 ** failures)): raise _Stopped()

        def worker():
            while True:
                i = next_chunk()
                if i is None: return
                try:
                    finished = fetch(i)
                except _Stopped:
                    return
                except _Released:
                    with cond:
                        inflight.pop(i, None)
                        cond.notify_all()
                    continue
                except DownloadError as e:
                    with cond:
                        errors.append(e)
                        inflight.pop(i, None)
                        cond.notify_all()
                    return
                with cond:
                    done.update(finished)
                    inflight.pop(i, None)
                    cond.notify_all()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(1 if single[0] else conns)]
        for t in threads: t.start()
        last_t, last_b = time.monotonic(), None
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.2)
                try:
                    self.check_control()
                except _Stopped:
                    abort.set()
                    raise
                hash_prefix()
                with cond:
                    fs['done'] = sorted(done)
                    got = sum(min(chunk, size - i * chunk) for i in done) + sum(inflight.values())
                fs['bytes'] = got
                total = sum(f.get('bytes', 0) for f in self.state['files'].values())
                t = time.monotonic()
                if last_b is not None and t > last_t:
                    inst = max(0.0, (total - last_b) / (t - last_t))
                    self.state['speed'] = 0.7 * self.state.get('speed', 0) + 0.3 * inst
                last_t, last_b = t, total
                self.state['bytes'] = total
                self.save()
        finally:
            abort.set()
            for t in threads: t.join()
            with cond: fs['done'] = sorted(done)
            self.save(True)
        if self.stop.is_set(): raise _Stopped()
        if single[0] and len(done) < n and not errors:
            raise _NoRanges()                        # resumed onto a whole-file source: the file starts over
        if errors:
            e = errors[0]
            if e.code in ('disk_full', 'disk_error'): raise e
            raise _SourceFailed(e)

    def fetch_range(self, url, part, a, b, total, on_bytes):
        """Write bytes a..b of url into part; returns the end written. A server that ignores Range answers 200 with
        the whole file: from offset 0 that is taken whole (returns total), anywhere else it raises no_ranges."""
        with get(url, {'Range': f'bytes={a}-{b - 1}'}) as r:
            if r.status == 206:
                m = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', (r.headers.get('Content-Range') or '').strip())
                if not m or int(m[1]) != a or int(m[2]) != b - 1 or (m[3] != '*' and int(m[3]) != total):
                    raise DownloadError('bad_range', host=host_of(url))
            elif r.status == 200 and a == 0:
                if int(r.headers.get('Content-Length') or total) != total:
                    raise DownloadError('bad_range', host=host_of(url))
                b = total
            else:
                raise DownloadError('no_ranges', host=host_of(url))
            with open(part, 'r+b') as f:
                f.seek(a)
                pos = a
                while pos < b:
                    if self.stop.is_set(): raise _Stopped()
                    try:
                        data = r.read(min(READ, b - pos))
                    except (OSError, http.client.HTTPException) as e:
                        raise DownloadError('network', reason=repr(e)[:200], host=host_of(url))
                    if not data: raise DownloadError('short_read', host=host_of(url))
                    try:
                        f.write(data)
                    except OSError as e:
                        raise DownloadError('disk_full' if e.errno == 28 or getattr(e, 'winerror', 0) == 112
                                            else 'disk_error', reason=str(e))
                    pos += len(data)
                    on_bytes(len(data))
                try:
                    f.flush()
                    os.fsync(f.fileno())            # a chunk counts as done only once it is on the disk
                except OSError as e:
                    raise DownloadError('disk_error', reason=str(e))
        return b


def main():
    if len(sys.argv) != 2: sys.exit('usage: downloader.py <job folder>')
    status = Job(sys.argv[1]).run()
    sys.exit(0 if status in ('done', 'stopped') else 1)


if __name__ == '__main__': main()
