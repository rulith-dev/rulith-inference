"""Authenticated installation and shared generation leases for the desktop hosts."""
import contextlib
import ctypes
from ctypes import wintypes
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from functools import lru_cache

MANIFEST = 'rulith-brain-runtime.json'
SIG = MANIFEST + '.sig'
MODULE = 'manager/rulith_brain_manager.py'
CAPS = ('ui.frame.v1', 'manager.python.v1', 'memory.stream.v1', 'ui.store.v1')
DOMAIN = b'rulith-brain-runtime-manifest-v2\n'
CHANNEL = 'https://github.com/rulith-dev/rulith-brain/releases/'
PIN = Path(__file__).with_name('brain_runtime_trust.bin')
MAX_MANIFEST = 1 << 20
MAX_ZIP = 16 << 30
MAX_TREE = 64 << 30
OPS = frozenset(('brain_runtime', 'brain_runtime_check', 'brain_runtime_install', 'brain_runtime_cancel'))


class HostError(ValueError):
    def __init__(self, code, **params):
        self.code, self.params = code, params
        messages = {
            'runtime_missing':'Install the Brain runtime to continue.',
            'runtime_legacy':'Update the Brain runtime to use it in this app.',
            'runtime_untrusted':'The Brain runtime publisher could not be verified.',
            'runtime_integrity':'The installed Brain runtime needs repair.',
            'runtime_abi':'This Brain runtime requires a different host version.',
            'runtime_schema':'This app cannot read the runtime manifest.',
            'runtime_capability':'The runtime is missing a required capability.',
            'owner_required':'Open the desktop app to control the engine.',
            'owner_unknown':'The engine process could not be identified.',
            'owner_busy':'Another application owns the engine.',
            'switch_busy':'An engine transition is already in progress.',
            'stop_timeout':'The engine is still running.',
            'persist_failed':'The engine could not save its conversations.',
            'save_failed':'The engine could not save the current memory.',
            'brain_no_model':'The required model files are missing.',
            'brain_no_folder':'Select a memory folder before starting Brain.',
            'brain_not_running':'Start Brain to use this operation.',
            'brain_module_failed':'The Brain runtime could not complete the request.',
            'store_conflict':'This document changed in another window.',
        }
        super().__init__(messages.get(code,code.replace('_', ' ').capitalize() + '.'))


def require(condition, code='runtime_integrity'):
    if not condition:
        raise HostError(code)


def home():
    buf = ctypes.create_unicode_buffer(32768)
    require(ctypes.windll.shell32.SHGetFolderPathW(None, 0x1c, None, 0, buf) == 0, 'runtime_missing')
    base=Path(buf.value).resolve()/'Rulith'
    result=base/'Brain'
    for folder in (base,result):
        if os.path.lexists(folder):
            regular(folder,directory=True)
    return result


@contextlib.contextmanager
def lease(path, shared=False, wait=True):
    """All participating readers and writers lock byte zero of the same stable file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as file:
        if os.name == 'nt':
            import msvcrt
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(file.fileno()))
            k = ctypes.WinDLL('kernel32', use_last_error=True)
            # OVERLAPPED is two pointer-width words, two DWORD offsets and an event handle.
            overlapped = (ctypes.c_size_t * 5)()
            locked = k.LockFileEx(handle, (0 if shared else 2) | (0 if wait else 1),
                                  0, 1, 0, ctypes.byref(overlapped))
            if not locked and (wait or ctypes.get_last_error() != 33):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                yield bool(locked)
            finally:
                if locked:
                    k.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlapped))
        else:
            import fcntl
            try:
                fcntl.flock(file, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            yield True


@lru_cache(maxsize=1)
def personal_dacl():
    adv=ctypes.WinDLL('advapi32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    token=wintypes.HANDLE()
    if not adv.OpenProcessToken(wintypes.HANDLE(-1),8,ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        n=wintypes.DWORD()
        adv.GetTokenInformation(token,1,None,0,ctypes.byref(n))
        buf=ctypes.create_string_buffer(n.value)
        if not adv.GetTokenInformation(token,1,buf,n,ctypes.byref(n)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid=ctypes.c_void_p.from_buffer(buf)
        name=wintypes.LPWSTR()
        if not adv.ConvertSidToStringSidW(sid,ctypes.byref(name)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return f'D:P(A;;FA;;;{name.value})(A;;FA;;;SY)'
        finally:
            kernel.LocalFree(ctypes.cast(name,ctypes.c_void_p))
    finally:
        kernel.CloseHandle(token)


def private_file(path):
    if os.name!='nt':
        path.chmod(0o600)
        return
    adv=ctypes.WinDLL('advapi32',use_last_error=True)
    sd=ctypes.c_void_p()
    if not adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(personal_dacl(),1,ctypes.byref(sd),None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not adv.SetFileSecurityW(str(path),wintypes.DWORD(0x80000004),sd):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        ctypes.windll.kernel32.LocalFree(sd)


def durable(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('xb') as f:
            private_file(tmp)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        deadline=time.monotonic()+2
        while True:
            try:
                os.replace(tmp,path)
                break
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    durable(path, json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode())


def unique(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), 'runtime_schema')
    return obj


def limited(path, limit):
    with path.open('rb') as f:
        raw = f.read(limit + 1)
    require(len(raw) <= limit)
    return raw


def version(v):
    if isinstance(v, str) and re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', v):
        return tuple(map(int, v.split('.')))
    raise HostError('runtime_schema')


def verify_signature(raw, sig, pin):
    """Use the Ed25519 verifier in the embedded interpreter's OpenSSL distribution."""
    require(len(pin) == 32 and len(sig) == 64, 'runtime_untrusted')
    if os.name == 'nt':
        import _ssl
        lib = ctypes.CDLL(str(Path(_ssl.__file__).with_name('libcrypto-3.dll')))
    else:
        import ctypes.util as libs
        lib = ctypes.CDLL(libs.find_library('crypto'))
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    def function(name, result, *args):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = list(args), result
        return fn
    new_key = function('EVP_PKEY_new_raw_public_key_ex', ptr, ptr, ptr, ptr, ptr, size)
    key = new_key(None, b'ED25519', None, pin, len(pin))
    ctx = function('EVP_MD_CTX_new', ptr)()
    try:
        init = function('EVP_DigestVerifyInit', ctypes.c_int, ptr, ptr, ptr, ptr, ptr)
        verify = function('EVP_DigestVerify', ctypes.c_int, ptr, ptr, size, ptr, size)
        msg = DOMAIN + hashlib.sha256(raw).digest()
        require(ctx and key and init(ctx, None, None, None, key) == 1 and
                verify(ctx, sig, len(sig), msg, len(msg)) == 1, 'runtime_untrusted')
    finally:
        function('EVP_MD_CTX_free', None, ptr)(ctx)
        function('EVP_PKEY_free', None, ptr)(key)


def safe_name(name):
    require(isinstance(name, str) and bool(name))
    for part in name.split('/'):
        require(part and not part.endswith(('.', ' ')) and part not in ('.', '..') and
                all(32 <= ord(c) <= 126 and c not in '<>:"\\|?*' for c in part) and
                not re.fullmatch(r'(?i)(?:CON|PRN|AUX|NUL|CLOCK\$|CONIN\$|CONOUT\$|(?:COM|LPT)[0-9])',
                                 part.split('.')[0].rstrip()))
    require(name.lower().split('/')[0] not in (MANIFEST, SIG))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parse(raw, sig, pin):
    verify_signature(raw, sig, pin)
    require(len(raw) <= MAX_MANIFEST, 'runtime_schema')
    try:
        m = json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=lambda _: require(False))
        require(isinstance(m, dict), 'runtime_schema')
        schema = m.get('schema_version', 1)
        require(type(schema) is int, 'runtime_schema')
        require(schema != 1, 'runtime_legacy')
        require(schema == 2, 'runtime_schema')
        v = m['version']
        require(version(v) >= (0, 2, 2) and m['tag'] == 'v' + v and
                m['file'] == f'rulith-brain-runtime-{v}.zip', 'runtime_schema')
        require(type(m['size']) is int and 0 < m['size'] <= MAX_ZIP and
                isinstance(m['notes'], str) and len(m['notes']) <= 2000 and not m['notes'].lstrip().upper().startswith('DRAFT') and
                m.get('draft', False) is False and type(m.get('signed', False)) is bool, 'runtime_schema')
        abi = m['host_abi']
        require(all(type(abi[k]) is int for k in ('min', 'max')) and 1 <= abi['min'] <= abi['max'], 'runtime_schema')
        require(abi['min'] <= 1 <= abi['max'], 'runtime_abi')
        caps = m['capabilities']
        require(isinstance(caps, list) and all(type(c) is str and c.isascii() for c in caps)
                and len(set(caps)) == len(caps), 'runtime_schema')
        require(set(CAPS) <= set(caps), 'runtime_capability')
        files, ui, manager = m['files'], m['ui'], m['manager']
        require(ui['dir'] == 'ui' and ui['entry'] == 'index.html' and manager['path'] == MODULE and
                manager['module'] == 'rulith_brain_manager' and isinstance(files, dict), 'runtime_schema')
        require({'llama-server.exe', 'ui/index.html', MODULE, 'EULA.md', 'THIRD-PARTY-NOTICES.md'} <= files.keys()
                and any(n.startswith('licenses/') for n in files))
        aliases, expanded = {}, 0
        for name, entry in files.items():
            safe_name(name)
            require(set(entry) == {'size', 'sha256'} and type(entry['size']) is int and entry['size'] >= 0
                    and re.fullmatch('[0-9a-f]{64}', entry['sha256']), 'runtime_schema')
            expanded += entry['size']
            require(expanded <= MAX_TREE)
            for prefix in (p.as_posix() for p in Path(name).parents if p != Path('.')):
                require(prefix not in files and aliases.setdefault(prefix.lower(), prefix) == prefix)
            require(aliases.setdefault(name.lower(), name) == name)
        require(re.fullmatch('[0-9a-f]{64}', m['sha256']) and files[MODULE]['sha256'] == manager['sha256'])
        lines = ''.join(f"{files[n]['sha256']}  {n}\n" for n in sorted(files) if n.startswith('ui/'))
        require(hashlib.sha256(lines.encode()).hexdigest() == ui['sha256'])
        return m
    except (TypeError, KeyError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
        raise HostError('runtime_schema') from exc


def regular(path, directory=False):
    st = path.lstat()
    require(not getattr(st, 'st_file_attributes', 0) & 0x400 and
            (stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode) and st.st_nlink == 1))


# Consume comments and whole string literals before looking for a module declaration.
_JS_STRING = r"""(?:"(?:\\[\s\S]|[^"\\\r\n])*"|'(?:\\[\s\S]|[^'\\\r\n])*')"""
_JS_SPACE = r'(?:\s|/\*[\s\S]*?\*/|//[^\n]*(?:\n|$))*'
_JS_NAME = r'[\w$]+'
_JS_BINDINGS = rf'(?:\{{[^}}]*\}}|\*{_JS_SPACE}(?:as\s+{_JS_NAME})?)'
_JS_FROM = rf'(?:(?:{_JS_NAME}{_JS_SPACE},{_JS_SPACE})?{_JS_BINDINGS}|{_JS_NAME}){_JS_SPACE}\bfrom\b{_JS_SPACE}'
_JS_REFERENCES = re.compile(
    rf'//[^\n]*|/\*[\s\S]*?\*/|{_JS_STRING}|`(?:\\[\s\S]|[^`\\])*`'
    rf'|\b(?:import\b{_JS_SPACE}(?:\({_JS_SPACE}|{_JS_FROM})?|export\b{_JS_SPACE}{_JS_FROM})'
    rf'(?P<source>{_JS_STRING})')


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for field in ('src', 'href') if tag in ('script', 'link', 'img', 'source') else ():
            if field in attrs:
                self.refs.append(attrs[field])


def verify_tree(root, m, installed):
    regular(root, directory=True)
    seen = set()
    directories = {p.as_posix() for n in m['files'] for p in Path(n).parents}
    for parent, dirs, files in os.walk(root, onerror=lambda _: require(False)):
        for d in dirs:
            folder=Path(parent,d)
            regular(folder, directory=True)
            require(folder.relative_to(root).as_posix() in directories)
        for name in files:
            p = Path(parent, name)
            regular(p)
            rel = p.relative_to(root).as_posix()
            if installed and rel in (MANIFEST, SIG):
                continue
            entry = m['files'].get(rel)
            require(entry and p.stat().st_size == entry['size'] and digest(p) == entry['sha256'])
            seen.add(rel)
    require(seen == m['files'].keys())
    for name in seen:
        if not name.startswith('ui/'):
            continue
        refs = []
        if name.endswith('.html'):
            parser = References()
            parser.feed((root / name).read_text('utf-8'))
            refs = parser.refs
        elif name.endswith(('.js', '.mjs')):
            refs = [m['source'][1:-1] for m in _JS_REFERENCES.finditer((root/name).read_text('utf-8'))
                    if m['source']]
        elif name.endswith('.css'):
            refs = re.findall(r'''url\(\s*["']?([^)'"]+)''', (root/name).read_text('utf-8'))
        for ref in refs:
            if ref.startswith(('data:', '#')):
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), ref.split('?')[0].split('#')[0]))
            require(target.startswith('ui/') and target in seen)


class Runtime:
    def __init__(self, brain_home, pin=PIN, idle=None, smoke=None):
        self.home, self.pin = Path(brain_home), Path(pin)
        self.config = self.home / 'config/brain'
        self.live = self.home / 'bin/brain'
        self.new, self.old = (self.live.with_name(n) for n in ('brain.new', 'brain.old'))
        self.cache = self.config / 'runtime-download'
        self.idle = idle or self.engine_idle
        self.smoke = smoke or self.check_engine

    def check_layout(self):
        for path in (self.home,self.config.parent,self.config,self.cache,self.live.parent):
            if os.path.lexists(path):
                regular(path,directory=True)

    def engine_idle(self):
        import manager
        # Legacy processes cannot hold the generation lock.
        for p in manager.discover():
            exe = p.get('ExecutablePath')
            if not exe or Path(exe).resolve().parent in (self.live, self.old, self.new):
                return False
        return True

    @staticmethod
    def check_engine(root):
        r = subprocess.run([str(root / 'llama-server.exe'), '--version'], cwd=root, timeout=30,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        require(r.returncode == 0)

    def verified(self, root):
        self.check_layout()
        require(root.exists(), 'runtime_missing')
        regular(root, directory=True)
        try:
            raw = limited(root / MANIFEST, MAX_MANIFEST)
        except FileNotFoundError:
            raise HostError('runtime_integrity') from None
        try:
            meta = json.loads(raw.decode(), object_pairs_hook=unique)
            require(isinstance(meta, dict), 'runtime_schema')
            if type(meta.get('schema_version', 1)) is int and meta.get('schema_version', 1) == 1:
                raise HostError('runtime_legacy')
            sig, key = limited(root / SIG, 64), limited(self.pin, 32)
        except FileNotFoundError:
            raise HostError('runtime_untrusted') from None
        except (UnicodeError, json.JSONDecodeError):
            raise HostError('runtime_schema') from None
        m = parse(raw, sig, key)
        verify_tree(root, m, True)
        return m, hashlib.sha256(raw).hexdigest()

    def generation(self, shared=True, wait=True):
        self.check_layout()
        return lease(self.config / 'runtime.lock', shared, wait)

    def install_lock(self, wait=True):
        self.check_layout()
        return lease(self.config / 'runtime-install.lock', wait=wait)

    def stage(self, archive, raw, sig):
        self.check_layout()
        m = parse(raw, sig, limited(self.pin, 32))
        require(archive.stat().st_size == m['size'] and digest(archive) == m['sha256'])
        self.check_upgrade(m, raw)
        self.live.parent.mkdir(parents=True,exist_ok=True)
        regular(self.live.parent,directory=True)
        if self.new.exists():
            if self.good(self.new) and not self.good(self.live) and not self.good(self.old):
                self.reconcile()
                require(not self.new.exists(),'runtime_busy')
        if self.new.exists():
            regular(self.new, directory=True)
            shutil.rmtree(self.new)
        self.new.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            seen = set()
            for info in z.infolist():
                name = info.orig_filename
                entry = m['files'].get(name)
                require(entry and name == info.filename and name not in seen and info.file_size == entry['size'] and
                        stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG) and
                        not info.external_attr & 0x410 and not info.flag_bits & 1)
                seen.add(name)
                target = self.new / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, 1 << 20)
            require(seen == m['files'].keys())
        verify_tree(self.new, m, False)
        self.smoke(self.new)
        durable(self.new / SIG, sig)
        durable(self.new / MANIFEST, raw)

    def check_upgrade(self, m, raw):
        for tree in (self.live, self.old):
            path = tree / MANIFEST
            if not path.exists():
                continue
            previous = limited(path, MAX_MANIFEST)
            try:
                v = version(json.loads(previous).get('version'))
            except (ValueError, AttributeError):
                continue
            require(version(m['version']) >= v, 'runtime_downgrade')
            if v == version(m['version']) and previous != raw:
                try:
                    verify_signature(previous,limited(tree/SIG,64),limited(self.pin,32))
                except (FileNotFoundError,HostError):
                    # A damaged proof can be repaired. A stripped ZIP digest still identifies its release.
                    old_digest=json.loads(previous).get('sha256')
                    require(not old_digest or old_digest==m['sha256'],'runtime_substitution')
                else:
                    raise HostError('runtime_substitution')

    def good(self, tree):
        try:
            return self.verified(tree)[0]
        except (OSError, ValueError, zipfile.BadZipFile):
            return None

    def reconcile(self):
        with self.generation(False, False) as exclusive:
            if not exclusive or not self.idle():
                return
            for tree in (self.live,self.old,self.new):
                if os.path.lexists(tree):regular(tree,directory=True)
            live, old, new = (self.good(p) for p in (self.live, self.old, self.new))
            if not live and old:
                # Keep a newer install in place even when this host cannot read it.
                try:
                    parse(limited(self.live/MANIFEST,MAX_MANIFEST),limited(self.live/SIG,64),limited(self.pin,32))
                except (OSError,ValueError):
                    self.check_upgrade(old, (self.old / MANIFEST).read_bytes())
                if self.live.exists():
                    self.live.rename(self.live.with_name('brain.repair-' + uuid.uuid4().hex))
                self.old.rename(self.live)
                return
            if live and old and not new:
                shutil.rmtree(self.old)
                return
            if not self.live.exists() and self.old.exists() and not old:
                try:
                    self.verified(self.old)
                except HostError as e:
                    if e.code=='runtime_legacy':
                        self.old.rename(self.live)
                        return
                except OSError:
                    pass
            if new:
                self.check_upgrade(new, (self.new / MANIFEST).read_bytes())
                if self.old.exists():
                    shutil.rmtree(self.old)
                if self.live.exists():
                    self.live.rename(self.old)
                self.new.rename(self.live)
                self.verified(self.live)
                if self.old.exists():
                    shutil.rmtree(self.old)

    def status(self, held_digest=None):
        installed = dict(dir=str(self.live), found=(self.live/'llama-server.exe').is_file(), version=None, draft=False,
                         schema_version=None, verified=False, host_abi=None, capabilities=[], reason=None)
        with self.generation():
            try:
                if held_digest:
                    raw=limited(self.live/MANIFEST,MAX_MANIFEST)
                    require(hashlib.sha256(raw).hexdigest()==held_digest,'runtime_untrusted')
                    m=json.loads(raw)
                else:
                    m, _ = self.verified(self.live)
                installed.update({k:m[k] for k in ('version', 'schema_version', 'host_abi', 'capabilities')}, verified=True)
            except (OSError, ValueError) as e:
                installed['reason'] = getattr(e, 'code', 'runtime_integrity')
        staged = self.good(self.new)
        try:
            job = json.loads((self.cache/'job.json').read_bytes())
        except FileNotFoundError:
            job = None
        if job and job.get('alive'):
            import manager
            with lease(self.config/'runtime-worker.lock',wait=False) as inspect:
                if inspect:
                    record = manager.read_json(self.cache/'worker.json',{})
                    if not record or manager.process_identity(record['pid'])!=record:
                        job.update(alive=False,status='stopped',speed=0,eta=None)
        return dict(installed=installed, pending=staged and staged['version'], job=job, seed=None)

    def release(self, seed=None):
        def asset(name, limit):
            if seed:
                return limited(seed / name, limit)
            with urllib.request.urlopen(CHANNEL + 'latest/download/' + name, timeout=30) as response:
                raw = response.read(limit + 1)
                require(len(raw) <= limit)
                return raw
        raw, sig = asset(MANIFEST, MAX_MANIFEST), asset(SIG, 64)
        m = parse(raw, sig, limited(self.pin, 32))
        return m, raw, sig

    def run_job(self, seed=None):
        job_file = self.cache / 'job.json'
        with self.install_lock() as locked:
            try:
                m, raw, sig = self.release(seed)
                self.check_upgrade(m, raw)
                archive = seed / m['file'] if seed else self.cache / (m['sha256'] + '.part')
                job = dict(version=m['version'], status='running', alive=True, error=None, bytes=0,
                           total=m['size'], speed=0, eta=None)
                write_json(job_file, job)
                if not seed:
                    offset = archive.stat().st_size if archive.exists() else 0
                    if offset >= m['size'] and (offset != m['size'] or digest(archive) != m['sha256']):
                        archive.unlink()
                        offset = 0
                    if offset < m['size']:
                        req = urllib.request.Request(CHANNEL + 'download/' + m['tag'] + '/' + m['file'],
                                                     headers={'Range': f'bytes={offset}-'} if offset else {})
                        started = time.monotonic()
                        with urllib.request.urlopen(req, timeout=60) as response:
                            if response.status == 206:
                                require(response.headers.get('Content-Range') == f"bytes {offset}-{m['size']-1}/{m['size']}")
                            else:
                                require(response.status == 200)
                                offset = 0
                            with archive.open('ab' if offset else 'wb') as out:
                                while chunk := response.read(1 << 20):
                                    if (self.cache / 'cancel').exists():
                                        raise HostError('cancelled')
                                    require(out.tell() + len(chunk) <= m['size'])
                                    out.write(chunk)
                                    speed = (out.tell()-offset)/max(.001, time.monotonic()-started)
                                    job.update(bytes=out.tell(), speed=speed, eta=(m['size']-out.tell())/max(speed,1))
                                    write_json(job_file, job)
                if (self.cache / 'cancel').exists():
                    raise HostError('cancelled')
                self.stage(archive, raw, sig)
                self.reconcile()
                job.update(status='done', alive=False, bytes=m['size'], speed=0, eta=0)
            except Exception as e:
                job = locals().get('job', self.initial_job())
                code = getattr(e, 'code', None)
                job.update(status='stopped' if code == 'cancelled' else 'failed', alive=False,
                           error={'code': code if isinstance(code, str) else 'runtime_install_failed', 'params': {}})
            write_json(job_file, job)

    @staticmethod
    def initial_job():
        return dict(version='', status='starting', alive=True, error=None, bytes=0,total=0,speed=0,eta=None)

    def operation(self, op, data, seed=None, inside_generation=False, held_digest=None):
        require(op in OPS and data == {}, 'request_invalid')
        if op == 'brain_runtime_cancel':
            durable(self.cache / 'cancel', b'')
        elif op == 'brain_runtime_install':
            # Serialize the spawn and worker identity separately from long-running staging.
            with lease(self.config / 'runtime-worker.lock'):
                import manager
                record = manager.read_json(self.cache/'worker.json', {})
                if not record or manager.process_identity(record['pid']) != record:
                    self.cache.mkdir(parents=True, exist_ok=True)
                    (self.cache/'cancel').unlink(missing_ok=True)
                    write_json(self.cache/'job.json', self.initial_job())
                    worker = subprocess.Popen([sys.executable, '-I', '-B', '-c',
                        "import sys;sys.path.insert(0,sys.argv[1]);from brain_runtime_host import worker;worker()",
                        str(Path(__file__).parent), *([str(seed)] if seed else [])],
                        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    ident = manager.process_identity(worker.pid)
                    if ident:
                        write_json(self.cache/'worker.json', ident)
        elif op == 'brain_runtime' and not inside_generation and (self.old.exists() or self.new.exists()):
            with self.install_lock(False) as locked:
                if locked:
                    self.reconcile()
        result = self.status(held_digest)
        if op == 'brain_runtime_check':
            m, _, _ = self.release(seed)
            installed = result['installed']['version']
            result['available'] = dict(version=m['version'], notes=m['notes'], size=m['size'], draft=False,
                                       newer=not installed or version(m['version']) > version(installed))
        if seed:
            try:
                m, _, _ = self.release(seed)
                installed = result['installed']['version']
                result['seed'] = dict(version=m['version'], newer=not installed or version(m['version']) > version(installed), error=None)
            except (OSError, ValueError) as e:
                result['seed'] = dict(version='', newer=False, error=dict(code=getattr(e,'code','runtime_untrusted'),params={}))
        return result


def worker():
    Runtime(home()).run_job(Path(sys.argv[2]) if len(sys.argv)>2 else None)
