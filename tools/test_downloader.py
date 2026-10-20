"""Tests of tools/downloader.py and the manager's download ops against a local HTTP server (plain http to 127.0.0.1,
allowed through RULITH_DL_TEST_HOSTS for tests only).

    python tools/test_downloader.py
"""
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ['RULITH_DL_TEST_HOSTS'] = '127.0.0.1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import downloader as D  # noqa: E402


class Server:
    """files by path; per path options: range (default True), drop (drop each range's first try halfway),
    corrupt (flip the byte at this offset), status (answer it), redirect (Location), delay (seconds per 16 KiB)"""

    def __init__(self):
        self.files, self.opts, self.served, self.dropped, self.requests = {}, {}, 0, set(), []
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split('?')[0]
                srv.requests.append((path, self.headers.get('Range')))
                o = srv.opts.get(path, {})
                if o.get('status'):
                    self.send_response(o['status']); self.send_header('Content-Length', '0'); self.end_headers(); return
                if o.get('redirect'):
                    self.send_response(302); self.send_header('Location', o['redirect'])
                    self.send_header('Content-Length', '0'); self.end_headers(); return
                if path in srv.json:
                    body, link = srv.json[path]
                    self.send_response(200); self.send_header('Content-Type', 'application/json')
                    if link: self.send_header('Link', link)
                    self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body); return
                data = srv.files.get(path)
                if data is None:
                    self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
                if o.get('corrupt') is not None:
                    b = bytearray(data); b[o['corrupt']] ^= 0xFF; data = bytes(b)
                rng = self.headers.get('Range')
                a, b = 0, len(data)
                if rng and o.get('range', True):
                    x, y = rng.split('=')[1].split('-')
                    a, b = int(x), min(len(data), int(y) + 1)
                    self.send_response(206)
                    self.send_header('Content-Range', f'bytes {a}-{b - 1}/{len(data)}')
                else:
                    self.send_response(200)
                self.send_header('Content-Length', str(b - a)); self.end_headers()
                cut = b
                if o.get('drop') and (path, a) not in srv.dropped:
                    srv.dropped.add((path, a)); cut = a + (b - a) // 2
                pos = a
                try:
                    while pos < cut:
                        n = min(16384, cut - pos)
                        self.wfile.write(data[pos:pos + n]); pos += n; srv.served += n
                        if o.get('delay'): time.sleep(o['delay'])
                except (ConnectionError, OSError):
                    return
                if cut < b: self.close_connection = True

        self.json = {}
        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.httpd.daemon_threads = True
        self.base = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


def blob(n, seed=1):
    return random.Random(seed).randbytes(n)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.srv = Server(); self.addCleanup(self.srv.close)
        for k, v in dict(CHUNK=64 << 10, READ=16 << 10, STATE_EVERY=0.05).items():
            p = patch.object(D, k, v); p.start(); self.addCleanup(p.stop)
        p = patch.object(D.time, 'sleep', lambda s: None) if False else None

    def job(self, files, conns=4):
        jd = self.root / 'job'
        jd.mkdir(exist_ok=True)
        D.write_json(jd / 'job.json', {'id': 'j', 'folder': str(self.root / 'out'), 'connections': conns, 'files': files})
        return jd

    def spec(self, name, data, *paths):
        return {'name': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'urls': [self.srv.base + p for p in paths]}

    def run_job(self, jd):
        return D.Job(jd).run()

    def state(self, jd):
        return json.loads((jd / 'state.json').read_text(encoding='utf-8'))


class WorkerTests(Base):
    def test_multi_chunk(self):
        data = blob((1 << 20) + 123)
        self.srv.files['/a.gguf'] = data
        jd = self.job([self.spec('a.gguf', data, '/a.gguf')])
        self.assertEqual(self.run_job(jd), 'done')
        out = self.root / 'out' / 'a.gguf'
        self.assertEqual(out.read_bytes(), data)
        self.assertFalse((self.root / 'out' / 'a.gguf.part').exists())
        self.assertTrue(self.state(jd)['files']['a.gguf']['verified'])
        self.assertGreater(len(self.srv.requests), 10)                    # ranged, not one stream

    def test_window_stays_near_the_front(self):
        data = blob(2 << 20)
        self.srv.files['/w.gguf'] = data
        jd = self.job([self.spec('w.gguf', data, '/w.gguf')], conns=2)
        self.assertEqual(self.run_job(jd), 'done')
        starts = [int(r.split('=')[1].split('-')[0]) // D.CHUNK for _, r in self.srv.requests if r]
        for k, c in enumerate(starts):                                   # never more than conns*WINDOW ahead
            self.assertLess(c - k, 2 * D.WINDOW + 1)

    def test_resume_after_stop(self):
        data = blob(1 << 20)
        self.srv.files['/r.gguf'] = data
        self.srv.opts['/r.gguf'] = {'delay': 0.01}
        jd = self.job([self.spec('r.gguf', data, '/r.gguf')], conns=2)
        t = threading.Thread(target=self.run_job, args=(jd,)); t.start()
        for _ in range(500):
            time.sleep(0.02)
            st = D.read_json(jd / 'state.json', {}) or {}
            if len(st.get('files', {}).get('r.gguf', {}).get('done', [])) >= 4: break
        D.write_json(jd / 'control.json', {'stop': True}); t.join()
        self.assertEqual(self.state(jd)['status'], 'stopped')
        part = self.root / 'out' / 'r.gguf.part'
        self.assertTrue(part.exists())
        if os.name == 'nt':
            import ctypes
            self.assertTrue(ctypes.windll.kernel32.GetFileAttributesW(str(part)) & 0x200)   # sparse while partial
        first = self.srv.served
        (jd / 'control.json').unlink()
        self.srv.opts['/r.gguf'] = {}
        self.assertEqual(self.run_job(jd), 'done')
        self.assertEqual((self.root / 'out' / 'r.gguf').read_bytes(), data)
        self.assertLess(self.srv.served - first, len(data) - 3 * D.CHUNK)  # the finished chunks were kept
        if os.name == 'nt':
            import ctypes
            self.assertFalse(ctypes.windll.kernel32.GetFileAttributesW(str(self.root / 'out' / 'r.gguf')) & 0x200)

    def test_dropped_connections_retry(self):
        data = blob(600 << 10)
        self.srv.files['/d.gguf'] = data
        self.srv.opts['/d.gguf'] = {'drop': True}
        jd = self.job([self.spec('d.gguf', data, '/d.gguf')])
        with patch.object(D.Job, 'check_control', lambda self: None), patch.object(D, 'FIRST_TRIES', 3):
            self.assertEqual(self.run_job(jd), 'done')
        self.assertEqual((self.root / 'out' / 'd.gguf').read_bytes(), data)

    def test_hash_mismatch_moves_to_the_next_source(self):
        data = blob(300 << 10)
        self.srv.files['/bad.gguf'] = data; self.srv.files['/good.gguf'] = data
        self.srv.opts['/bad.gguf'] = {'corrupt': 1000}
        jd = self.job([self.spec('h.gguf', data, '/bad.gguf', '/good.gguf')])
        self.assertEqual(self.run_job(jd), 'done')
        self.assertEqual((self.root / 'out' / 'h.gguf').read_bytes(), data)
        self.assertEqual(self.state(jd)['files']['h.gguf']['source'], 1)

    def test_hash_mismatch_only_source_fails(self):
        data = blob(200 << 10)
        self.srv.files['/bad.gguf'] = data
        self.srv.opts['/bad.gguf'] = {'corrupt': 5}
        jd = self.job([self.spec('h.gguf', data, '/bad.gguf')])
        self.assertEqual(self.run_job(jd), 'failed')
        self.assertEqual(self.state(jd)['error']['code'], 'hash_mismatch')
        self.assertFalse(any((self.root / 'out').iterdir()))

    def test_404_moves_to_the_next_source(self):
        data = blob(100 << 10)
        self.srv.files['/ok.gguf'] = data
        jd = self.job([self.spec('n.gguf', data, '/missing.gguf', '/ok.gguf')])
        self.assertEqual(self.run_job(jd), 'done')

    def test_server_without_ranges(self):
        data = blob(500 << 10)
        self.srv.files['/nr.gguf'] = data
        self.srv.opts['/nr.gguf'] = {'range': False}
        jd = self.job([self.spec('nr.gguf', data, '/nr.gguf')])
        self.assertEqual(self.run_job(jd), 'done')
        self.assertEqual((self.root / 'out' / 'nr.gguf').read_bytes(), data)

    def test_redirect_to_another_host_refused(self):
        data = blob(100 << 10)
        self.srv.files['/ok.gguf'] = data
        self.srv.opts['/r.gguf'] = {'redirect': 'http://example.com/x.gguf'}
        jd = self.job([self.spec('x.gguf', data, '/r.gguf', '/ok.gguf')])
        self.assertEqual(self.run_job(jd), 'done')
        self.assertEqual(self.state(jd)['files']['x.gguf']['last_error']['code'], 'redirect_refused')

    def test_existing_file_is_never_touched(self):
        data = blob(100 << 10)
        self.srv.files['/e.gguf'] = data
        out = self.root / 'out'; out.mkdir()
        mine = blob(100 << 10, seed=7)
        (out / 'e.gguf').write_bytes(mine)
        before = (out / 'e.gguf').stat().st_mtime_ns
        jd = self.job([self.spec('e.gguf', data, '/e.gguf')])
        self.assertEqual(self.run_job(jd), 'done')                       # same size: taken as it is
        self.assertEqual((out / 'e.gguf').read_bytes(), mine)
        self.assertEqual((out / 'e.gguf').stat().st_mtime_ns, before)
        (out / 'f.gguf').write_bytes(b'short')
        self.srv.files['/f.gguf'] = data
        jd2 = self.job([self.spec('f.gguf', data, '/f.gguf')])
        (jd2 / 'state.json').unlink(missing_ok=True)
        self.assertEqual(self.run_job(jd2), 'failed')
        self.assertEqual(self.state(jd2)['error']['code'], 'name_taken')
        self.assertEqual((out / 'f.gguf').read_bytes(), b'short')


class RepoTests(Base):
    def test_parse_repo(self):
        self.assertEqual(D.parse_repo('https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF'),
                         ('hf', 'unsloth', 'Qwen3.8-Flash-Next-GGUF', 'main'))
        self.assertEqual(D.parse_repo('https://hf-mirror.com/a/b/tree/dev/sub'), ('hfm', 'a', 'b', 'dev'))
        self.assertEqual(D.parse_repo('a/b', 'hfm'), ('hfm', 'a', 'b', 'main'))
        for bad in ('https://evil.com/a/b', 'https://huggingface.co/datasets/a/b', 'a/b/c', '../x', ''):
            with self.assertRaises(D.DownloadError): D.parse_repo(bad)

    def test_safe_name(self):
        self.assertEqual(D.safe_name('dir/sub/x-Q4_K_M.gguf'), 'x-Q4_K_M.gguf')
        for bad in ('x.bin', '../..', 'CON.gguf', ''):
            with self.assertRaises(D.DownloadError): D.safe_name(bad)

    def test_list_repo_groups_and_pages(self):
        cat = {'sources': {'t': {'name': 'test', 'kind': 'hf', 'base': self.srv.base}}}
        sha = 'a' * 40
        self.srv.json['/api/models/o/r/revision/main'] = (json.dumps({'sha': sha}).encode(), None)
        page1 = [{'type': 'file', 'path': 'Q4/m-Q4_K_M-00001-of-00002.gguf', 'size': 10, 'lfs': {'oid': 'b' * 64, 'size': 10}},
                 {'type': 'file', 'path': 'README.md', 'size': 3}]
        page2 = [{'type': 'file', 'path': 'Q4/m-Q4_K_M-00002-of-00002.gguf', 'size': 20, 'lfs': {'oid': 'c' * 64, 'size': 20}},
                 {'type': 'file', 'path': 'mmproj-F16.gguf', 'size': 5},
                 {'type': 'file', 'path': 'm-Q8_0.gguf', 'size': 30, 'lfs': {'oid': 'd' * 64, 'size': 30}}]
        self.srv.json[f'/api/models/o/r/tree/{sha}'] = (json.dumps(page1).encode(), f'<{self.srv.base}/p2>; rel="next"')
        self.srv.json['/p2'] = (json.dumps(page2).encode(), None)
        listing = D.list_repo(cat, 't', 'o', 'r')
        self.assertEqual(listing['commit'], sha)
        g = {x['name']: x for x in listing['groups']}
        self.assertEqual(g['m-Q4_K_M']['size'], 30)
        self.assertTrue(g['m-Q4_K_M']['complete'])
        self.assertEqual(g['m-Q4_K_M']['quant'], 'Q4_K_M')
        self.assertEqual(g['mmproj-F16']['role'], 'projection')
        self.assertIsNone(g['mmproj-F16']['files'][0]['sha256'])
        urls = D.repo_urls(cat, 't', 'o', 'r', sha, 'Q4/m-Q4_K_M-00001-of-00002.gguf')
        self.assertEqual(urls, [f'{self.srv.base}/o/r/resolve/{sha}/Q4/m-Q4_K_M-00001-of-00002.gguf'])

    def test_gated_repo_refused(self):
        cat = {'sources': {'t': {'name': 'test', 'kind': 'hf', 'base': self.srv.base}}}
        self.srv.opts['/api/models/o/g/revision/main'] = {'status': 401}
        with self.assertRaises(D.DownloadError) as c: D.list_repo(cat, 't', 'o', 'g')
        self.assertEqual(c.exception.code, 'repo_unavailable')

    def test_catalog_urls(self):
        cat = D.load_catalog()
        m = next(x for x in cat['models'] if x['owner'] == 'unsloth')
        head = next(f for f in m['files'] if f['role'] == 'head')
        model = m['files'][0]
        self.assertTrue(D.entry_urls(cat, model, 'hfm')[0].startswith('https://hf-mirror.com/unsloth/'))
        self.assertTrue(D.entry_urls(cat, head, 'hfm')[0].startswith('https://gh-proxy.com/https://github.com/'))
        self.assertTrue(all(D.allowed(u) for f in m['files'] for u in D.entry_urls(cat, f, 'hf')))
        self.assertTrue(all(D.allowed(u) for x in cat['models'] for f in x['files'] for u in D.entry_urls(cat, f, 'hfm')))


@unittest.skipUnless(os.name == 'nt', 'the manager runs on Windows')
class ManagerDownloadTests(Base):
    def setUp(self):
        super().setUp()
        spec = importlib.util.spec_from_file_location('manager', HERE / 'manager.py')
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        p = patch.object(self.m, 'DATA', self.root / 'state'); p.start(); self.addCleanup(p.stop)
        (self.root / 'models').mkdir()
        self.m.atomic_json(self.m.DATA / 'settings.json', {'roots': [str(self.root / 'models')], 'profiles': {}})
        p = patch.object(self.m, 'lmstudio_models_dir', lambda: None); p.start(); self.addCleanup(p.stop)
        self.data = blob(300 << 10)
        self.srv.files['/o/r/resolve/rev/m.gguf'] = self.data
        cat = {'version': 1, 'sources': {'t': {'name': 'test', 'kind': 'hf', 'base': self.srv.base}},
               'models': [{'id': 'tm', 'name': 'Test', 'variant': 'Q', 'owner': 'o', 'repo': 'r', 'license': '',
                           'capabilities': [], 'files': [{'name': 'm.gguf', 'role': 'model', 'required': True,
                                                          'size': len(self.data),
                                                          'sha256': hashlib.sha256(self.data).hexdigest(),
                                                          'hf': 'o/r@rev:m.gguf'}]}]}
        (self.root / 'catalog.json').write_text(json.dumps(cat), encoding='utf-8')
        p = patch.object(D, 'CATALOG', self.root / 'catalog.json'); p.start(); self.addCleanup(p.stop)
        self.m._dl()
        for job in self.m.dl_job_dirs(): self.m.dl_stop(job)

    def wait(self, job_id, want=('done', 'failed'), timeout=60):
        for _ in range(int(timeout * 10)):
            v = self.m.dl_view(self.m.dl_job(job_id))
            if v['status'] in want and not v['alive']: return v
            time.sleep(0.1)
        self.fail('no %s in %ds: %s' % (want, timeout, v))

    def test_catalog_download_end_to_end(self):
        out = self.root / 'dl'
        v = self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': str(out)})
        self.addCleanup(lambda: [self.m.dl_stop(j) for j in self.m.dl_job_dirs()])
        v = self.wait(v['id'])
        self.assertEqual(v['status'], 'done', v)
        self.assertEqual((out / 'm.gguf').read_bytes(), self.data)
        self.assertIn(str(out.resolve()), self.m.settings()['roots'])
        c = self.m.dl_catalog()
        self.assertEqual(Path(c['models'][0]['files'][0]['present']).resolve(), (out / 'm.gguf').resolve())
        with self.assertRaises(self.m.ManagerError) as e:
            self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': str(out)})
        self.assertEqual(e.exception.code, 'dl_nothing')

    def test_one_at_a_time_pause_resume_cancel(self):
        self.srv.opts['/o/r/resolve/rev/m.gguf'] = {'delay': 0.05}
        out = self.root / 'dl'
        v = self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': str(out)})
        self.addCleanup(lambda: [self.m.dl_stop(j) for j in self.m.dl_job_dirs()])
        with self.assertRaises(self.m.ManagerError) as e:
            self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': str(self.root / 'dl2')})
        self.assertEqual(e.exception.code, 'dl_running')
        j = self.m.dl_job(v['id'])
        self.m.dl_stop(j)
        self.assertEqual(self.m.dl_view(j)['status'], 'stopped')
        self.srv.opts['/o/r/resolve/rev/m.gguf'] = {}
        self.m.dl_resume(j)
        self.assertEqual(self.wait(v['id'])['status'], 'done')
        self.m.dl_cancel(j)
        self.assertFalse(j.exists())
        self.assertTrue((out / 'm.gguf').exists())                        # cancel removes .part files only

    def test_space_and_folder_refusals(self):
        with patch.object(D, 'free_bytes', lambda p: 1000):
            with self.assertRaises(self.m.ManagerError) as e:
                self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': str(self.root / 'dl')})
            self.assertEqual(e.exception.code, 'dl_space')
        with self.assertRaises(self.m.ManagerError) as e:
            self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': 'relative/dir'})
        self.assertEqual(e.exception.code, 'dl_folder')
        with patch.object(self.m, 'lmstudio_models_dir', lambda: str(self.root / 'lms')):
            with self.assertRaises(self.m.ManagerError) as e:
                self.m.dl_start({'kind': 'catalog', 'model': 'tm', 'source': 't', 'folder': str(self.root / 'lms' / 'x')})
            self.assertEqual(e.exception.code, 'dl_folder_lmstudio')


if __name__ == '__main__':
    unittest.main()
