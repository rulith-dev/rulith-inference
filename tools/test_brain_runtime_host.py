import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import brain_runtime_host as rt
from brain_host_abi import STORE_LIMIT, ui_store, load_module


class TestSigner:
    """An ephemeral fixture signer, backed by the same local TLS library as the verifier."""
    def __init__(self):
        import _ssl
        if os.name=='nt':
            self.lib = ctypes.CDLL(str(Path(_ssl.__file__).with_name('libcrypto-3.dll')))
        else:
            from ctypes.util import find_library
            self.lib = ctypes.CDLL(find_library('crypto'))
        p,z = ctypes.c_void_p,ctypes.c_size_t
        def fn(name,ret,*args):
            f=getattr(self.lib,name)
            f.restype,f.argtypes=ret,list(args)
            return f
        self.key=fn('EVP_PKEY_new_raw_private_key_ex',p,p,p,p,p,z)(None,b'ED25519',None,os.urandom(32),32)
        self.free_key=fn('EVP_PKEY_free',None,p)
        self.free_ctx=fn('EVP_MD_CTX_free',None,p)
        out,n=ctypes.create_string_buffer(32),z(32)
        assert fn('EVP_PKEY_get_raw_public_key',ctypes.c_int,p,p,ctypes.POINTER(z))(self.key,out,ctypes.byref(n))==1
        self.public=out.raw
        self.new=fn('EVP_MD_CTX_new',p)
        self.init=fn('EVP_DigestSignInit',ctypes.c_int,p,p,p,p,p)
        self.final=fn('EVP_DigestSign',ctypes.c_int,p,p,ctypes.POINTER(z),p,z)

    def sign(self, raw):
        ctx=self.new()
        try:
            msg=rt.DOMAIN+hashlib.sha256(raw).digest()
            out,n=ctypes.create_string_buffer(64),ctypes.c_size_t(64)
            assert self.init(ctx,None,None,None,self.key)==1
            assert self.final(ctx,out,ctypes.byref(n),msg,len(msg))==1
            return out.raw
        finally:
            self.free_ctx(ctx)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir=Path(tmp.name)
        self.signer=TestSigner()
        self.addCleanup(self.signer.free_key,self.signer.key)
        pin=self.dir/'fixture.pub'
        pin.write_bytes(self.signer.public)
        self.runtime=rt.Runtime(self.dir/'home',pin,idle=lambda:True,smoke=lambda root:None)
        self.files={'llama-server.exe':b'fixture','ui/index.html':b'<script type="module" src="./app.mjs"></script>',
                    'ui/app.mjs':b'export default 1',rt.MODULE:b'HOST_ABI=1\ndef dispatch(op,data,host):return host.abi\n',
                    'EULA.md':b'test','THIRD-PARTY-NOTICES.md':b'test','licenses/notice':b'test'}
        self.zip=self.dir/'archive.zip'

    def release(self,v='0.2.2'):
        with zipfile.ZipFile(self.zip,'w') as z:
            for k,b in self.files.items():z.writestr(k,b)
        files={k:dict(size=len(b),sha256=hashlib.sha256(b).hexdigest()) for k,b in self.files.items()}
        tree=''.join(f"{files[n]['sha256']}  {n}\n" for n in sorted(files) if n.startswith('ui/'))
        m=dict(schema_version=2,version=v,tag='v'+v,file=f'rulith-brain-runtime-{v}.zip',size=self.zip.stat().st_size,
               sha256=rt.digest(self.zip),notes='',host_abi={'min':1,'max':1},capabilities=list(rt.CAPS),files=files,
               ui=dict(dir='ui',entry='index.html',sha256=hashlib.sha256(tree.encode()).hexdigest()),
               manager=dict(path=rt.MODULE,module='rulith_brain_manager',sha256=files[rt.MODULE]['sha256']))
        return m

    def stage(self,v='0.2.2'):
        raw=json.dumps(self.release(v)).encode()
        self.runtime.stage(self.zip,raw,self.signer.sign(raw))
        return raw

    def test_install_and_offline_module_load_with_tamper(self):
        r=self.runtime
        self.assertEqual(r.status()['installed']['reason'],'runtime_missing')
        self.stage()
        with r.generation():
            r.reconcile()
            self.assertEqual(r.status()['pending'],'0.2.2')
        r.reconcile()
        self.assertTrue(r.status()['installed']['verified'])
        _,h=r.verified(r.live)
        with r.generation():
            mod=load_module(r,h)
            self.assertEqual(mod.dispatch('brain_app',{},type('Host',(),{'abi':1})()),1)
            f=r.live/rt.MODULE
            data=f.read_bytes()
            f.write_bytes(data.replace(b'host.abi',b'host.xyz'))
            with self.assertRaises(rt.HostError):load_module(r,h)
        f.write_bytes(data)
        (r.live/'ui/app.mjs').write_bytes(b'export default 2')
        self.assertEqual(r.status()['installed']['reason'],'runtime_integrity')

    def test_manifest_auth_schema_abi_capabilities(self):
        m=self.release()
        for patch_,code in (({'schema_version':True},'runtime_schema'),({'schema_version':8},'runtime_schema'),
                            ({'capabilities':[]},'runtime_capability'),({'host_abi':{'min':2,'max':4}},'runtime_abi'),
                            ({'draft':True},'runtime_schema'),({'version':'00.2.2'},'runtime_schema')):
            b=json.dumps({**m,**patch_}).encode()
            with self.subTest(code=code),self.assertRaises(rt.HostError) as e:
                rt.parse(b,self.signer.sign(b),self.signer.public)
            self.assertEqual(e.exception.code,code)
        b=json.dumps(m).encode()
        for raw,sig in ((b+b' ',self.signer.sign(b)),(b,bytes(64))):
            with self.assertRaises(rt.HostError) as caught:rt.parse(raw,sig,self.signer.public)
            self.assertEqual(caught.exception.code,'runtime_untrusted')
        raw=b.replace(b'"schema_version": 2',b'"schema_version": 2,"schema_version": 2')
        with self.assertRaises(rt.HostError):rt.parse(raw,self.signer.sign(raw),self.signer.public)

    def test_staging_requires_complete_local_assets_and_windows_names(self):
        del self.files['ui/app.mjs']
        with self.assertRaises(rt.HostError):self.stage()
        for name in ('../bad','/asset','C:asset','ui/a.','ui/con.txt','A/file','ui\\bad',rt.MANIFEST,rt.SIG):
            if name=='A/file':
                self.files['a/other']=b'x'
            self.files[name]=b'x'
            raw=json.dumps(self.release()).encode()
            with self.subTest(name=name),self.assertRaises(rt.HostError):
                rt.parse(raw,self.signer.sign(raw),self.signer.public)
            del self.files[name]

    def test_interruption_and_no_downgrade_or_substitution(self):
        r=self.runtime
        self.stage();r.reconcile();self.stage('0.2.3')
        r.live.rename(r.old)
        r.reconcile()
        self.assertEqual(r.status()['installed']['version'],'0.2.2')
        r.reconcile()
        self.assertEqual(r.status()['installed']['version'],'0.2.3')
        with self.assertRaises(rt.HostError):self.stage()
        self.files['ui/app.mjs']=b'export default 3'
        with self.assertRaises(rt.HostError):self.stage('0.2.3')

    def test_legacy_and_damaged_proof(self):
        r=self.runtime
        r.live.mkdir(parents=True)
        (r.live/'llama-server.exe').write_bytes(b'old')
        (r.live/rt.MANIFEST).write_text('{"version":"0.2.1"}')
        self.assertEqual(r.status()['installed']['reason'],'runtime_legacy')
        self.stage();r.reconcile()
        (r.live/rt.SIG).unlink()
        self.assertEqual(r.status()['installed']['reason'],'runtime_untrusted')

    def test_setup_never_executes_module_and_foreign_engine_defers_swap(self):
        sentinel=self.dir/'imported'
        self.files[rt.MODULE]=f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n".encode()
        self.stage()
        r=self.runtime
        r.idle=lambda:False
        pending=r.operation('brain_runtime',{})
        self.assertEqual(pending['pending'],'0.2.2')
        self.assertFalse(r.live.exists())
        r.idle=lambda:True
        self.assertTrue(r.operation('brain_runtime',{})['installed']['verified'])
        self.assertFalse(sentinel.exists())

    def test_worker_resumes_pinned_archive_and_reopens_offline(self):
        r=self.runtime
        m=self.release()
        raw=json.dumps(m).encode()
        sig=self.signer.sign(raw)
        archive=self.zip.read_bytes()
        split=len(archive)//2
        r.cache.mkdir(parents=True)
        (r.cache/(m['sha256']+'.part')).write_bytes(archive[:split])
        seen=[]
        def fake_open(req,timeout):
            if isinstance(req,str):
                return io.BytesIO(sig if req.endswith(rt.SIG) else raw)
            seen.append((req.full_url,req.get_header('Range')))
            response=io.BytesIO(archive[split:])
            response.status=206
            response.headers={'Content-Range':f'bytes {split}-{len(archive)-1}/{len(archive)}'}
            return response
        with patch('urllib.request.urlopen',side_effect=fake_open):r.run_job()
        self.assertEqual(seen,[(rt.CHANNEL+'download/v0.2.2/'+m['file'],f'bytes={split}-')])
        with patch('urllib.request.urlopen',side_effect=AssertionError('unexpected connection')):
            state=r.operation('brain_runtime',{})
            self.assertEqual(state['job']['status'],'done')
            self.assertTrue(state['installed']['verified'])

    def test_bad_archive_and_missing_or_extra_zip_member_leave_live(self):
        r=self.runtime
        self.stage();r.reconcile()
        m=self.release('0.2.3')
        with self.zip.open('ab') as f:f.write(b'x')
        raw=json.dumps(m).encode()
        with self.assertRaises(rt.HostError):r.stage(self.zip,raw,self.signer.sign(raw))
        for extra in ('unlisted.dat',rt.MANIFEST,rt.SIG):
            m=self.release('0.2.3')
            with zipfile.ZipFile(self.zip,'a') as f:f.writestr(extra,b'x')
            m.update(size=self.zip.stat().st_size,sha256=rt.digest(self.zip))
            raw=json.dumps(m).encode()
            with self.assertRaises(rt.HostError):r.stage(self.zip,raw,self.signer.sign(raw))
        self.assertEqual(r.status()['installed']['version'],'0.2.2')

    def test_store_cas_tombstone_preserves_prior_bytes_on_disk_full(self):
        d=self.dir
        read=lambda:ui_store(d,'ui_store_read',{'name':'conversation'})
        write=lambda rev,v:ui_store(d,'ui_store_write',dict(name='conversation',expected_rev=rev,value=v))
        self.assertEqual(read(),dict(rev=0,value=None))
        value=dict(id='A',messages=[dict(role='user',unsent=True)])
        self.assertEqual(write(0,value),{'rev':1})
        with self.assertRaises(rt.HostError) as e:write(0,None)
        self.assertEqual(e.exception.params,{'rev':1})
        with patch('brain_runtime_host.os.replace',side_effect=OSError('disk full')),self.assertRaises(rt.HostError):write(1,None)
        self.assertEqual(read(),dict(rev=1,value=value))
        write(1,None)
        self.assertEqual(read(),dict(rev=2,value=None))
        for raw,code in ((b'{','store_corrupt'),(b'{"schema_version":2}','store_version')):
            (d/'ui-state.json').write_bytes(raw)
            with self.assertRaises(rt.HostError) as e:read()
            self.assertEqual(e.exception.code,code)
            self.assertEqual((d/'ui-state.json').read_bytes(),raw)

    def test_store_byte_boundary(self):
        base=len(b'{"persona":""}')
        for i,n in enumerate((STORE_LIMIT-1,STORE_LIMIT,STORE_LIMIT+1)):
            d=dict(name='settings',expected_rev=i,value={'persona':'a'*(n-base)})
            if i==2:
                with self.assertRaisesRegex(rt.HostError,'too large'):ui_store(self.dir,'ui_store_write',d)
            else:self.assertEqual(ui_store(self.dir,'ui_store_write',d),{'rev':i+1})

    def test_manager_wire_limit_and_plain_start(self):
        cmd=[sys.executable,'-I','-S','-B',str(Path(__file__).with_name('manager.py'))]
        for n in (65535,65536,65537):
            raw=b'{"op":"unknown"}'.ljust(n,b' ')
            p=subprocess.run(cmd,input=raw,capture_output=True)
            self.assertEqual(json.loads(p.stdout)['code'],'request_too_large' if n>65536 else 'unknown_op')
        for raw,code in ((b'{"op":"start"}','owner_required'),(b'{"op":"status","data":[]}','request_invalid')):
            p=subprocess.run(cmd,input=raw,capture_output=True)
            self.assertEqual(json.loads(p.stdout)['code'],code)
