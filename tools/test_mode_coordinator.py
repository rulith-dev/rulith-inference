import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from brain_runtime_host import HostError, lease
from mode_coordinator import Coordinator, Engines, identity


def request_id():
    return str(uuid.uuid4())


class FakeEngines:
    def __init__(self):
        self.mode=None
        self.record=None
        self.events=[]
        self.fail=None
        self.owner=dict(host='inference',install_id=request_id(),instance_id=request_id())
        self.start('inference',dict(model_id='original'))

    def event(self,event):
        self.events.append(event)
        if self.fail==event:raise HostError({'save':'persist_failed','stop':'stop_timeout'}.get(event,'launch_failed'))

    def current(self):return self.mode,self.record
    def check(self,mode,record):
        if (mode,record)!=(self.mode,self.record):raise HostError('owner_busy')
    def snapshot(self,mode,record):
        return dict(mode=mode,model_id=(record or {}).get('model_id'),profile={'prompt_cache_disk':True},
                    cache_dir='cache',model_path='model.gguf',memory_dir=None,runtime_sha256=None)
    def preflight(self,mode,dest):self.event('preflight')
    def admission(self,mode,record,tid,admit):
        self.event('open' if admit else 'close')
        return dict(admit=admit,in_flight=0)
    def save(self,mode,record):self.event('save')
    def stop(self,mode,record):
        self.event('stop')
        self.check(mode,record)
        self.mode=self.record=None
    def prepare_launch(self,tid):self.launch_id=tid
    def start(self,mode,dest):
        assert self.mode is None, 'overlapping processes'
        self.event('start')
        self.mode=mode
        self.record=dict(owner=self.owner,launch_id=getattr(self,'launch_id',request_id()),identity=dict(pid=25,birth=134420001000000001,exe='fixture.exe'),
                         model_id=dest['model_id'] if dest and dest.get('model_id') else 'brain-preset')
        return self.record
    def ready(self,mode,record,tid):self.event('ready')
    def reattach(self,mode,record,owner):self.owner=record['owner']=owner


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.dir=Path(tmp.name)
        self.e=FakeEngines()
        self.c=Coordinator(self.dir,self.e.owner,self.e)
        self.e.events.clear()

    def switch(self,target='brain'):
        self.id=request_id()
        j,fresh=self.c.accept('mode_switch_'+target,{'request_id':self.id})
        self.assertTrue(fresh)
        self.assertEqual(j['state'],'checking')
        return self.c.run()

    def retry(self,action):
        self.c.accept('mode_retry',dict(request_id=request_id(),transition_id=self.id,action=action))
        return self.c.run()

    def test_no_overlap_return_snapshot_and_retry_deduplication(self):
        result=self.switch()
        self.assertEqual(self.e.events,['preflight','close','save','stop','start','ready','open'])
        self.assertEqual((result['active'],result['selected']),('brain','brain'))
        self.assertEqual(self.c.read()['dest_identity']['birth'],'134420001000000001')
        self.assertFalse(self.c.accept('mode_switch_brain',dict(request_id=self.id))[1])
        original=self.id
        with self.assertRaises(HostError):self.c.accept('mode_switch_inference',dict(request_id=self.id))
        self.assertEqual(self.switch('inference')['active'],'inference')
        self.assertEqual(self.e.record['model_id'],'original')
        self.assertIsNone(self.c.read()['inference_return'])
        snapshot,fresh=self.c.accept('mode_switch_brain',dict(request_id=original))
        self.assertFalse(fresh)
        self.assertEqual(self.c.status(snapshot),result)

    def test_failures_at_each_barrier_and_retry(self):
        for step,state,live in [('preflight','blocked','inference'),('close','blocked','inference'),
                                ('save','save_failed','inference'),('stop','blocked','inference'),
                                ('start','launch_failed','none'),('ready','blocked','brain')]:
            with self.subTest(step=step):
                self.c.path.unlink(missing_ok=True);self.e=FakeEngines();self.c=Coordinator(self.dir,self.e.owner,self.e)
                self.e.fail=step
                r=self.switch()
                self.assertEqual((r['state'],r['active']),(state,live))
                if step=='save':self.assertEqual(self.e.events[-1],'open')
                if step=='stop':self.assertEqual(r['error']['code'],'stop_timeout')
                self.e.fail=None
                self.assertEqual(self.retry('retry')['active'],'brain')

    def test_recovery_at_each_recorded_phase_leaves_source_alive(self):
        for phase in ('checking','draining','persisting_inference','saving_brain','stopping','starting','waiting_ready'):
            with self.subTest(phase=phase):
                self.c.path.unlink(missing_ok=True)
                tid=request_id()
                self.c.accept('mode_switch_brain',{'request_id':tid})
                self.c.update(state=phase,source='inference',source_identity=identity(self.e.record))
                recovered=Coordinator(self.dir,dict(self.e.owner,instance_id=request_id()),self.e)
                recovered.recover()
                self.assertEqual(recovered.status()['state'],'blocked')
                self.assertEqual(self.e.mode,'inference')
                self.assertNotIn('stop',self.e.events)

    def test_matching_destination_recovery_does_not_launch_again(self):
        self.e.fail='ready'
        self.switch()
        self.c.update(state='waiting_ready')
        recovered=Coordinator(self.dir,dict(self.e.owner,instance_id=request_id()),self.e)
        self.e.fail=None
        recovered.recover()
        self.assertEqual(recovered.status()['state'],'brain_ready')
        self.assertEqual(self.e.events.count('start'),1)

    def test_restore_after_launch_failure(self):
        self.e.fail='start'
        self.assertEqual(self.switch()['state'],'launch_failed')
        self.e.fail=None
        r=self.retry('restore')
        self.assertEqual((r['selected'],self.e.record['model_id']),('inference','original'))

    def test_owner_lock_excludes_an_independent_process(self):
        path=self.dir/'engine-owner.lock'
        script="import sys;sys.path.insert(0,sys.argv[1]);from brain_runtime_host import lease;from pathlib import Path\nwith lease(Path(sys.argv[2]),wait=False) as held:print(int(held))"
        cmd=[sys.executable,'-I','-B','-c',script,str(Path(__file__).parent),str(path)]
        with lease(path):
            self.assertEqual(subprocess.check_output(cmd).strip(),b'0')
        self.assertEqual(subprocess.check_output(cmd).strip(),b'1')
        self.assertTrue(path.exists())

    def test_two_desktop_processes_race_to_launch(self):
        script="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from brain_runtime_host import lease
folder=Path(sys.argv[2])
print('ready',flush=True)
sys.stdin.readline()
with lease(folder/'engine-owner.lock',wait=False) as held:
    if held:
        with (folder/'launches').open('a') as f:f.write('start\\n')
    print('started' if held else 'busy',flush=True)
    sys.stdin.readline()
"""
        cmd=[sys.executable,'-I','-B','-c',script,str(Path(__file__).parent),str(self.dir)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as a, \
             subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as b:
            try:
                for child in (a,b):self.assertEqual(child.stdout.readline().strip(),'ready')
                for child in (a,b):child.stdin.write('go\n');child.stdin.flush()
                outcomes=[child.stdout.readline().strip() for child in (a,b)]
                self.assertCountEqual(outcomes,['started','busy'])
                self.assertEqual((self.dir/'launches').read_text(),'start\n')
            finally:
                for child in (a,b):child.stdin.close()

    def test_save_waits_for_forwarded_request_and_excludes_new_calls(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        path = self.dir/'frame-requests.lock'
        draining = Event()
        update = self.c.update
        def observe(**fields):
            result = update(**fields)
            if fields.get('state') == 'draining':
                draining.set()
            return result
        def save(mode, record):
            with lease(path, shared=True, wait=False) as admitted:
                self.assertFalse(admitted)
        with patch.object(self.c, 'update', side_effect=observe), patch.object(self.e, 'save', side_effect=save) as saved:
            with ThreadPoolExecutor(1) as pool:
                with lease(path, shared=True):
                    result = pool.submit(self.switch)
                    self.assertTrue(draining.wait(5))
                    saved.assert_not_called()
                self.assertEqual(result.result(timeout=5)['state'], 'brain_ready')
            saved.assert_called_once()
        with lease(path, shared=True, wait=False) as admitted:
            self.assertTrue(admitted)

    def test_brain_drain_counts_slots_and_queued_work_without_control_route(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        manager = Mock(DATA=self.dir)
        adapter = Engines(manager, SimpleNamespace(config_dir=self.dir), {})
        with patch.object(adapter, 'endpoint', return_value=('http://127.0.0.1', {})):
            for raw, waiting, count in [([{'is_processing': True}], 2, 3), ([{'is_processing': False}], 0, 0),
                                        ([], 0, None), ([{}], 0, None)]:
                body = io.BytesIO(json.dumps(raw).encode())
                body.headers = {'X-Strix-Waiting': str(waiting)}
                manager.HTTP.open.return_value = body
                if count is None:
                    with self.assertRaises(HostError):
                        adapter.admission('brain', {}, 'switch', False)
                else:
                    self.assertEqual(adapter.admission('brain', {}, 'switch', False), dict(admit=False, in_flight=count))
                self.assertEqual(manager.HTTP.open.call_args.args[0].full_url, 'http://127.0.0.1/slots')
            manager.HTTP.open.reset_mock()
            self.assertEqual(adapter.admission('brain', {}, 'switch', True), dict(admit=True, in_flight=0))
            manager.HTTP.open.assert_not_called()

    def test_live_persist_acknowledgement(self):
        import http.server
        import threading
        import manager
        from types import SimpleNamespace
        current=os.urandom(16).hex()
        admin=os.urandom(16).hex()
        outcomes=[]
        saved=[True,False]
        class Server(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                correct=(self.path=='/strix/persist' and self.headers.get('Authorization')=='Bearer '+current
                         and self.headers.get('X-Rulith-Admin')==admin and self.rfile.read(2)==b'{}')
                outcomes.append(correct)
                self.send_response(200 if correct else 401);self.end_headers()
                self.wfile.write(json.dumps(dict(success=saved.pop(0))).encode())
        with http.server.HTTPServer(('127.0.0.1',0),Server) as srv:
            thread=threading.Thread(target=srv.serve_forever);thread.start()
            try:
                adapter=Engines(manager,SimpleNamespace(config_dir=self.dir,runtime_dir=self.dir),{'owner':self.e.owner})
                record=dict(identity=manager.process_identity(os.getpid()),owner=self.e.owner,launch_id=request_id(),
                            port=srv.server_port,host='127.0.0.1',admin_token=admin,api_key=current,profile={'prompt_cache_disk':True})
                with patch.object(adapter,'current',return_value=('inference',record)):
                    adapter.save('inference',record)
                    with self.assertRaises(HostError) as e:adapter.save('inference',record)
                    self.assertEqual(e.exception.code,'persist_failed')
                self.assertEqual(outcomes,[True,True])
            finally:srv.shutdown();thread.join()

    def test_reused_pid_and_foreign_candidate(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        m=Mock()
        m.DATA=self.dir;m.RUNTIME=self.dir/'engine.exe'
        h=SimpleNamespace(config_dir=self.dir/'brain',runtime_dir=self.dir/'bin')
        adapter=Engines(m,h,{'owner':self.e.owner})
        record=self.e.record
        m.read_json.side_effect=lambda p,default:record if p.parent==self.dir else {}
        m.process_identity.return_value=dict(record['identity'],birth=record['identity']['birth']+1)
        m.discover.return_value=[]
        self.assertEqual(adapter.current(),(None,None))
        m.discover.return_value=[dict(ProcessId=25,ExecutablePath='elsewhere.exe')]
        with self.assertRaises(HostError) as e:adapter.current()
        self.assertEqual(e.exception.code,'owner_busy')
        self.assertFalse(any(len(call.args)>1 for call in m.process_identity.call_args_list))
