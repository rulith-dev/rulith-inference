"""Durable engine handoff. The desktop instance holds the owner lease across worker calls."""
import json
import os
from pathlib import Path
import time
import uuid
import urllib.request

from brain_runtime_host import HostError, lease, require, write_json

OPS = frozenset(('mode_status', 'mode_switch_brain', 'mode_switch_inference', 'mode_retry'))
BUSY = frozenset('checking draining persisting_inference saving_brain stopping starting waiting_ready recovering'.split())


def initial():
    return dict(schema_version=1, selected='inference', source=None, target=None, state='idle', transition_id=None,
                owner=None, source_identity=None, dest_identity=None, restore=None, inference_return=None, error=None,
                requests=[])


def uid(s):
    try:
        v = uuid.UUID(s)
        return v.version == 4 and str(v) == s
    except (TypeError, AttributeError, ValueError):
        return False


def identity(record):
    if not record:
        return None
    return dict(record['identity'], birth=str(record['identity']['birth']), launch_id=record.get('launch_id'))


class Coordinator:
    def __init__(self, directory, owner, engines):
        self.directory, self.owner, self.engines = directory, owner, engines
        self.path = directory/'mode.json'

    def read(self):
        try:
            j = json.loads(self.path.read_bytes())
            require(j['schema_version'] == 1 and j['selected'] in ('inference','brain'), 'owner_unknown')
            return j
        except FileNotFoundError:
            return initial()

    def update(self, **fields):
        previous=self.read()
        changing=fields.get('transition_id',previous['transition_id'])!=previous['transition_id']
        snapshot=self.status(previous) if changing and previous['transition_id'] and previous['state'] not in BUSY else None
        with lease(self.directory/'mode.lock'):
            j = self.read()
            if snapshot and j==previous:
                self.archive(j,snapshot)
            j.update(fields)
            write_json(self.path,j)
            return j

    def status(self, j=None):
        j = j or self.read()
        if 'active' in j:
            return j
        state, err = j['state'], j['error']
        try:
            active, record = self.engines.current()
        except HostError as e:
            active, record = None, None
            state, err = 'blocked', dict(code=e.code,params=e.params)
        if state in ('idle','brain_ready','inference_ready'):
            state = active+'_ready' if active else 'idle'
        owner = (record or {}).get('owner') or j['owner'] or {}
        safe_owner = {k:owner[k] for k in ('host','pid','exe') if k in owner} or None
        if err and err['code'] in ('owner_busy','owner_unknown'):
            safe_owner = (err['params'] or {}).get('owner',safe_owner)
        failed = state in ('blocked','save_failed','launch_failed') and bool(j['target']) and bool(err) and err['code'] not in ('owner_busy','owner_unknown')
        return dict(selected=j['selected'],active=active or 'none',target=j['target'],state=state,
                    transition_id=j['transition_id'],owner=safe_owner,error=err,can_retry=failed,
                    can_restore=bool(failed and j['restore']),previous_model_id=(j['inference_return'] or {}).get('model_id'))

    def results(self):
        try:
            return json.loads((self.directory/'mode-results.json').read_bytes())
        except FileNotFoundError:
            return {}

    def archive(self,j,snapshot):
        results=self.results()
        results[snapshot['transition_id']]=snapshot
        retained={r['transition_id'] for r in j['requests']} | {j['transition_id']}
        write_json(self.directory/'mode-results.json',{k:v for k,v in results.items() if k in retained})

    def replay(self,j,op,data):
        for r in j['requests']:
            if r['request_id'] == data.get('request_id'):
                require((r['op'],r['data'])==(op,data),'request_invalid')
                if r['transition_id']==j['transition_id']:
                    return j
                past=self.results().get(r['transition_id'])
                require(past,'transition_stale')
                return past
        return None

    def accept(self, op, data):
        require(self.owner, 'owner_required')
        rid = data.get('request_id')
        retry = op == 'mode_retry'
        require(op in OPS - {'mode_status'} and uid(rid) and
                set(data) == ({'request_id','transition_id','action'} if retry else {'request_id'}), 'request_invalid')
        if retry:
            require(uid(data['transition_id']) and data['action'] in ('retry','restore'), 'request_invalid')
        previous=self.read()
        snapshot=self.status(previous) if previous['transition_id'] and previous['state'] not in BUSY else None
        with lease(self.directory/'mode.lock'):
            j = self.read()
            prior=self.replay(j,op,data)
            if prior is not None:
                return prior,False
            require(j['state'] not in BUSY,'switch_busy')
            require(j==previous,'switch_busy')
            if retry:
                require(j['transition_id'] == data['transition_id'] and j['error'], 'transition_stale')
                if data['action'] == 'restore':
                    require(j['restore'], 'transition_stale')
                    j['target'] = j['restore']['mode']
            else:
                require(not j['target'] or not j['error'],'switch_busy')
                j.update(target='brain' if op == 'mode_switch_brain' else 'inference', transition_id=rid,
                         source=None, source_identity=None, dest_identity=None, restore=None)
            j['requests'].append(dict(request_id=rid,op=op,data=data,transition_id=j['transition_id']))
            while len(j['requests']) > 32:
                del j['requests'][int(j['requests'][0]['request_id'] == j['transition_id'])]
            if snapshot:
                # Keep the exact journal/request schema and archive only safe terminal projections.
                # Write first so a crash before mode.json is replaced still permits replay.
                self.archive(previous,snapshot)
            j.update(owner=self.owner,state='checking',error=None)
            write_json(self.path,j)
        return j, True

    def barrier(self, mode, record, tid):
        self.update(state='draining')
        end = time.monotonic() + 120
        try:
            while True:
                self.engines.check(mode,record)
                r = self.engines.admission(mode,record,tid,False)
                require(r.get('admit') is False and type(r.get('in_flight')) is int and r['in_flight'] >= 0,'engine_busy')
                if not r['in_flight']:
                    break
                require(time.monotonic() < end, 'engine_busy')
                time.sleep(.15)
            self.update(state='saving_brain' if mode == 'brain' else 'persisting_inference')
            self.engines.check(mode,record)
            self.engines.save(mode,record)
        except Exception:
            self.engines.check(mode,record)
            self.engines.admission(mode,record,tid,True)
            raise
        self.update(state='stopping')
        self.engines.check(mode,record)
        self.engines.stop(mode,record)
        require(self.engines.current()[0] is None,'stop_timeout')

    def run(self):
        j = self.read()
        require(self.owner and j['owner'] == self.owner and j['state'] == 'checking', 'owner_required')
        target, tid = j['target'], j['transition_id']
        try:
            mode, record = self.engines.current()
            original = next((r for r in j['requests'] if r['request_id']==tid), {})
            restoring = j['requests'][-1]['data'].get('action')=='restore' if j['requests'] else False
            if original.get('op') == 'host_stop' and not restoring:
                if mode:
                    require(mode==j['source'] and identity(record)==j['source_identity'],'owner_busy')
                    self.barrier(mode,record,tid)
                self.update(state='idle',target=None,error=None)
                return self.status()
            if j['restore'] is None:
                snap = self.engines.snapshot(mode or j['selected'], record)
                fields = dict(source=mode,source_identity=identity(record),restore=snap)
                if self.owner['host']=='inference' and (mode == 'inference' or
                        j['selected']=='inference' and target=='brain' and not j['inference_return']):
                    fields['inference_return'] = dict(install_id=self.owner['install_id'],model_id=snap['model_id'],
                        model_path=snap['model_path'],profile=snap['profile'],cache_dir=snap.get('cache_dir'))
                j = self.update(**fields)
            elif record and identity(record) not in (j['source_identity'],j['dest_identity']):
                require(mode==target and record.get('launch_id')==tid,'owner_unknown')
            dest = j['inference_return'] if target=='inference' else j['restore'] if j['restore']['mode']=='brain' else None
            if target=='inference' and dest and dest.get('install_id')!=self.owner['install_id']:
                dest=None
            if original.get('op')=='start' and not restoring:
                dest=original['data']
            if mode != target:
                self.engines.preflight(target,dest)
                if mode:
                    self.barrier(mode,record,tid)
                self.update(state='starting')
                require(self.engines.current()[0] is None,'owner_busy')
                self.engines.prepare_launch(tid)
                record = self.engines.start(target,dest)
            if record:
                self.update(state='waiting_ready',dest_identity=identity(record))
                self.engines.ready(target,record,tid)
                self.engines.admission(target,record,tid,True)
            fields = dict(state=target+'_ready' if record else 'idle', selected=target,target=None,error=None)
            if target == 'inference':
                fields['inference_return'] = None
            self.update(**fields)
        except Exception as e:
            code = getattr(e,'code','launch_failed')
            if not isinstance(code,str):
                code = 'launch_failed'
            phase = self.read()['state']
            if phase in ('persisting_inference','saving_brain'):
                state, code = 'save_failed', 'persist_failed' if phase=='persisting_inference' else 'save_failed'
            elif phase in ('starting','waiting_ready'):
                try:
                    live = self.engines.current()[0]
                except Exception:
                    live = True
                state = 'blocked' if live else 'launch_failed'
            else:
                state = 'blocked'
            self.update(state=state,error=dict(code=code,params=getattr(e,'params',{})))
        return self.status()

    def recover(self):
        j = self.read()
        require(self.owner,'owner_required')
        if j['owner'] == self.owner or not j['owner']:
            return
        mode, record = self.engines.current()
        if mode:
            require(record['owner']['install_id']==self.owner['install_id'] and record['owner']['host']==self.owner['host'],'owner_busy')
            dest_match=(mode==j['target'] and (identity(record)==j['dest_identity'] or
                        j['state']=='starting' and record.get('launch_id')==j['transition_id']))
            if j['state'] in BUSY and not dest_match and identity(record)!=j['source_identity']:
                # A new intent can have crashed before its first source snapshot. Reopen only a proven launch.
                if not (j['state']=='checking' and not j['source_identity'] and not j['restore']):
                    self.update(state='blocked',error=dict(code='owner_unknown',params={}))
                    return
            self.engines.reattach(mode,record,self.owner)
        if j['state'] in BUSY:
            if record and dest_match:
                self.update(owner=self.owner,state='checking',dest_identity=identity(record))
                self.run()
            else:
                if record:
                    self.engines.admission(mode,record,j['transition_id'],True)
                self.update(owner=self.owner,state='blocked',error=dict(code='transition_interrupted',params={}))
        else:
            self.update(owner=self.owner)


class Engines:
    def __init__(self, manager, host, context):
        self.manager,self.host,self.context = manager,host,context
        self.paths = dict(brain=host.config_dir/'process.json',inference=manager.DATA/'process.json')

    def prepare_launch(self,tid):
        self.context['launch_id']=tid

    def current(self):
        live = []
        for mode,path in self.paths.items():
            rec = self.manager.read_json(path,{})
            ident = rec.get('identity')
            if ident and self.manager.process_identity(ident['pid']) == ident:
                owner = rec.get('owner') or {}
                ours = (self.context or {}).get('owner') or {}
                if not owner or (owner.get('host'),owner.get('install_id')) != (ours.get('host'),ours.get('install_id')):
                    raise HostError('owner_busy',owner=dict(host=owner.get('host','foreign'),pid=ident['pid'],exe=ident['exe']))
                expected = self.host.runtime_dir/'llama-server.exe' if mode=='brain' else self.manager.RUNTIME
                require(Path(ident['exe']).resolve() == expected.resolve() and uid(rec.get('launch_id')),'owner_unknown')
                live.append((mode,rec))
        for candidate in self.manager.discover():
            if not any(candidate['ProcessId']==r['identity']['pid'] for _,r in live):
                raise HostError('owner_busy' if candidate.get('ExecutablePath') else 'owner_unknown',
                                owner=dict(host='foreign',pid=candidate['ProcessId'],exe=candidate.get('ExecutablePath')))
        require(len(live)<=1,'owner_unknown')
        return live[0] if live else (None,None)

    def check(self, mode, record):
        actual,current = self.current()
        require(mode==actual and identity(current)==identity(record) and current['owner']==self.context['owner'],'owner_busy')

    def snapshot(self, mode, record):
        r = record or {}
        return dict(mode=mode,model_id=r.get('model_id'),profile=r.get('profile'),model_path=r.get('model_path'),
                    runtime_sha256=self.context.get('digest') if mode=='brain' else None,memory_dir=(r.get('brain') or {}).get('dir'),
                    cache_dir=str(self.manager.DATA/'prompt-cache') if mode=='inference' and record else None)

    def preflight(self, target, snapshot):
        if target=='brain':
            self.host._runtime.verified(self.host.runtime_dir)
            app = self.host.invoke('brain_app',{'refresh':True})
            if not app['model']['ready']:
                raise HostError('brain_no_model',name=app['model']['name'])
            if not app['memory']['found']:
                raise HostError('brain_no_folder',dir=app['memory']['dir'])
            if not app['licence'].get('checked') or app['licence'].get('enforced',True) and not app['licence'].get('ok'):
                raise HostError('brain_licence',reason=app['licence'].get('error') or 'A valid licence is required.')
        elif snapshot and snapshot['model_id']:
            model = self.manager.model_by_id(snapshot['model_id'])
            require(model['path']==snapshot['model_path'],'model_unlisted')
            self.manager.validate_profile(snapshot['profile'],model)

    def endpoint(self, mode, record):
        self.check(mode,record)
        require(self.manager.listener_matches(record) and 'api_key' in record and record.get('admin_token'),'owner_unknown')
        return self.manager.base_url(self.manager.listening(record)), {
            **self.manager.auth_headers(record),'X-Rulith-Admin':record['admin_token'],'Content-Type':'application/json'}

    def request(self, mode, record, path, body=None, timeout=5):
        base, headers = self.endpoint(mode,record)
        req = urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        with self.manager.HTTP.open(req,timeout=timeout) as response:
            return json.loads(response.read(1<<20))

    def admission(self,mode,record,tid,admit):
        try:
            return self.request(mode,record,'/rulith/control/admission',dict(transition_id=tid,admit=admit))
        except OSError:
            raise HostError('engine_busy') from None

    def save(self,mode,record):
        if mode=='brain':
            try:
                self.host.invoke(None,{},save=True)
            except Exception:
                raise HostError('save_failed') from None
        elif (record.get('profile') or {}).get('prompt_cache_disk'):
            try:
                reply=self.request(mode,record,'/strix/persist',{},150)
                require(isinstance(reply,dict) and reply.get('success') is True,'persist_failed')
            except (OSError,ValueError,TypeError):
                raise HostError('persist_failed') from None

    def stop(self,mode,record):
        self.manager.process_identity(record['identity']['pid'],True,record['identity'])
        write_json(self.paths[mode],{'last_log':record.get('log','')})

    def start(self,mode,snapshot):
        if mode=='brain':
            self.host.invoke('brain_app_start',{})
        elif snapshot and snapshot['model_id']:
            self.manager.launch(self.manager.model_by_id(snapshot['model_id']),snapshot['profile'],self.context)
        else:
            return None
        active,record = self.current()
        require(active==mode,'launch_failed')
        return record

    def ready(self,mode,record,tid):
        end = time.monotonic()+600
        while time.monotonic()<end:
            self.check(mode,record)
            if self.manager.listener_matches(record):
                try:
                    if self.request(mode,record,'/health').get('status')=='ok':
                        models = self.request(mode,record,'/v1/models')['data']
                        expected = {Path(record['model_path']).name,record['model_id'],Path(record['model_path']).stem}
                        if record.get('served_id'):
                            expected={record['served_id']}
                        if mode=='inference':
                            expected.add(self.manager.api_model_name(self.manager.model_by_id(record['model_id'])))
                        if any(m.get('id') in expected for m in models) and (mode!='brain' or
                                self.request(mode,record,'/v1/memory/status').get('dir')==record['brain']['dir']):
                            return
                except (OSError,ValueError,KeyError):
                    pass
            time.sleep(.25)
        raise HostError('ready_timeout')

    def reattach(self,mode,record,owner):
        record['owner'] = {k:owner[k] for k in ('host','instance_id','install_id')}
        write_json(self.paths[mode],record)
