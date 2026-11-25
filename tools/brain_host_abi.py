"""The public Python host ABI. Product operations live in the authenticated release."""
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import uuid

from brain_runtime_host import HostError, MODULE, OPS as RUNTIME_OPS, lease, limited, require, write_json

OPS = frozenset('''brain brain_save brain_passages brain_memory brain_licence brain_trial brain_activate
brain_transfer brain_upgrade brain_upgrade_finish brain_create brain_runtime brain_runtime_install
brain_runtime_cancel brain_runtime_check brain_app brain_app_select brain_app_readonly brain_room
brain_app_start brain_self brain_self_set brain_capacity brain_capacity_set brain_folders brain_folder_add
brain_folder_remove brain_folder_set brain_folder_scan brain_web_settings brain_web_set brain_web_search
brain_web_fetch'''.split())
PRIVATE = {'brain', 'brain_passages', 'brain_save'}
PRESET = 'qwen3.8-flash-next-exl3-4.05bpw'
DOWNLOAD = frozenset('dl_start dl_status dl_pause dl_resume dl_cancel'.split())

DOCS = {'settings','conversation','study','learn','activity'}
STORE_LIMIT = 16 << 20
SETTINGS = {
    bool:'remember themeChosen introSeen web studyOn studyDayLimit studyAcOnly studyPaused showActivity',
    int:'focusK focusEvery studyPagesPerDay',
    str:'persona thinking recall theme learnRead language studyAreas ownerName'}


def document(name,value):
    if value is None:
        return True
    if name=='settings' and isinstance(value,dict):
        types = {k:t for t,keys in SETTINGS.items() for k in keys.split()}
        return all(k in types and type(v) is types[k] for k,v in value.items())
    if name=='conversation' and isinstance(value,dict):
        msgs = value.get('messages')
        return (isinstance(value.get('id'),str) and isinstance(msgs,list) and len(msgs)<=400
                and all(isinstance(m,dict) for m in msgs))
    if name=='study' and isinstance(value,dict):
        return (isinstance(value.get('memories'),dict) and isinstance(value.get('order'),list)
                and all(isinstance(k,str) for k in value['order']) and isinstance(value.get('day'),dict)
                and isinstance(value['day'].get('date'),str) and type(value['day'].get('pages')) is int
                and value['day']['pages']>=0 and isinstance(value.get('log'),list))
    if name=='learn':
        return isinstance(value,list) and len(value)<=6 and all(isinstance(v,dict) for v in value)
    if name=='activity' and isinstance(value,dict):
        rows = value.get('records')
        return (isinstance(value.get('order'),list) and isinstance(rows,dict) and len(rows)<=4 and
                len(value['order'])<=4 and all(isinstance(k,str) and k in rows for k in value['order']) and
                all(isinstance(v,dict) and isinstance(v.get('folders'),dict) and isinstance(v.get('entries'),list)
                    and len(v['entries'])<=2000 for v in rows.values()))
    return False


def ui_store(directory,op,data):
    name = data.get('name')
    reading = op == 'ui_store_read'
    require(op in ('ui_store_read','ui_store_write') and isinstance(name,str) and name in DOCS and
            set(data)==({'name'} if reading else {'name','value','expected_rev'}),'request_invalid')
    path = directory/'ui-state.json'
    with lease(directory/'ui-state.lock'):
        try:
            store = json.loads(path.read_bytes().decode('utf-8'))
            require(isinstance(store,dict),'store_corrupt')
            require(type(store.get('schema_version')) is int and store['schema_version']==1,'store_version')
            require(isinstance(store.get('docs'),dict) and isinstance(store.get('legacy_imported'),list) and
                    all(isinstance(k,str) and k in DOCS for k in store['legacy_imported']), 'store_corrupt')
            for k,d in store['docs'].items():
                require(k in DOCS and isinstance(d,dict) and type(d.get('rev')) is int and d['rev']>=1
                        and 'value' in d and document(k,d['value']), 'store_corrupt')
        except FileNotFoundError:
            store = dict(schema_version=1,docs={},legacy_imported=[])
        except (UnicodeError,json.JSONDecodeError):
            raise HostError('store_corrupt') from None
        doc = store['docs'].get(name,dict(rev=0,value=None))
        if reading:
            return doc
        require(type(data['expected_rev']) is int and data['expected_rev']>=0 and document(name,data['value']),'request_invalid')
        if data['expected_rev'] != doc['rev']:
            raise HostError('store_conflict',rev=doc['rev'])
        size = len(json.dumps(data['value'],ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
        require(size <= STORE_LIMIT, 'request_too_large')
        rev = doc['rev']+1
        store['docs'][name] = dict(rev=rev,value=data['value'])
        try:
            write_json(path,store)
        except OSError:
            raise HostError('store_write_failed') from None
        return dict(rev=rev)


# A question mark marks an optional member. Folder paths receive a second native approval check.
FIELDS = {
    'brain_memory':'method:s path:s body?:o?', 'brain_licence':'refresh?:b','brain_trial':'',
    'brain_activate':'key:s','brain_transfer':'key:s','brain_upgrade':'key:s to:s','brain_upgrade_finish':'key:s',
    'brain_create':'name:s parent?:d select?:b','brain_app':'refresh?:b','brain_app_select':'dir:d',
    'brain_app_readonly':'dir?:d read_only?:b','brain_room':'dir?:d','brain_self':'dir?:d',
    'brain_self_set':'dir?:d text?:s?','brain_capacity':'dir?:d','brain_capacity_set':'dir?:d tokens?:u? confirm_forget?:b',
    'brain_folders':'sync?:b','brain_folder_add':'path:d','brain_folder_remove':'id:s','brain_folder_set':'id:s watch?:b retry?:b',
    'brain_folder_scan':'id?:s','brain_web_settings':'','brain_web_set':'provider?:s searxng_url?:s',
    'brain_web_search':'query:s lang?:s','brain_web_fetch':'url:s lang?:s','logs':'offset?:u',
    'dl_start':'model:s include:a source?:s','dl_status':'','dl_pause':'id:s','dl_resume':'id:s',
    'dl_cancel':'id:s delete_parts?:b','roots':'roots:a','status':'','stop':'','brain_app_start':'',
    **dict.fromkeys(RUNTIME_OPS,'')}

_PRIVATE_FIELDS = {'api_key','admin_token','authorization','x-rulith-admin','purchase_key','password',
                   'owner','launch_id','identity','command','argv','runtime_env'}


def frame_result(value):
    if isinstance(value,list):
        return [frame_result(row) for row in value]
    if isinstance(value,dict):
        return {k:frame_result(v) for k,v in value.items() if k.lower() not in _PRIVATE_FIELDS}
    return value


def log_slice(path,offset,record):
    size=path.stat().st_size
    reset=offset>size
    start=0 if reset else offset
    secrets=[v.encode() for k in ('api_key','admin_token') if (v:=record.get(k))]
    overlap=max([4,*map(len,secrets)])
    left=max(0,start-overlap)
    with path.open('rb') as f:
        f.seek(left)
        raw=f.read(start-left+65536+overlap)
    redacted=bytearray(raw)
    for secret in secrets:
        pos=raw.find(secret)
        while pos>=0:
            redacted[pos:pos+len(secret)]=b'*'*len(secret)
            pos=raw.find(secret,pos+1)
    chunk=redacted[start-left:start-left+65536]
    try:
        text=chunk.decode()
    except UnicodeDecodeError as e:
        if e.reason=='unexpected end of data':
            chunk=chunk[:e.start]
        text=chunk.decode(errors='replace')
    text=re.sub(r'(?i)(authorization:\s*bearer\s+|x-rulith-admin:\s*|--api-key\s+)\S+',r'\1***',text)
    return dict(file=path.name,offset=start+len(chunk),reset=reset,text=text)


def frame_data(op,data,approved):
    require(op in FIELDS,'op_denied')
    require(isinstance(data,dict),'request_invalid')
    remaining = set(data)
    for field in FIELDS[op].split():
        name,kind = field.split(':')
        k = name.rstrip('?')
        if k not in data:
            require(name.endswith('?'),'request_invalid')
            continue
        remaining.remove(k)
        v = data[k]
        if v is None and kind.endswith('?'):
            continue
        typ = {'d':str,'s':str,'o':dict,'b':bool,'u':int,'a':list}[kind[0]]
        require(type(v) is typ and (kind[0]!='u' or v>=0),'request_invalid')
        if kind=='d':
            p = Path(v)
            require(p.is_absolute() and p.is_dir() and p.resolve() in approved,'folder_denied')
    require(not remaining,'request_invalid')
    if op=='roots':
        require(1<=len(data['roots'])<=12 and all(isinstance(v,str) and Path(v).is_absolute() and
                Path(v).resolve() in approved for v in data['roots']),'folder_denied')
    if op=='brain_create':
        require(1<=len(data['name'].strip())<=80,'request_invalid')
    if op=='brain_self_set' and data.get('text'):
        require(len(data['text'])<=8000,'request_invalid')
    if op=='brain_capacity_set' and data.get('tokens') is not None:
        require(data['tokens']>0 and data['tokens']%1000000==0,'request_invalid')
    if op=='brain_web_set' and 'provider' in data:
        require(data['provider'] in ('wikipedia','searxng'),'request_invalid')
    if op=='brain_upgrade':
        require(data['to'] in ('10M','50M','200M','unlimited'),'request_invalid')
    if op=='dl_start':
        require(data['model']==PRESET and data['include']==['draft'],'op_denied')


def load_module(runtime, manifest_digest):
    # The caller holds a generation lease. Read the bytes once and execute that same buffer.
    from brain_runtime_host import MANIFEST, MAX_MANIFEST
    raw = limited(runtime.live / MANIFEST, MAX_MANIFEST)
    require(hashlib.sha256(raw).hexdigest() == manifest_digest, 'runtime_untrusted')
    m = json.loads(raw)
    entry = m['files'][MODULE]
    path = runtime.live / MODULE
    code = limited(path, entry['size'])
    require(len(code) == entry['size'] and hashlib.sha256(code).hexdigest() == m['manager']['sha256'])
    mod = types.ModuleType('rulith_brain_manager')
    mod.__file__ = str(path)
    sys.dont_write_bytecode = True
    require(sys.flags.isolated and sys.flags.no_user_site, 'runtime_untrusted')
    sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(runtime.live.resolve())]
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(code, str(path), 'exec', dont_inherit=True), mod.__dict__)
        require(type(mod.HOST_ABI) is int and mod.HOST_ABI == 1 and callable(mod.dispatch), 'runtime_abi')
    except HostError:
        raise
    except Exception:
        raise HostError('brain_module_failed') from None
    return mod


class BrainHostV1:
    abi = 1

    def __init__(self, runtime, install_root, inference_roots, context=None, host_kind='inference'):
        self.host_kind = host_kind
        self.brain_home = runtime.home.resolve()
        self.config_dir = self.brain_home / 'config'
        self.runtime_dir = runtime.live
        self.install_root = Path(install_root).resolve()
        self.licence_file = self.brain_home / 'licence.txt'
        self.inference_roots = tuple(map(Path, inference_roots))
        self._runtime, self._context = runtime, context
        # A separate instance gives existing catalog and download functions their own config.
        spec = importlib.util.spec_from_file_location('brain_scoped_catalog', Path(__file__).with_name('manager.py'))
        scoped = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoped)
        scoped.DATA = self.config_dir
        self._manager = scoped

    def read_json(self, path, default):
        require(Path(path).is_absolute(), 'request_invalid')
        return self._manager.read_json(Path(path), default)

    def atomic_json(self, path, value):
        require(Path(path).is_absolute(), 'request_invalid')
        write_json(Path(path), value)

    def lock(self, name, wait=True):
        require(isinstance(name,str) and re.fullmatch('[a-z][a-z0-9_.-]*',name), 'request_invalid')
        return lease(self.config_dir/'brain'/name, wait=wait)

    def catalog(self, roots, refresh):
        with lease(self.config_dir/'manager.lock'):
            return self._manager.catalog(refresh, roots=[str(p) for p in roots])

    def profile(self, model_id):
        return self.read_json(self.config_dir/'settings.json', {}).get('profiles', {}).get(model_id, {})

    def save_profile(self, model_id, profile):
        with lease(self.config_dir/'manager.lock'):
            cfg = self.read_json(self.config_dir/'settings.json', {})
            cfg.setdefault('profiles', {})[model_id] = profile
            self.atomic_json(self.config_dir/'settings.json', cfg)
        return profile

    def process_state(self):
        state = self.read_json(self.config_dir/'process.json', {})
        ident = state.get('identity')
        if not ident or self._manager.process_identity(ident['pid']) != ident:
            return {}
        require(self._context and state.get('owner') == self._context.get('owner'), 'owner_busy')
        require(self._manager.listener_matches(state),'owner_unknown')
        return state

    def _permit(self):
        require(self._context and self._context.get('permit') and self._context.get('digest'), 'owner_required')

    def spawn_engine(self, args, env, record):
        self._permit()
        require(not self.process_state(), 'owner_busy')
        net = self._manager.saved_network()
        host = '0.0.0.0' if net['lan'] else '127.0.0.1'
        require(not any(a.split('=')[0] in ('--host','--port','--api-key','--api-key-file') for a in args), 'request_invalid')
        engine = self.runtime_dir/'llama-server.exe'
        command = [str(engine), *args, '--host', host, '--port', str(net['port'])]
        if net['api_key']:
            command.extend(('--api-key', net['api_key']))
        launch_id, admin = self._context.get('launch_id') or str(uuid.uuid4()), os.urandom(32).hex()
        child_env = dict(os.environ)
        child_env.update(env)
        child_env.update(RULITH_APP='brain',RULITH_HOME=str(self.brain_home),RULITH_LICENCE_FILE=str(self.licence_file),
                         RULITH_BRAIN_ADMIN_TOKEN=admin)
        log = self.brain_home/'logs'/('brain-'+launch_id+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('xb') as out:
            from brain_runtime_host import private_file
            private_file(log)
            child = subprocess.Popen(command,cwd=self.runtime_dir,env=child_env,stdin=subprocess.DEVNULL,
                                     stdout=out,stderr=subprocess.STDOUT,creationflags=self._manager.HIDDEN)
        ident = self._manager.process_identity(child.pid)
        require(ident and Path(ident['exe']) == engine, 'owner_unknown')
        saved = {**record, 'identity':ident, 'owner':self._context['owner'], 'launch_id':launch_id,
                 'host':host, 'port':net['port'], 'api_key':net['api_key'], 'admin_token':admin, 'log':str(log),
                 'command':subprocess.list2cmdline(self._manager.masked(command)), 'adopted':False, 'runtime_env':{},
                 'started_at':dt.datetime.now().astimezone().isoformat()}
        if '--alias' in args:
            saved['served_id']=args[args.index('--alias')+1]
        saved['brain'] = {**record['brain'], 'runtime':str(self.runtime_dir)}
        self.atomic_json(self.config_dir/'process.json', saved)
        return saved

    def stop_engine(self, identity):
        self._permit()
        state = self.process_state()
        require(state.get('identity') == identity, 'owner_busy')
        self._manager.process_identity(identity['pid'], True, identity)
        self.atomic_json(self.config_dir/'process.json', {'last_log':state['log']})

    def download(self, op, data):
        require(op in DOWNLOAD, 'op_denied')
        if op == 'dl_start':
            require(data.get('model') == PRESET and data.get('include') == ['draft'] and not data.get('kind'), 'op_denied')
            # Refresh outside the download lock. The module assembles local shards and reads the job list.
            model = self.invoke('brain_app', {'refresh': True})['model']
            if model['ready']:
                raise HostError('dl_nothing')
            present = {f['name'] for f in model['files'] if f['present']}
            with lease(self.config_dir/'manager.lock'):
                return self._manager.dl_start(data, present=present)
        with lease(self.config_dir/'manager.lock'):
            if op in ('dl_pause', 'dl_resume', 'dl_cancel'):
                p = self._manager.dl_job(data.get('id'))
                require(self.read_json(p/'job.json', {}).get('model') == PRESET, 'op_denied')
            result = self._manager.handle(op, data)
            if 'jobs' in result:
                result['jobs'] = [j for j in result['jobs'] if j.get('model') == PRESET]
                if result.get('active') not in {j['id'] for j in result['jobs']}:
                    result['active'] = None
            return result

    def runtime(self, op, data):
        return self._runtime.operation(op, data, inside_generation=True,held_digest=(self._context or {}).get('digest'))

    def fail(self, code, **params):
        raise HostError(code, **params)

    def invoke(self, op, data, *, save=False):
        require(save or op in OPS - RUNTIME_OPS, 'unknown_op')
        with self._runtime.generation():
            require(self._context and self._context.get('digest'), 'runtime_untrusted')
            env=dict(RULITH_APP='brain',RULITH_HOME=str(self.brain_home),RULITH_LICENCE_FILE=str(self.licence_file))
            previous={k:os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                mod = load_module(self._runtime, self._context['digest'])
                if save:
                    self._permit()
                    result = mod.save_before_stop(self)
                    require(isinstance(result,dict) and result.get('outcome') in ('saved','empty','frozen') and
                            isinstance(result.get('reason'),str), 'save_failed')
                    return result
                return mod.dispatch(op, data, self)
            except HostError:
                raise
            except Exception:
                raise HostError('save_failed' if save else 'brain_module_failed') from None
            finally:
                for k,v in previous.items():
                    if v is None:
                        os.environ.pop(k,None)
                    else:
                        os.environ[k]=v
