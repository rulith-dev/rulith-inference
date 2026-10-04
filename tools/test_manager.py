import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('manager',Path(__file__).with_name('manager.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def gguf(path):
    def s(text):
        b=text.encode();return struct.pack('<Q',len(b))+b
    fields=s('general.name')+struct.pack('<I',8)+s('Small model')
    fields+=s('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,2)+s('one')+s('two')
    fields+=s('general.architecture')+struct.pack('<I',8)+s('qwen4exp')
    fields+=s('qwen4exp.context_length')+struct.pack('<II',4,32768)
    path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,4)+fields)

def gguf_tensors(path,tensors,arch='qwen4exp'):
    """A GGUF v3 file with real tensor infos and a data section: tensors = [(name, dims, type, bytes)]."""
    def s(text):
        b=text.encode();return struct.pack('<Q',len(b))+b
    kv=s('general.architecture')+struct.pack('<I',8)+s(arch)+s('general.name')+struct.pack('<I',8)+s('tiny')
    infos,offset,data=b'',0,b''
    for name,dims,ttype,payload in tensors:
        infos+=s(name)+struct.pack('<I',len(dims))+b''.join(struct.pack('<Q',d) for d in dims)+struct.pack('<IQ',ttype,offset)
        data+=payload;offset+=len(payload)
        pad=-offset%32;data+=b'\0'*pad;offset+=pad
    head=b'GGUF'+struct.pack('<IQQ',3,len(tensors),2)+kv+infos
    path.write_bytes(head+b'\0'*(-len(head)%32)+data)

def first_model(refresh=True):
    # the catalog sorts by role, and the draft and projector created in setUp sort before the model
    return next(x for x in m.catalog(refresh)['models'] if x['role']=='model')

class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.patcher=patch.object(m,'DATA',self.root/'state');self.patcher.start();self.addCleanup(self.patcher.stop)
        # RULITH_PORT, RULITH_HOST and RULITH_API_KEY override the network setting: none unless a test sets one
        env=patch.dict(os.environ,{k:v for k,v in os.environ.items() if not k.startswith('RULITH_')},clear=True)
        env.start();self.addCleanup(env.stop)
        m.atomic_json(m.DATA/'settings.json',{'roots':[str(self.root/'models')],'profiles':{}})
        (self.root/'models').mkdir()
        self.file=self.root/'models'/'model-Q8_0-00001-of-00002.gguf';gguf(self.file)
        self.second=self.file.with_name('model-Q8_0-00002-of-00002.gguf');gguf(self.second)
        # vision defaults on, and the projector is looked for beside the model. Without this the
        # suite passed only on a machine that happened to have a real one at the old default path.
        self.mmproj=self.file.with_name(m.MMPROJ_NAME);gguf(self.mmproj)
        # MTP likewise defaults on only when a draft for the family is on disk
        self.draft=self.file.with_name('mtp-'+m.MODEL_FAMILY+'-shared-Q4_K_M.gguf');gguf(self.draft)
        self.model={'id':'model','path':str(self.file),'architecture':'qwen4exp','context':32768,'role':'model'}
        # a fake runtime, so the suite runs on a machine that has never built one
        rt=self.root/'bin'/'hip';rt.mkdir(parents=True)
        for name in ('llama-server.exe','ggml-hip.dll'):(rt/name).write_bytes(b'')
        rocm=self.root/'toolchain'/'rocm';rocm.mkdir(parents=True);(rocm/'amdhip64_7.dll').write_bytes(b'')
        for attr,value in (('RUNTIME',rt/'llama-server.exe'),('ROCM_BIN',rocm)):
            pat=patch.object(m,attr,value);pat.start();self.addCleanup(pat.stop)
    def test_metadata_skips_tokenizer_and_reads_later_fields(self):
        self.assertEqual(m.metadata(self.file)['qwen4exp.context_length'],32768)
    def test_catalog_groups_shards_and_marks_missing(self):
        first=[x for x in m.catalog(True)['models'] if x['role']=='model'];self.assertEqual(len(first),1)
        self.assertEqual(first[0]['size'],self.file.stat().st_size*2)
        self.second.unlink()
        self.assertEqual(first_model()['missing'],[self.second.name])
    def test_model_path_cannot_escape_registered_roots(self):
        outside=self.root/'outside.gguf';gguf(outside)
        with self.assertRaises(ValueError):m.checked_file(outside)
        with self.assertRaises(ValueError):m.checked_file(self.second)
    def test_invalid_profiles_fail_before_launch(self):
        for values in [{'context':999999},{'context':True},{'ubatch':1024,'batch':512},{'kv':'q4_0'},{'kv':'Q8_0'}, {'flash_attention':'auto'}, {'flash_attention':True}, {'flash_attention':['on']},{'draft_min':float('nan')},{'command':'calc.exe'}]:
            with self.subTest(values=values),self.assertRaises(ValueError):m.validate_profile(values,self.model)
    def test_argv_keeps_paths_as_single_arguments(self):
        cfg=m.validate_profile({'mtp':False},self.model)
        args=m.argv(self.model,cfg)
        self.assertEqual(args[args.index('-m')+1],str(self.file))
        self.assertIn('f16',args);self.assertNotIn('--spec-type',args)
    def test_large_batches_survive_save_and_reach_launch_without_starting_model(self):
        model=first_model()
        with patch.object(m,'state',return_value={}),patch.object(m.subprocess,'Popen') as popen:
            for batch,ubatch in ((4096,4096),(8192,4096),(8192,8192),(16384,16384),(32768,32768)):
                with self.subTest(batch=batch,ubatch=ubatch):
                    m.handle('save',{'id':model['id'],'profile':{'mtp':False,'batch':batch,'ubatch':ubatch}})
                    cfg=m.profile(model)
                    self.assertEqual((cfg['batch'],cfg['ubatch']),(batch,ubatch))
                    args=m.argv(model,cfg)
                    self.assertEqual(args[args.index('-b')+1],str(batch))
                    self.assertEqual(args[args.index('-ub')+1],str(ubatch))
            popen.assert_not_called()
        for values in ({'batch':32769},{'batch':32768,'ubatch':32769},
                       {'batch':4096,'ubatch':8192},{'batch':31},{'ubatch':31}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                m.validate_profile({'mtp':False,**values},model)
    def test_ngram_spec_legacy_default_and_invalid_values(self):
        model=first_model()
        settings=m.settings();settings['profiles'][model['id']]={'mtp':False,'context':8192}
        m.atomic_json(m.DATA/'settings.json',settings)
        before=(m.DATA/'settings.json').read_bytes()
        cfg=m.validate_profile({},model)
        self.assertIs(cfg['ngram_spec'],False)
        self.assertEqual(cfg['context'],8192)
        self.assertNotIn('--spec-type',m.argv(model,cfg))
        self.assertEqual((m.DATA/'settings.json').read_bytes(),before)
        for value in ('true',1,None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                m.validate_profile({'ngram_spec':value},model)
    def test_ngram_save_start_is_independent_of_mtp_and_embedding_placement(self):
        from types import SimpleNamespace
        model=first_model()
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        with patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]), \
             patch.object(m,'process_identity',return_value=ident), \
             patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')), \
             patch('socket.socket'),patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as popen:
            for mtp,ngram in ((False,False),(False,True),(True,False),(True,True)):
                if True:
                    with self.subTest(mtp=mtp,ngram=ngram):
                        popen.reset_mock()
                        m.handle('save',{'id':model['id'],'profile':{'mtp':mtp,'ngram_spec':ngram,
                            'draft':str(self.file) if mtp else str(self.root/'missing.gguf'),
                            'draft_max':3,'draft_min':0.3}})
                        popen.assert_not_called()
                        self.assertIs(m.profile(model)['ngram_spec'],ngram)
                        result=m.handle('start',{'id':model['id']})
                        args=popen.call_args.args[0]
                        expected=','.join((['ngram-mod'] if ngram else [])+(['draft-mtp'] if mtp else []))
                        self.assertEqual(args.count('--spec-type'),int(bool(expected)))
                        if expected:self.assertEqual(args[args.index('--spec-type')+1],expected)
                        self.assertEqual('-md' in args,mtp)
                        # this runtime reads the per-layer embedding table itself, never pinned to CPU
                        self.assertNotIn('-ot',args)
                        self.assertIn('--lazy-mode',args)
                        self.assertEqual('--spec-ngram-mod-n-max' in args,ngram)
                        if ngram:
                            for option,value in (('--spec-ngram-mod-n-match','24'),('--spec-ngram-mod-n-min','4'),('--spec-ngram-mod-n-max','8')):
                                self.assertEqual(args[args.index(option)+1],value)
                        if mtp:
                            self.assertEqual(args[args.index('--spec-draft-n-max')+1],'3')
                            self.assertEqual(args[args.index('--spec-draft-p-min')+1],'0.3')
                        self.assertEqual(args[args.index('-ctk')+1],'f16')
                        self.assertEqual(args[args.index('-ctv')+1],'f16')
                        self.assertIs(result['profile']['ngram_spec'],ngram)
                        self.assertIn('n-gram draft=on' if ngram else 'n-gram draft=off',Path(result['log']).read_text())
                        m.handle('stop',{})
    def kwargs_for(self,thinking):
        import json
        cfg=m.validate_profile({'mtp':False,'thinking':thinking},self.model)
        args=m.argv(self.model,cfg)
        self.assertIn('--cache-prompt',args)
        return json.loads(args[args.index('--chat-template-kwargs')+1])
    def test_thinking_levels_map_to_the_template_s_reasoning_effort(self):
        # the chat template raises on anything outside low/medium/xhigh, so these are not free-form
        self.assertEqual(self.kwargs_for('off'),{'enable_thinking':False})
        for level,effort in (('low','low'),('medium','medium'),('high','xhigh')):
            self.assertEqual(self.kwargs_for(level),
                             {'enable_thinking':True,'reasoning_effort':effort})
    def test_thinking_still_accepts_the_boolean_it_used_to_be(self):
        self.assertEqual(self.kwargs_for(False),{'enable_thinking':False})
        self.assertEqual(self.kwargs_for(True),{'enable_thinking':True,'reasoning_effort':'xhigh'})
    def test_thinking_rejects_a_level_the_template_would_raise_on(self):
        for bad in ('false','xhigh','extra-high',3):
            with self.assertRaises(ValueError):m.validate_profile({'thinking':bad},self.model)
    def test_profile_normalises_the_old_boolean_thinking_and_drops_unknown_fields(self):
        model=first_model()
        settings=m.settings();settings['profiles'][model['id']]={'thinking':True,'ngram_cpu':True,'context':8192}
        m.atomic_json(m.DATA/'settings.json',settings)
        cfg=m.profile(model)
        # the page displays what profile() returns, so a stored switch must already be a level
        self.assertEqual(cfg['thinking'],'high');self.assertNotIn('ngram_cpu',cfg);self.assertEqual(cfg['context'],8192)
        # and sends it straight back: an earlier version's field must not make the whole profile unknown
        self.assertEqual(m.validate_profile({**cfg,'mtp':False},model)['context'],8192)
    def test_no_unified_memory_variable_is_set_or_inherited(self):
        # up to 0.1.14 the manager set GGML_HIP_ENABLE_UNIFIED_MEMORY, which nothing reads; ggml reads only
        # GGML_CUDA_ENABLE_UNIFIED_MEMORY, and any value of it - "0" too - turns managed memory on
        cfg=m.validate_profile({'mtp':False},self.model)
        with patch.dict(m.os.environ,{'GGML_HIP_ENABLE_UNIFIED_MEMORY':'1','GGML_CUDA_ENABLE_UNIFIED_MEMORY':'0'}):
            env=m.runtime_environment(cfg)
        self.assertFalse([k for k in env if 'UNIFIED_MEMORY' in k.upper()])
    def test_a_profile_saved_with_the_retired_shared_vram_field_still_loads(self):
        cfg=m.validate_profile({'mtp':False,'shared_vram':True},self.model)
        self.assertNotIn('shared_vram',cfg)
        with self.assertRaises(m.ManagerError) as caught:m.validate_profile({'mtp':False,'no_such_field':1},self.model)
        self.assertEqual(caught.exception.code,'unknown_field')
    def running(self,alive,dedicated=64*2**30):
        """The launch fixture: a fake runtime, no listener on 8080, and process_identity answering
        from `alive` - a queue whose last entry repeats, so [None, ident] is one dead check and
        then a live process."""
        from types import SimpleNamespace
        from contextlib import ExitStack
        stack=ExitStack()
        for p in (patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]),
                  patch.object(m,'dedicated_vram_bytes',return_value=dedicated),
                  patch.object(m,'process_identity',side_effect=lambda pid,*a,**k: alive.pop(0) if len(alive)>1 else alive[0]),
                  patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')),
                  patch('socket.socket')):
            stack.enter_context(p)
        popen=stack.enter_context(patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)))
        return stack,popen
    def test_a_ready_server_with_little_commit_left_is_flagged(self):
        model=first_model()
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        stack,popen=self.running([ident])
        with stack:
            m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            answers=lambda path,server:{'status':'ok'} if path=='/health' else {'data':[]}
            with patch.object(m,'http_json',side_effect=answers):
                for available,low in ((20<<30,False),(3<<30,True)):
                    with self.subTest(available=available),patch.object(m,'commit_bytes',return_value=(128<<30,available)):
                        s=m.status()
                        self.assertEqual((s['status'],s['commit_available'],s['commit_low']),('ready',available,low))
            # still loading: nothing to say about commit yet
            with patch.object(m,'commit_bytes',return_value=(128<<30,1<<30)):
                self.assertNotIn('commit_low',m.status())
    def test_out_of_memory_is_reported_not_retried(self):
        model=first_model()
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        alive=[ident]
        stack,popen=self.running(alive)
        with stack:
            first=m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            self.assertNotIn('unified',first);self.assertNotIn('shared memory=',Path(first['log']).read_text())
            with open(first['log'],'a') as f:f.write('ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3488.00 MiB on device 0: cudaMalloc failed: out of memory\n')
            alive[0]=None
            status=m.status()
            # the same load again would run out the same way: it is reported, and nothing is started
            self.assertEqual(popen.call_count,1)
            self.assertEqual(status['status'],'stopped');self.assertIn('out of memory',status['failure'])
            self.assertNotIn('notice',status)
            # a plain exit - the app closing takes the server with it - is not a failure to report
            m.handle('stop',{});alive[0]=ident
            m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            alive[0]=None
            self.assertNotIn('failure',m.status())
            m.handle('stop',{})
            self.assertNotIn('exited',m.read_json(m.DATA/'process.json',{}))
    def test_installed_layout_needs_no_sdk_directory_and_no_path_entry(self):
        # tools/make_runtime_bundle.py puts the ROCm DLLs beside llama-server
        with patch.object(m,'ROCM_BIN',self.root/'nowhere'):
            self.assertFalse(m.runtime_available())
            for name in ('amdhip64_7.dll','hipblas.dll'):(m.RUNTIME.parent/name).write_bytes(b'')
            self.assertTrue(m.runtime_available())
            env=m.runtime_environment(m.validate_profile({'mtp':False},self.model))
            self.assertNotIn('nowhere',env['PATH'])
    def test_build_tree_with_the_hip_runtime_beside_it_still_takes_the_sdk_through_path(self):
        # bootstrap.py --build copies only the DLLs System32 would shadow; hipBLAS and the rest are the SDK's
        (m.RUNTIME.parent/'amdhip64_7.dll').write_bytes(b'')
        self.assertFalse(m.bundled_rocm())
        self.assertTrue(m.runtime_available())
        env=m.runtime_environment(m.validate_profile({'mtp':False},self.model))
        self.assertTrue(env['PATH'].startswith(str(m.ROCM_BIN)))
    def test_draft_falls_back_to_the_family_s_shared_head_beside_the_model(self):
        # an installed copy has no models/ of ours; Unsloth's mtp-*.gguf sits next to the model
        shared=self.file.with_name('mtp-'+m.MODEL_FAMILY+'-shared-Q4_K_M.gguf');gguf(shared)
        other=self.file.with_name('mtp-Other-Model-shared-Q4_K_M.gguf');gguf(other)
        m.catalog(True)
        with patch.object(m,'DEFAULT_DRAFT',self.root/'models'/'missing-head.gguf'):
            # the catalog stores resolved paths; the temp dir may be an 8.3 short name
            self.assertEqual(Path(m.family_draft()).resolve(),shared.resolve())
            model={**self.model,'path':str(self.file.with_name(m.MODEL_FAMILY+'-UD-IQ4_XS.gguf'))}
            self.assertEqual(Path(m.profile(model)['draft']).resolve(),shared.resolve())
            # a head made by tools/make_draft_head.py beside the model wins over the shared file
            head=self.file.with_name('mtp-'+m.MODEL_FAMILY+'-shared-Q4_K_M-head-iq4_xs.gguf');gguf(head)
            m.catalog(True)
            self.assertEqual(Path(m.profile(model)['draft']).resolve(),head.resolve())
            head.unlink();shared.unlink();m.catalog(True)
            with self.assertRaisesRegex(ValueError,'draft'):m.validate_profile({'mtp':True},model)
            # and the default no longer asks for a file that is not there
            self.assertFalse(m.profile(model)['mtp']);self.assertEqual(m.profile(model)['draft'],'')
    def test_saving_configuration_does_not_start_a_process(self):
        m.catalog(True)
        model=first_model(False)
        with patch.object(m,'state',return_value={}),patch.object(m.subprocess,'Popen') as popen:
            result=m.handle('save',{'id':model['id'],'profile':{'context':8192,'mtp':False}})
            self.assertEqual(result['profile']['context'],8192);popen.assert_not_called()
        self.assertEqual(m.profile(model)['context'],8192)
    def test_flash_attention_survives_save_and_reaches_launch_arguments(self):
        model=first_model()
        self.assertEqual(m.profile(model)['flash_attention'],'on')
        with patch.object(m,'state',return_value={}),patch.object(m.subprocess,'Popen') as popen:
            for mode in ('on','off'):
                # sparse attention needs flash attention, so turning the latter off means both
                m.handle('save',{'id':model['id'],'profile':{'mtp':False,'flash_attention':mode,'qsa':mode=='on'}})
                cfg=m.validate_profile(m.profile(model),model)
                args=m.argv(model,cfg)
                self.assertEqual(args[args.index('-fa')+1],mode)
                self.assertEqual(args[args.index('-ctk')+1],'f16')
                self.assertEqual(args[args.index('-ctv')+1],'f16')
            popen.assert_not_called()
    def test_unknown_operation_rejected(self):
        with self.assertRaises(ValueError):m.handle('execute',{'cmd':'calc.exe'})
    def test_runtime_allowlist_is_exact_and_status_uses_live_binary(self):
        runtime=m.RUNTIME
        self.assertTrue(m.managed_runtime(runtime))
        self.assertFalse(m.managed_runtime(runtime.parent/'other.exe'))
        ident={'pid':123,'exe':str(runtime.resolve()),'birth':456}
        with patch.object(m,'state',return_value={'identity':ident,'adopted':True}),patch.object(m,'http_json',return_value={'status':'ok','data':[]}):
            status=m.status()
            self.assertEqual(status['runtime'],str(runtime.resolve()))
            # a binary at the right path proves nothing about the gates it was started with
            self.assertNotIn('runtime_env',status)
        self.assertFalse(m.managed_runtime(self.root/'some-other-build'/'llama-server.exe'))
        # the folder was bin/hip-rocm101 up to 0.2.4: a server an older install started there is still ours
        self.assertTrue(m.managed_runtime(runtime.parent.parent/'hip-rocm101'/'llama-server.exe'))
    def test_runtime_info_names_the_rocm_release_and_gpu_of_the_bundle(self):
        (self.root/'BUNDLE.json').write_text(json.dumps({'gfx':'gfx1151','rocm':'rocm_sdk_devel-10.2.0a20260925.dist-info'}),encoding='utf-8')
        self.assertEqual(m.runtime_info(),{'rocm':'10.2.0a20260925','gfx':'gfx1151'})
        with patch.object(m,'state',return_value={}):
            self.assertEqual(m.status()['runtime_info']['rocm'],'10.2.0a20260925')
    def test_answers_stopped_for_lack_of_memory_are_reported_with_their_time(self):
        log=self.root/'jan-managed-20260926-132707-dacb1d.log'
        log.write_text('0.29.818.996 W llama_context: n_ctx_seq (500224) > n_ctx_train (262144)\n'
                       '6.19.909.169 E srv  update_slots: decode() failed: bad allocation\n'
                       '6.19.909.251 E srv    send_error: task id = 2742, error: decode() failed: bad allocation\n'
                       '6.20.035.229 E srv    send_error: task id = 2894, error: decode() failed: bad allocation\n'
                       '7.07.387.619 W srv  persist_runs:  - disk cache: not enough memory to copy the checkpoints, not saving\n',encoding='utf-8')
        self.assertEqual(m.memory_events(log),{'answers_stopped':2,'saves_skipped':1,
                                               'last_stop':'2026-09-26T13:33:27','last_skip':'2026-09-26T13:34:14'})
        clean=self.root/'jan-managed-20260926-140000-aaaaaa.log'
        clean.write_text('0.33.207.886 I srv  llama_server: model loaded\n',encoding='utf-8')
        self.assertIsNone(m.memory_events(clean))
        self.assertIsNone(m.memory_events(self.root/'missing.log'))
        self.assertIsNone(m.memory_events(None))
    def test_slots_reports_each_slots_counters_and_nothing_without_a_server(self):
        with patch.object(m,'state',return_value={}):
            self.assertEqual(m.slots(),{'slots':[]})
        answer=[{'id':0,'is_processing':True,'id_task':12,'n_prompt_tokens':51008,'n_prompt_tokens_processed':15,
                 'n_prompt_tokens_cache':50993,'next_token':[{'n_decoded':1892,'n_remain':-1}]},
                {'id':1,'is_processing':False,'n_ctx':262144}]
        class Res:
            headers={'X-Strix-Waiting':'2'}
            def __enter__(s): return s
            def __exit__(s,*a): return False
            def read(s,*a): return json.dumps(answer).encode()
        with patch.object(m,'state',return_value={'identity':{'pid':1}}),patch.object(m.HTTP,'open',return_value=Res()):
            r=m.slots()
        got=r['slots']
        self.assertEqual(r['waiting'],2)
        self.assertEqual(got[0],{'id':0,'active':True,'task':12,'context':51008,'prompt_processed':15,'prompt_cached':50993,'generated':1892})
        self.assertEqual(got[1],{'id':1,'active':False,'task':None,'context':0,'prompt_processed':0,'prompt_cached':0,'generated':0})
        with patch.object(m,'state',return_value={'identity':{'pid':1}}),patch.object(m.HTTP,'open',side_effect=OSError('busy')):
            self.assertEqual(m.slots(),{'slots':None})
    def test_a_server_from_another_copy_of_the_manager_is_adopted_and_can_be_unloaded(self):
        # an earlier install's runtime, on our port, started with our flags - not the pinned binary
        other=str((self.root/'elsewhere'/'hip'/'llama-server.exe').resolve())
        ident={'pid':321,'exe':other,'birth':7}
        entry={'ProcessId':321,'ExecutablePath':other,'CommandLine':'llama-server.exe -m x.gguf --host 127.0.0.1 --port 8080 --load-mode none --lazy-mode on-direct'}
        with patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident) as identity:
            self.assertEqual(m.state()['identity'],ident)
            m.handle('stop',{});identity.assert_called_with(321,True,ident)
        # but not an unrelated llama-server that merely sits on the port
        entry['CommandLine']='llama-server.exe -m x.gguf --host 127.0.0.1 --port 8080'
        m.atomic_json(m.DATA/'process.json',{})
        with patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident):
            self.assertNotIn('identity',m.state())
    def test_candidate_can_be_adopted_and_stopped_by_exact_identity(self):
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        entry={'ProcessId':123,'ExecutablePath':ident['exe'],'CommandLine':'llama-server.exe --host 127.0.0.1 --port 8080'}
        with patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident) as identity:
            adopted=m.state()
            self.assertTrue(adopted['adopted'])
            self.assertEqual(adopted['identity'],ident)
            self.assertNotIn('runtime_env',adopted)
            m.handle('stop',{})
            identity.assert_called_with(123,True,ident)
        self.assertNotIn('identity',m.read_json(m.DATA/'process.json',{}))
    def test_a_stop_with_the_disk_tier_on_first_has_the_server_write_its_conversations(self):
        # the tier writes a conversation's recurrent state only when it leaves memory, and a stop is that
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        calls=[]
        class Res:
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def read(self):return b'{"success":true}'
        def opener(req,timeout=None):
            calls.append((req.full_url,req.get_method()));return Res()
        for disk,asked in ((True,1),(False,0)):
            calls.clear()
            m.atomic_json(m.DATA/'process.json',{'identity':ident,'adopted':False,'profile':{'prompt_cache_disk':disk}})
            with patch.object(m,'process_identity',return_value=ident) as identity,patch.object(m.HTTP,'open',side_effect=opener):
                m.handle('stop',{});identity.assert_called_with(123,True,ident)
            self.assertEqual(len(calls),asked)
            if asked:self.assertEqual(calls[0],('http://127.0.0.1:8080/strix/persist','POST'))
        # a runtime without the endpoint, or one that does not answer, is stopped all the same
        m.atomic_json(m.DATA/'process.json',{'identity':ident,'adopted':False,'profile':{'prompt_cache_disk':True}})
        with patch.object(m,'process_identity',return_value=ident) as identity,patch.object(m.HTTP,'open',side_effect=m.urllib.error.URLError('404')):
            m.handle('stop',{});identity.assert_called_with(123,True,ident)

    def test_defaults_are_the_measured_configuration_per_architecture(self):
        model=first_model()
        cfg=m.profile(model)
        # the numbers in docs/results.md, so a first load performs as claimed
        self.assertEqual((cfg['context'],cfg['batch'],cfg['ubatch'],cfg['flash_attention']),(32768,8192,8192,'on'))  # context clamped to the model's declared length
        self.assertTrue(cfg['qsa'])
        # MTP follows the model family in the file name, since the draft head is that family's
        family={**model,'path':str(self.file.with_name(m.MODEL_FAMILY+'-UD-IQ4_XS.gguf'))}
        self.assertTrue(m.profile(family)['mtp']);self.assertFalse(cfg['mtp'])
        # sparse attention is qwen4exp's; another architecture must not be handed a profile that fails validation
        other={**model,'architecture':'qwen35moe','path':str(self.file.with_name('Other-35B-Q4.gguf'))}
        self.assertFalse(m.profile(other)['qsa']);self.assertFalse(m.profile(other)['mtp'])
        m.validate_profile(m.profile(other),other)
    def test_qsa_legacy_profile_keeps_its_saved_fields(self):
        model=first_model()
        settings=m.settings();settings['profiles'][model['id']]={'context':16384,'flash_attention':'on'}
        m.atomic_json(m.DATA/'settings.json',settings)
        cfg=m.validate_profile({},model)
        self.assertTrue(cfg['qsa'])
        self.assertEqual(m.selected_runtime(cfg),m.RUNTIME)
        self.assertEqual(cfg['context'],16384)
    def test_qsa_invalid_dependencies_rejected(self):
        for values in ({'qsa':'true'},{'qsa':1},{'qsa':None},
                       {'qsa':True,'flash_attention':'off'},
                       {'qsa':True,'flash_attention':'off'}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                m.validate_profile(values,self.model)
        with self.assertRaisesRegex(ValueError,'qwen4exp'):
            m.validate_profile({'qsa':True,'flash_attention':'on'},{**self.model,'architecture':'qwen35moe'})
    def test_qsa_save_start_stop_roundtrip_preserves_model_settings(self):
        from types import SimpleNamespace
        model=first_model()
        baseline=m.validate_profile({'context':16384,'flash_attention':'on','mtp':True,
                                     'draft':str(self.file),'draft_max':3,'draft_min':0.35},model)
        settings=m.settings();settings['profiles'][model['id']]=baseline
        m.atomic_json(m.DATA/'settings.json',settings)
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        with patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]),              patch.object(m,'process_identity',return_value=ident) as identity,              patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')),              patch('socket.socket'),patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as popen:
            popen.reset_mock()
            m.handle('save',{'id':model['id'],'profile':{'qsa':True}})
            popen.assert_not_called()
            cfg=m.profile(model)
            self.assertEqual({**cfg,'qsa':baseline['qsa']},baseline)
            result=m.handle('start',{'id':model['id']})
            args=popen.call_args.args[0];env=popen.call_args.kwargs['env']
            self.assertEqual(args[0],str(m.RUNTIME))
            for option,value in (('-fa','on'),('-ctk','f16'),('-ctv','f16'),
                                 ('--spec-draft-n-max','3'),('--spec-draft-p-min','0.35')):
                self.assertEqual(args[args.index(option)+1],value)
            # this runtime has no context threshold: the gates are simply on
            self.assertEqual(env['LLAMA_QSA_SPARSE'],'1')
            self.assertEqual(env['LLAMA_QSA_QUERY_STRIP'],'512')
            self.assertEqual(result['runtime_env']['LLAMA_QSA_SPARSE'],'1')
            self.assertIn('QSA=on',Path(result['log']).read_text())
            self.assertEqual(m.status()['runtime_env'],result['runtime_env'])
            m.handle('stop',{});identity.assert_called_with(123,True,ident)
            m.handle('save',{'id':model['id'],'profile':{'qsa':False}})
            self.assertEqual(m.selected_runtime(m.profile(model)),m.RUNTIME)

    def test_log_polling_preserves_partial_utf8(self):
        log=self.root/'logs'/'unicode.log';log.parent.mkdir()
        log.write_bytes(b'line\n'+'中'.encode()[:2])
        with patch.object(m,'ROOT',self.root),patch.object(m,'state',return_value={'log':str(log)}):
            first=m.logs();self.assertEqual(first['text'],'line\n')
            with log.open('ab') as f:f.write('中'.encode()[2:])
            self.assertEqual(m.logs(first['offset'])['text'],'中')
    @unittest.skipUnless(os.name=='nt','Windows process identity')
    def test_process_guard_rejects_pid_reuse_and_other_executable(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'],creationflags=m.HIDDEN)
        try:
            ident=m.process_identity(child.pid)
            self.assertIsNotNone(ident)
            with self.assertRaises(ValueError):m.process_identity(child.pid,True,{**ident,'birth':ident['birth']+1})
            with self.assertRaises(ValueError):m.process_identity(child.pid,True,ident)
            self.assertIsNone(child.poll())
        finally:
            child.terminate();child.wait(timeout=5)
    def test_draft_head_merge_splices_the_tensor_without_decoding_anything(self):
        base=self.root/'models'/'mtp-Fam-shared-Q4_K_M.gguf';gguf_tensors(base,[('a',[4],0,b'A'*16),('b',[2],0,b'B'*8)])
        head=self.root/'models'/'mtp-Fam-head-iq4_xs.gguf';gguf_tensors(head,[('output.weight',[8,2],1,b'H'*32)])
        out=self.root/'models'/'merged.gguf'
        m.merge_draft_head(base,head,out)
        lay=m.gguf_layout(out)
        self.assertEqual([t[0] for t in lay['tensors']],['a','b','output.weight'])
        self.assertEqual(lay['tensors'][2][1:],([8,2],1,64))   # after the 40 base bytes, aligned to 32
        data=out.read_bytes()[lay['data_start']:]
        self.assertEqual(data[:40],b'A'*16+b'\0'*16+b'B'*8);self.assertEqual(data[64:96],b'H'*32)
        self.assertEqual(m.metadata(out)['general.architecture'],'qwen4exp')
        self.assertFalse(out.with_suffix('.part').exists())
        # only a lone output.weight may be appended, and only once
        two=self.root/'models'/'mtp-Fam-head-two.gguf';gguf_tensors(two,[('output.weight',[2],0,b'x'*8),('other',[2],0,b'y'*8)])
        with self.assertRaises(ValueError) as caught:m.merge_draft_head(base,two,self.root/'models'/'x.gguf')
        self.assertEqual(caught.exception.code,'head_mismatch')
        with self.assertRaises(ValueError):m.merge_draft_head(out,head,self.root/'models'/'y.gguf')
    def test_rescan_merges_a_downloaded_head_with_the_shared_draft_once_and_prefers_it(self):
        head=self.file.with_name('mtp-'+m.MODEL_FAMILY+'-head-iq4_xs.gguf');gguf_tensors(head,[('output.weight',[4],0,b'H'*16)])
        merged=self.draft.with_name(self.draft.stem+'-head-iq4_xs.gguf')
        with patch.object(m,'DEFAULT_DRAFT',self.root/'models'/'missing-head.gguf'):
            c=m.catalog(True)
            self.assertEqual([Path(p).resolve() for p in c['merged']],[merged.resolve()]);self.assertEqual(c['merge_errors'],{})
            roles={x['filename']:x['role'] for x in c['models']}
            self.assertEqual(roles[head.name],'head');self.assertEqual(roles[merged.name],'draft')
            self.assertEqual(Path(m.family_draft()).resolve(),merged.resolve())
            model={**self.model,'path':str(self.file.with_name(m.MODEL_FAMILY+'-UD-IQ4_XS.gguf'))}
            self.assertEqual(Path(m.profile(model)['draft']).resolve(),merged.resolve())
            self.assertEqual(m.catalog(True)['merged'],[])   # already there: not written again
            # a head alone, nothing to merge it with, is listed and left alone
            self.draft.unlink();merged.unlink()
            c=m.catalog(True);self.assertEqual(c['merged'],[]);self.assertIn(head.name,{x['filename'] for x in c['models']})
            self.assertFalse(m.profile(model)['mtp'])
    def test_errors_carry_a_code_for_the_app_to_translate(self):
        with self.assertRaises(m.ManagerError) as caught:m.validate_profile({'context':1},self.model)
        self.assertEqual((caught.exception.code,caught.exception.params),('out_of_range',{'field':'context','low':512,'high':262144}))
        self.assertIn('between 512 and 262144',str(caught.exception))
        for code in m.ERRORS: m.ERRORS[code].format(**{k:'' for k in ('field','low','high','levels','name','head','base','limit','context','port','host')})
    def test_a_kv_pool_larger_than_the_context_holds_more_conversations_each_capped_at_the_context(self):
        cfg=m.validate_profile({'parallel':4,'vision':False,'mtp':False,'kv_pool':40000},self.model)
        args=m.argv(self.model,cfg)
        self.assertEqual(args[args.index('-c')+1],'40192')   # one allocation, rounded up to 256 cells
        self.assertEqual(args[args.index('--kv-unified-per-slot')+1],str(cfg['context']))
        # 0, or a single slot: the pool is the context and nothing caps the slots
        for raw in ({'parallel':4,'vision':False,'mtp':False},{'parallel':1,'mtp':False,'kv_pool':40000}):
            args=m.argv(self.model,m.validate_profile(raw,self.model))
            self.assertEqual(args[args.index('-c')+1],str(m.profile(self.model)['context']))
            self.assertNotIn('--kv-unified-per-slot',args)
        with self.assertRaises(m.ManagerError) as caught:m.validate_profile({'parallel':4,'vision':False,'kv_pool':1000},self.model)
        self.assertEqual(caught.exception.code,'kv_pool_below_context')
    def test_the_draft_ubatch_shrinks_with_a_kv_pool_past_512k_cells(self):
        # the draft reserves a dense mask over the whole pool for its ubatch; past 512K cells a smaller one keeps it there
        self.assertEqual([m.draft_ubatch(p) for p in (131072,524288,786432,1048576)],[2048,2048,1280,1024])
        cfg=dict(m.profile(self.model),parallel=4,mtp=True,kv_pool=786432)
        self.assertEqual(m.runtime_environment(cfg)['STRIX_SPEC_DRAFT_UBATCH'],'1280')
        cfg['kv_pool']=0
        self.assertEqual(m.runtime_environment(cfg)['STRIX_SPEC_DRAFT_UBATCH'],'2048')
    def test_disk_prompt_cache_is_a_switch_that_sets_the_server_environment(self):
        # off unless asked for: nothing goes to the SSD by default
        self.assertIs(m.validate_profile({'mtp':False},self.model)['prompt_cache_disk'],False)
        on=m.runtime_environment(m.validate_profile({'mtp':False,'prompt_cache_disk':True},self.model))
        self.assertEqual(Path(on['STRIX_PROMPT_CACHE_DIR']),m.DATA/'prompt-cache');self.assertEqual(on['STRIX_PROMPT_CACHE_MIB'],str(m.PROMPT_CACHE_DISK_MIB))
        off=m.runtime_environment(m.validate_profile({'mtp':False,'prompt_cache_disk':False},self.model))
        self.assertNotIn('STRIX_PROMPT_CACHE_DIR',off)
        with self.assertRaises(ValueError):m.validate_profile({'prompt_cache_disk':'yes'},self.model)
    def test_idle_slots_stay_warm_and_the_disk_tier_writes_in_blocks(self):
        cfg=m.validate_profile({'parallel':4,'vision':False,'prompt_cache_disk':True},self.model)
        self.assertIn('--no-cache-idle-slots',m.argv(self.model,cfg))
        env=m.runtime_environment(cfg)
        self.assertEqual(env['STRIX_PROMPT_CACHE_BLOCK'],str(m.PROMPT_CACHE_BLOCK_TOKENS))
        self.assertNotIn('STRIX_PROMPT_CACHE_BLOCK',m.runtime_environment(m.validate_profile({'prompt_cache_disk':False},self.model)))

    def test_drafts_shrink_as_more_slots_generate(self):
        env=lambda raw: m.runtime_environment(m.validate_profile(dict({'vision':False},**raw),self.model))
        # one slot: nothing to cap; several: draft_max for one generating, at most 2 for two or three, none from four
        self.assertNotIn('STRIX_SPEC_DRAFT_BY_SLOTS',env({'parallel':1}))
        if m.validate_profile({'vision':False},self.model)['mtp']:
            self.assertEqual(env({'parallel':4})['STRIX_SPEC_DRAFT_BY_SLOTS'],'3,2,2,0')
            self.assertEqual(env({'parallel':4,'draft_max':1})['STRIX_SPEC_DRAFT_BY_SLOTS'],'1,1,1,0')
        self.assertNotIn('STRIX_SPEC_DRAFT_BY_SLOTS',env({'parallel':4,'mtp':False}))
    def test_sampled_requests_draft_less(self):
        env=lambda raw: m.runtime_environment(m.validate_profile(dict({'vision':False},**raw),self.model))
        # their own table, also with one slot: 2 drafts for one or two generating, 1 for three or four, none from five
        if m.validate_profile({'vision':False},self.model)['mtp']:
            self.assertEqual(env({'parallel':1})['STRIX_SPEC_DRAFT_BY_SLOTS_SAMPLED'],'2,2,1,1,0')
            self.assertEqual(env({'parallel':4,'draft_max':1})['STRIX_SPEC_DRAFT_BY_SLOTS_SAMPLED'],'1,1,1,1,0')
        self.assertNotIn('STRIX_SPEC_DRAFT_BY_SLOTS_SAMPLED',env({'parallel':4,'mtp':False}))
    def test_the_kv_cache_is_f16_or_q8_0_and_reaches_both_type_flags(self):
        for kv in m.KV_TYPES:
            a=m.argv(self.model,m.validate_profile({'vision':False,'kv':kv},self.model))
            self.assertEqual((a[a.index('-ctk')+1],a[a.index('-ctv')+1]),(kv,kv))
    def test_a_resident_conversation_keeps_its_last_checkpoints_and_a_sparse_few(self):
        # 0.11 GB each in system RAM: the last prompt's and one per 32K tokens, not llama-server's 32 per slot
        a=m.argv(self.model,m.validate_profile({'vision':False},self.model))
        self.assertEqual(a[a.index('--ctx-checkpoints')+1],str(m.CTX_CHECKPOINTS));self.assertEqual(m.CTX_CHECKPOINTS,8)
        self.assertEqual(a[a.index('--checkpoint-min-step')+1],str(m.CHECKPOINT_MIN_STEP));self.assertEqual(m.CHECKPOINT_MIN_STEP,4096)
    def test_the_disk_prompt_cache_ceiling_is_a_profile_field(self):
        env=m.runtime_environment(m.validate_profile({'prompt_cache_disk':True,'prompt_cache_disk_mib':204800},self.model))
        self.assertEqual(env['STRIX_PROMPT_CACHE_MIB'],'204800')
        # a profile saved before the field existed keeps the old ceiling
        env=m.runtime_environment(m.validate_profile({'prompt_cache_disk':True},self.model))
        self.assertEqual(env['STRIX_PROMPT_CACHE_MIB'],str(m.PROMPT_CACHE_DISK_MIB))
        with self.assertRaises(m.ManagerError) as caught:
            m.validate_profile({'prompt_cache_disk_mib':512},self.model)
        self.assertEqual(caught.exception.code,'out_of_range')

    def test_more_than_one_slot_keeps_the_full_context_and_the_ubatch(self):
        # since the mixed-sequence graphs took the sparse path there is no dense mask to size: four
        # slots at 262144 x 8192 load and cost ~1.3 GB over one (docs/results/concurrency-mtp-20260921.json)
        big={**self.model,'context':262144}
        cfg=m.validate_profile({'mtp':False,'vision':False,'context':262144,'parallel':4},big)
        self.assertEqual((cfg['context'],cfg['ubatch'],cfg['parallel']),(262144,8192,4))
        self.assertIn('-kvu',m.argv(big,cfg))
        self.assertNotIn('-kvu',m.argv(big,m.validate_profile({'mtp':False,'vision':False,'context':262144,'parallel':1},big)))

    def test_image_input_with_several_slots(self):
        # since 0.2.6 the sparse attention ranks an image's cells per sequence, so image input and several slots
        # go together, and four slots are the default: the conversations they hold stay resident
        self.assertEqual(m.DEFAULTS['parallel'],8)
        cfg=m.validate_profile({'mtp':False,'parallel':4},self.model)
        self.assertTrue(cfg['vision']);self.assertEqual(cfg['parallel'],4)
        self.assertTrue(m.validate_profile({'mtp':False,'parallel':1},self.model)['vision'])

    # Configuration › Network (GitHub issue #6): port, local network, API key, kept in settings.json
    def save_network(self,**net):
        # state() would look for a running llama-server on this machine; there is none in these tests
        with patch.object(m,'state',return_value={}),patch.object(m,'lan_addresses',return_value=[]):
            return m.handle('save_network',net)
    def test_without_a_network_setting_the_server_listens_on_loopback_8080_without_a_key(self):
        # a settings.json from before the setting: where the server always listened
        net=m.network()
        self.assertEqual((net['host'],net['port'],net['lan'],net['api_key'],net['forced']),('127.0.0.1',8080,False,'',{}))
        a=m.argv(self.model,m.validate_profile({'mtp':False},self.model))
        self.assertEqual((a[a.index('--host')+1],a[a.index('--port')+1]),('127.0.0.1','8080'));self.assertNotIn('--api-key',a)
        with patch.object(m,'state',return_value={}):s=m.status()
        self.assertEqual((s['endpoint'],s['api_key_set']),('http://127.0.0.1:8080/v1',False));self.assertNotIn('lan_endpoints',s)
    def test_a_saved_network_setting_reaches_the_launch_arguments_and_the_status(self):
        r=self.save_network(port=13305,lan=True,api_key='s3cret-Key_1')
        self.assertFalse(r['restart_required'])
        self.assertEqual(r['network']['saved'],{'port':13305,'lan':True,'api_key':'s3cret-Key_1'})
        self.assertEqual(r['network']['server'],{'endpoint':'http://127.0.0.1:13305/v1','api_key':'s3cret-Key_1'})
        saved=m.read_json(m.DATA/'settings.json',{})
        self.assertEqual(saved['network'],{'port':13305,'lan':True,'api_key':'s3cret-Key_1'});self.assertEqual(saved['profiles'],{})
        net=m.network();self.assertEqual((net['host'],net['port'],net['api_key']),('0.0.0.0',13305,'s3cret-Key_1'))
        a=m.argv(self.model,m.validate_profile({'mtp':False},self.model))
        self.assertEqual([a[a.index(o)+1] for o in ('--host','--port','--api-key')],['0.0.0.0','13305','s3cret-Key_1'])
        # bound to every address and reached on loopback; other devices use this PC's addresses; the key is never in it
        with patch.object(m,'state',return_value={}),patch.object(m,'lan_addresses',return_value=['192.168.1.20']):s=m.status()
        self.assertEqual((s['endpoint'],s['lan_endpoints'],s['api_key_set']),
                         ('http://127.0.0.1:13305/v1',['http://192.168.1.20:13305/v1'],True))
        self.assertNotIn('s3cret',json.dumps(s))
    def test_rulith_variables_override_the_saved_network_setting(self):
        self.save_network(port=13305,lan=True,api_key='filekey')
        with patch.dict(m.os.environ,{'RULITH_PORT':'13306','RULITH_HOST':'127.0.0.1','RULITH_API_KEY':'envkey'}):
            net=m.network()
            self.assertEqual((net['host'],net['port'],net['lan'],net['api_key']),('127.0.0.1',13306,False,'envkey'))
            self.assertEqual(net['forced'],{'port':'RULITH_PORT','lan':'RULITH_HOST','api_key':'RULITH_API_KEY'})
            a=m.argv(self.model,m.validate_profile({'mtp':False},self.model))
            self.assertEqual([a[a.index(o)+1] for o in ('--host','--port','--api-key')],['127.0.0.1','13306','envkey'])
            # the page edits the file's values, and shows what overrides them
            with patch.object(m,'state',return_value={}),patch.object(m,'lan_addresses',return_value=[]):view=m.handle('network',{})
            self.assertEqual(view['saved'],{'port':13305,'lan':True,'api_key':'filekey'})
            self.assertEqual((view['port'],view['host'],view['api_key'],view['server']['endpoint']),(13306,'127.0.0.1','envkey','http://127.0.0.1:13306/v1'))
        # a variable can also name the address to bind, and an empty one is not set
        with patch.dict(m.os.environ,{'RULITH_HOST':'192.168.1.20','RULITH_PORT':'','RULITH_API_KEY':' '}):
            net=m.network();self.assertEqual((net['host'],net['lan'],net['port'],net['api_key']),('192.168.1.20',True,13305,'filekey'))
            self.assertEqual(m.base_url(net),'http://192.168.1.20:13305')
        net=m.network();self.assertEqual((net['host'],net['port'],net['api_key'],net['forced']),('0.0.0.0',13305,'filekey',{}))
    def test_network_values_are_validated_before_they_are_saved_or_used(self):
        for raw,code in (({'port':80},'out_of_range'),({'port':65536},'out_of_range'),({'port':'8080'},'out_of_range'),
                         ({'port':True},'out_of_range'),({'lan':'yes'},'not_boolean'),({'lan':1},'not_boolean'),
                         ({'api_key':'two words'},'api_key_format'),({'api_key':'a,b'},'api_key_format'),
                         ({'api_key':'say"hi'},'api_key_format'),({'api_key':'ключ'},'api_key_format'),
                         ({'api_key':'k'*257},'api_key_format'),({'api_key':None},'api_key_format'),
                         ({'host':'0.0.0.0'},'unknown_field'),([13305],'unknown_field')):
            with self.subTest(raw=raw):
                with self.assertRaises(m.ManagerError) as caught:
                    with patch.object(m,'state',return_value={}):m.handle('save_network',raw)
                self.assertEqual(caught.exception.code,code)
        self.assertNotIn('network',m.read_json(m.DATA/'settings.json',{}))
        self.assertEqual(m.check_network({'port':1024,'api_key':'k'*256}),{'port':1024,'lan':False,'api_key':'k'*256})
        for env,code,params in ((('RULITH_PORT','http'),'out_of_range',{'field':'RULITH_PORT','low':1024,'high':65535}),
                                (('RULITH_PORT','99999'),'out_of_range',{'field':'RULITH_PORT','low':1024,'high':65535}),
                                (('RULITH_HOST','localhost'),'host_address',{}),(('RULITH_HOST','::'),'host_address',{}),
                                (('RULITH_API_KEY','a b'),'api_key_format',{'field':'RULITH_API_KEY'})):
            with self.subTest(env=env):
                with patch.dict(m.os.environ,dict([env])),self.assertRaises(m.ManagerError) as caught:m.network()
                self.assertEqual((caught.exception.code,caught.exception.params),(code,params))
        # a field a later version added to the file is dropped, not refused
        settings=m.settings();settings['network']={'port':13305,'cors':'*'};m.atomic_json(m.DATA/'settings.json',settings)
        self.assertEqual(m.saved_network(),{'port':13305,'lan':False,'api_key':''})
    def test_the_api_key_reaches_the_server_and_nothing_that_is_kept_or_shown(self):
        model=first_model()
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        self.save_network(port=13305,lan=False,api_key='s3cret')
        stack,popen=self.running([ident])
        with stack:
            self.assertNotIn('s3cret',json.dumps(m.handle('save',{'id':model['id'],'profile':{'mtp':False}})))
            result=m.handle('start',{'id':model['id']})
            args=popen.call_args.args[0]
            self.assertEqual(args[args.index('--api-key')+1],'s3cret')
            kept=m.read_json(m.DATA/'process.json',{})
            self.assertIn('--api-key ***',kept['command']);self.assertNotIn('s3cret',kept['command'])
            self.assertNotIn('api_key',result)
            for shown in (json.dumps(result),json.dumps(m.status()),Path(result['log']).read_text()):
                self.assertNotIn('s3cret',shown)
            self.assertIn('listen=127.0.0.1:13305 (API key required)',Path(result['log']).read_text())
            self.assertTrue(m.status()['api_key_set'])
            m.handle('stop',{})
        self.assertNotIn('s3cret',json.dumps(m.read_json(m.DATA/'process.json',{})))
    def test_the_manager_talks_to_the_running_server_where_it_listens_with_its_key(self):
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        calls=[]
        class Res:
            headers={}
            def __enter__(s):return s
            def __exit__(s,*a):return False
            def read(s,*a):return b'{"status":"ok","data":[]}'
        def opener(req,timeout=None):
            calls.append((req.full_url,req.get_header('Authorization')));return Res()
        self.save_network(port=13305,lan=True,api_key='k3y')
        m.atomic_json(m.DATA/'process.json',{'identity':ident,'adopted':False,'host':'0.0.0.0','port':13305,'api_key':'k3y',
                                             'profile':{'prompt_cache_disk':True},'command':'llama-server.exe --host 0.0.0.0 --port 13305 --api-key ***'})
        with patch.object(m,'process_identity',return_value=ident),patch.object(m.HTTP,'open',side_effect=opener), \
             patch.object(m,'lan_addresses',return_value=['10.0.0.5']):
            s=m.status();m.slots()
            self.assertEqual((s['status'],s['endpoint'],s['lan_endpoints'],s['network_pending']),
                             ('ready','http://127.0.0.1:13305/v1',['http://10.0.0.5:13305/v1'],False))
            # a setting saved while it runs is for the next load: it is still reached where it listens, and the pages say so
            r=m.handle('save_network',{'port':14000,'lan':False,'api_key':''})
            self.assertTrue(r['restart_required'])
            self.assertEqual((r['network']['port'],r['network']['server']),(14000,{'endpoint':'http://127.0.0.1:13305/v1','api_key':'k3y'}))
            s=m.status()
            self.assertEqual((s['endpoint'],s['api_key_set'],s['network_pending']),('http://127.0.0.1:13305/v1',True,True))
            m.handle('stop',{})
        self.assertEqual({url.split('/')[2] for url,_ in calls},{'127.0.0.1:13305'});self.assertEqual({key for _,key in calls},{'Bearer k3y'})
        self.assertEqual([url.split('13305')[1] for url,_ in calls],['/health','/v1/models','/slots','/health','/v1/models','/strix/persist'])
        # unloaded, the next load's address is the one shown
        with patch.object(m,'state',return_value={}):self.assertEqual(m.status()['endpoint'],'http://127.0.0.1:14000/v1')
    def test_a_server_an_earlier_version_started_is_still_reached_on_8080(self):
        # its process.json has no host, port or key, only the command it was launched with
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        m.atomic_json(m.DATA/'process.json',{'identity':ident,'adopted':False,'log':'x.log',
                                             'command':'llama-server.exe -m x.gguf --host 127.0.0.1 --port 8080 --jinja'})
        self.save_network(port=13305,lan=False,api_key='')
        with patch.object(m,'process_identity',return_value=ident),patch.object(m,'http_json',return_value={'status':'ok','data':[]}) as get:
            s=m.status()
        self.assertEqual(m.listening(m.read_json(m.DATA/'process.json',{})),{'host':'127.0.0.1','port':8080,'api_key':''})
        self.assertEqual((s['endpoint'],s['network_pending']),('http://127.0.0.1:8080/v1',True))
        self.assertEqual(get.call_args.args[1]['port'],8080)
    def test_a_server_on_the_configured_port_is_adopted_on_either_host_with_its_key(self):
        other=str((self.root/'elsewhere'/'hip'/'llama-server.exe').resolve())
        ident={'pid':321,'exe':other,'birth':7}
        flags='--load-mode none --lazy-mode on-direct'
        self.save_network(port=13305,lan=True,api_key='')
        for host in ('127.0.0.1','0.0.0.0'):
            entry={'ProcessId':321,'ExecutablePath':other,'CommandLine':f'llama-server.exe -m x.gguf --host {host} --port 13305 --api-key k3y {flags}'}
            m.atomic_json(m.DATA/'process.json',{})
            with self.subTest(host=host),patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident):
                s=m.state()
                self.assertEqual((s['identity'],s['host'],s['port']),(ident,host,13305))
                self.assertIn('--api-key ***',s['command']);self.assertNotIn('k3y',s['command'])
                # its own key, for the manager's requests to it
                self.assertEqual(m.listening(s),{'host':host,'port':13305,'api_key':'k3y'})
        # not on another port or another address: an earlier version's 127.0.0.1:8080 is ours while 8080 is the port
        for cmd in ('--host 127.0.0.1 --port 8080','--host 192.168.1.9 --port 13305'):
            entry={'ProcessId':321,'ExecutablePath':other,'CommandLine':f'llama-server.exe -m x.gguf {cmd} {flags}'}
            m.atomic_json(m.DATA/'process.json',{})
            with self.subTest(cmd=cmd),patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident):
                self.assertNotIn('identity',m.state())
        self.save_network(port=8080,lan=False,api_key='')
        with patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident):
            entry['CommandLine']=f'llama-server.exe -m x.gguf --host 127.0.0.1 --port 8080 {flags}'
            s=m.state();self.assertEqual((s['identity'],s['api_key']),(ident,''))
    def test_a_busy_port_is_named_and_a_lan_load_needs_loopback_free_too(self):
        model=first_model()
        self.save_network(port=13305,lan=True,api_key='')
        with patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]),patch.object(m.subprocess,'Popen') as popen:
            # something already answers there
            with patch.object(m,'http_json',return_value={'status':'ok'}) as probe,self.assertRaises(m.ManagerError) as caught:
                m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            self.assertEqual((caught.exception.code,caught.exception.params),('port_busy',{'port':13305}))
            self.assertIn('Port 13305 ',str(caught.exception));self.assertEqual(probe.call_args.args[1]['port'],13305)
            # nothing answers, but the port is taken on loopback: a server on all addresses would not get its requests
            bound=[]
            class Sock:
                def __init__(s,*a):pass
                def __enter__(s):return s
                def __exit__(s,*a):return False
                def bind(s,addr):
                    bound.append(addr)
                    if addr[0]=='127.0.0.1':raise OSError(10048,'in use')
                    if addr[0]=='192.168.1.9':raise OSError(10049,'not an address of this PC')
            with patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')),patch('socket.socket',Sock):
                with self.assertRaises(m.ManagerError) as caught:m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
                self.assertEqual((caught.exception.code,bound),('port_busy',[('0.0.0.0',13305),('127.0.0.1',13305)]))
                # an address RULITH_HOST names that this PC does not have is not a busy port
                bound.clear()
                with patch.dict(m.os.environ,{'RULITH_HOST':'192.168.1.9'}),self.assertRaises(m.ManagerError) as caught:
                    m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
                self.assertEqual((caught.exception.code,caught.exception.params,bound),('host_unavailable',{'host':'192.168.1.9'},[('192.168.1.9',13305)]))
            popen.assert_not_called()
    def test_lan_addresses_leave_out_loopback_and_link_local(self):
        class Probe:
            def __init__(s,*a):pass
            def __enter__(s):return s
            def __exit__(s,*a):return False
            def connect(s,addr):pass
            def getsockname(s):return ('192.168.1.20',50000)
        found=[(2,2,17,'',(a,0)) for a in ('127.0.0.1','169.254.3.4','10.0.0.5','192.168.1.20')]
        with patch('socket.socket',Probe),patch('socket.getaddrinfo',return_value=found):
            self.assertEqual(m.lan_addresses(),['192.168.1.20','10.0.0.5'])
        with patch('socket.socket',side_effect=OSError('no route')),patch('socket.getaddrinfo',side_effect=OSError('no name')):
            self.assertEqual(m.lan_addresses(),[])

if __name__=='__main__':unittest.main()
