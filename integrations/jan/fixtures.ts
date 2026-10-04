// Browser preview only (`yarn dev:web`, no Tauri): canned answers for the manager's ops, so the pages
// can be looked at and screenshotted without the app or a GPU. status.ts imports this module behind
// import.meta.env.DEV, so production builds never contain it.
//
// ?state=ready|loading|stopped|failed|commit|memory|none picks the starting state (none: no model files; memory:
// answers stopped because Windows ran out of commit);
// start and stop then move it along as the real server would, loading for three seconds.
const ROOT = 'D:\\models\\unsloth\\Qwen3.8-Flash-Next-GGUF\\'
const MODEL = ROOT + 'Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
const DRAFT = ROOT + 'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf'

const models = [
  { id: 'm1', path: MODEL, name: 'Qwen3.8 Flash Next', filename: 'Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf', architecture: 'qwen4exp', quant: 'IQ4_XS', size: 93682583224, shards: 3, role: 'model', context: 262144 },
  { id: 'm2', path: ROOT + 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', name: 'Qwen3.8 Flash Next', filename: 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', architecture: 'qwen4exp', quant: 'Q4_K_XL', size: 111334659784, shards: 4, role: 'model', context: 262144 },
  { id: 'd1', path: DRAFT, name: 'Qwen3.8 Flash Next MTP', filename: 'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf', architecture: 'qwen4exp_mtp', quant: 'Q4_K_M', size: 2244867200, shards: 1, role: 'draft' },
  { id: 'd2', path: ROOT + 'mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf', name: 'Qwen3.8 Flash Next MTP', filename: 'mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf', architecture: 'qwen4exp_mtp', quant: 'Q8_0', size: 2786568256, shards: 1, role: 'draft' },
  { id: 'p1', path: ROOT + 'mmproj-F16.gguf', name: 'Qwen3-VL projector', filename: 'mmproj-F16.gguf', architecture: 'clip', quant: 'F16', size: 904004000, shards: 1, role: 'projection' },
]

const saved: Record<string, Record<string, unknown>> = {}
let tick = 0
const baseProfile = { thinking: 'medium', context: 262144, gpu_layers: 999, threads: 16, batch: 8192, ubatch: 8192, mtp: true, draft: DRAFT, draft_max: 3, draft_min: 0.6, ngram_spec: false, kv: 'f16', flash_attention: 'on', qsa: true, trunk_decode_q6k: false, parallel: 8, kv_pool: 0, vision: false, mmproj: '', prompt_cache_disk: true, prompt_cache_disk_mib: 65536 }
const profileOf = (id: string) => ({ ...baseProfile, ...(saved[id] || {}) })

const initial = new URLSearchParams(location.search).get('state') || 'ready'
let state = initial === 'commit' || initial === 'memory' ? 'ready' : initial === 'none' ? 'stopped' : initial
let running = state === 'ready' || state === 'loading' ? 'm1' : ''
let runningProfile: Record<string, unknown> = profileOf('m1')

// Configuration › Network: what is saved, and what the running server was started with
let network = { port: 8080, lan: false, api_key: '' }
let listening = { ...network }
const ADDRESSES = ['192.168.1.20']
const serverNet = () => (running ? listening : network)
const lanEndpoints = (n: typeof network) => (n.lan ? ADDRESSES.map(a => `http://${a}:${n.port}/v1`) : [])
const networkView = () => ({
  saved: { ...network }, ...network, host: network.lan ? '0.0.0.0' : '127.0.0.1', forced: {}, addresses: ADDRESSES,
  server: { endpoint: `http://127.0.0.1:${serverNet().port}/v1`, api_key: serverNet().api_key },
})

const status = () => {
  const m = models.find(x => x.id === running)
  const n = serverNet()
  return {
    status: state === 'failed' ? 'stopped' : state, endpoint: `http://127.0.0.1:${n.port}/v1`,
    api_key_set: !!n.api_key, ...(n.lan ? { lan_endpoints: lanEndpoints(n) } : {}),
    ...(running ? { network_pending: JSON.stringify(listening) !== JSON.stringify(network) } : {}),
    runtime: 'C:\\Users\\me\\AppData\\Local\\Rulith Inference\\runtime\\bin\\hip\\llama-server.exe', runtime_available: true,
    runtime_info: { rocm: '10.2.0a20260925', gfx: 'gfx1151' },
    runtime_env: { LLAMA_QSA_SPARSE: '1', STRIX_SPEC_DRAFT_BY_SLOTS: '3,2,2,2,0' },
    ...(m ? { identity: { pid: 23456 }, model_path: m.path, model_name: m.name, profile: runningProfile,
      command: `llama-server.exe -m ${m.path} -c ${runningProfile.context} -ngl 999 -fa on -b 8192 -ub 8192 --parallel 8 --model-draft ${DRAFT} --draft-max 3 --draft-p-min 0.6` } : {}),
    ...(state === 'failed' ? { failure: 'The GPU ran out of memory', failure_code: 'oom' } : {}),
    // the server answers under the file's name, so a switch changes the id the chat talks to
    ...(state === 'ready' && m ? { served_models: [{ id: m.filename.replace(/(-\d{5}-of-\d{5})?\.gguf$/, '') }] } : {}),
    dedicated_vram: 103079215104, commit_limit: 180e9, commit_available: 3.2e9, commit_low: initial === 'commit',
    ...(initial === 'memory' && m ? { memory_events: { answers_stopped: 4, saves_skipped: 1, last_stop: '2026-09-26T13:33:27', last_skip: '2026-09-26T17:14:12' } } : {}),
  }
}

const LOG = [
  '0.00.012.101 I build: 7ce18660 (strixllama) with clang 20 for Windows x86_64',
  '0.00.030.612 I srv    load_model: loading model \'' + MODEL + '\'',
  '0.02.114.880 I llama_model_loader: loaded meta data with 52 key-value pairs and 1203 tensors',
  '0.18.402.113 I llama_context: n_ctx = 262144, n_batch = 8192, n_ubatch = 8192, flash_attn = enabled',
  '0.18.990.004 W srv    load_model: speculative decoding with the MTP head, draft_max = 3',
  '0.21.301.772 I srv  main: server is listening on http://127.0.0.1:8080 - starting the main loop',
  '1.02.455.664 I slot print_timing: id  0 | task 12 | prompt eval time = 3480.12 ms / 4117 tokens (1183.0 tokens per second)',
  '1.02.455.899 I slot print_timing: id  0 | task 12 | eval time = 12512.40 ms / 512 tokens (40.9 tokens per second)',
  '1.02.456.001 I slot print_timing: id  0 | task 12 | draft acceptance = 0.66102 (  332 accepted /   502 generated), mean len =  2.90',
  '1.05.101.200 E srv    operator(): got exception: {"error":{"code":500,"message":"example error line"}}',
].join('\n')

export async function mock<T>(op: string, data: Record<string, unknown>): Promise<T> {
  await new Promise(r => setTimeout(r, 150))
  const list = initial === 'none' ? [] : models
  switch (op) {
    case 'catalog': case 'roots':
      return { models: list, roots: (data.roots as string[]) || ['D:\\models', 'C:\\Users\\me\\.lmstudio\\models'], scanned_at: '2026-09-26T13:00:00' } as T
    case 'status': return status() as T
    case 'profile':
      return { profile: profileOf(String(data.id)), companions: { draft: DRAFT, draft_head: true, mmproj: true } } as T
    case 'save':
      saved[String(data.id)] = { ...(data.profile as object) }
      return { profile: profileOf(String(data.id)), restart_required: !!running } as T
    case 'start':
      // as the manager: one model at a time, so a switch is a stop and then a start
      if (running) throw JSON.stringify({ ok: false, error: 'Unload the current model before loading another', code: 'already_loaded' })
      running = String(data.id); runningProfile = profileOf(running); state = 'loading'; listening = { ...network }
      setTimeout(() => { state = 'ready' }, 3000)
      return { status: 'loading' } as T
    case 'stop':
      running = ''; state = 'stopped'
      return { status: 'stopped' } as T
    case 'network': return networkView() as T
    case 'save_network': {
      const next = { ...network, ...(data as Partial<typeof network>) }
      if (!Number.isInteger(next.port) || next.port < 1024 || next.port > 65535)
        throw JSON.stringify({ ok: false, error: 'port must be between 1024 and 65535', code: 'out_of_range', params: { field: 'port', low: 1024, high: 65535 } })
      network = next
      return { network: networkView(), restart_required: !!running } as T
    }
    case 'slots':
      // four slots: two generating, one processing a prompt, one idle (numbers move so the page visibly updates)
      if (!running) return { slots: [] } as T
      tick++
      return { slots: [
        { id: 0, active: true, task: 12, context: 51008 + tick * 40, prompt_processed: 15, prompt_cached: 50993, generated: 1850 + tick * 40 },
        { id: 1, active: true, task: 13, context: 38961 + tick * 35, prompt_processed: 24, prompt_cached: 38937, generated: 790 + tick * 35 },
        { id: 2, active: true, task: 14 + Math.floor(tick / 12), context: 4617, prompt_processed: 350 * (tick % 12), prompt_cached: 0, generated: 0 },
        { id: 3, active: false, task: 9, context: 98768, prompt_processed: 12, prompt_cached: 98756, generated: 2199 },
      ], waiting: tick % 20 < 10 ? 2 : 0 } as T
    case 'logs':
      return { text: Number(data.offset) ? '' : LOG, offset: LOG.length, file: 'C:\\Users\\me\\AppData\\Roaming\\dev.rulith.strixllama\\logs\\server.log', reset: false } as T
    default: return {} as T
  }
}
