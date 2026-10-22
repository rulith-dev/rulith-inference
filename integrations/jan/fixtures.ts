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
const baseProfile = { thinking: 'medium', context: 262144, gpu_layers: 999, threads: 16, batch: 8192, ubatch: 8192, mtp: true, draft: DRAFT, draft_max: 3, draft_min: 0.6, ngram_spec: false, kv: 'f16', flash_attention: 'on', qsa: true, trunk_decode_q6k: false, parallel: 8, kv_pool: 0, vision: false, mmproj: '', prompt_cache_disk: true, prompt_cache_disk_mib: 65536, host_experts: true, image_min_tokens: 1024 }
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

// Model downloads: one catalog model; a job moves ~3 GB a poll while it runs. ?dl=running|paused|failed|done starts
// with such a job.
const DL_FILES = [
  { name: 'Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf', role: 'model', required: true, size: 10946624 },
  { name: 'Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf', role: 'model', required: true, size: 49835229856 },
  { name: 'Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf', role: 'model', required: true, size: 43836407744 },
  { name: 'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf', role: 'draft', required: false, size: 1907151936 },
  { name: 'mmproj-F16.gguf', role: 'projection', required: false, size: 904004000 },
  { name: 'mtp-Qwen3.8-Flash-Next-head-lr512.gguf', role: 'head', required: false, size: 148652224 },
]
const DL_FOLDER = 'D:\\Rulith Models\\unsloth\\Qwen3.8-Flash-Next-GGUF'
const dlMode = new URLSearchParams(location.search).get('dl')
type MockJob = { id: string; title: string; files: { name: string; size: number; bytes: number }[]; status: string; alive: boolean; error?: unknown }
let dlJob: MockJob | null = dlMode ? {
  id: '20261019-140000-abc123', title: 'Qwen3.8 Flash Next (UD-IQ4_XS)', alive: dlMode === 'running',
  status: dlMode === 'paused' ? 'stopped' : dlMode === 'running' ? 'running' : dlMode,
  files: DL_FILES.map((f, i) => ({ name: f.name, size: f.size, bytes: dlMode === 'done' ? f.size : i === 0 ? f.size : i === 1 ? 31.4e9 : 0 })),
  ...(dlMode === 'failed' ? { error: { code: 'sources_failed', params: { file: DL_FILES[1].name } } } : {}),
} : null
const jobView = (j: MockJob) => {
  const bytes = j.files.reduce((n, f) => n + f.bytes, 0), total = j.files.reduce((n, f) => n + f.size, 0)
  return {
    id: j.id, kind: 'catalog', title: j.title, model: 'qwen3.8-flash-next-ud-iq4_xs', folder: DL_FOLDER, source: 'hfm',
    created_at: '2026-10-19T14:00:00', status: j.status, alive: j.alive, error: j.error ?? null, bytes, total,
    speed: j.alive ? 46.2e6 : 0, eta: j.alive ? (total - bytes) / 46.2e6 : null,
    files: j.files.map(f => ({ name: f.name, size: f.size, bytes: f.bytes, status: f.bytes >= f.size ? 'done' : j.alive && f.bytes > 0 ? 'downloading' : f.bytes > 0 ? 'downloading' : 'pending', verified: f.bytes >= f.size, source: 0 })),
  }
}
const dlStatus = () => {
  if (dlJob?.alive) {
    let step = 3e9
    for (const f of dlJob.files) { const add = Math.min(step, f.size - f.bytes); f.bytes += add; step -= add }
    if (dlJob.files.every(f => f.bytes >= f.size)) { dlJob.alive = false; dlJob.status = 'done' }
  }
  return { jobs: dlJob ? [jobView(dlJob)] : [], active: dlJob?.alive ? dlJob.id : null }
}
// a recommended EXL3 repository: the list has its 4.05 bpw files, the repository a 3.05 bpw folder as well (RULITH_REPO)
const EXL3_FILES = [
  ...[10946624, 2706906080, 39040187584, 23589139232, 23592085568, 15354420160].map((size, i) =>
    ({ name: `Qwen3.8-Flash-Next-EXL3-4.05bpw-0000${i + 1}-of-00006.gguf`, role: 'model', required: true, size })),
  { name: 'mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf', role: 'draft', required: false, size: 2566331680 },
  { name: 'mmproj-F16.gguf', role: 'projection', required: false, size: 904004000 },
]
const EXL3_FOLDER = 'D:\\Rulith Models\\Rulith\\Qwen3.8-Flash-Next-Rulith-EXL3-GGUF'
const dlCatalog = () => ({
  models: [{ id: 'qwen3.8-flash-next-exl3-4.05bpw', name: 'Qwen3.8 Flash Next', variant: 'EXL3 4.05 bpw', owner: 'Rulith', repo: 'Qwen3.8-Flash-Next-Rulith-EXL3-GGUF',
    license: 'https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF', capabilities: ['images', 'tools', 'mtp'], fits: true, commit: '1e13e0220f7f860a3030fcb80cfc5bcbac5fa54f',
    files: EXL3_FILES.map(f => ({ ...f, present: EXL3_FOLDER + '\\' + f.name })), missing: 0 },
    { id: 'qwen3.8-flash-next-ud-iq4_xs', name: 'Qwen3.8 Flash Next', variant: 'UD-IQ4_XS', owner: 'unsloth', repo: 'Qwen3.8-Flash-Next-GGUF',
    license: 'https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF', capabilities: ['images', 'tools', 'mtp'], fits: true, commit: '38bb39ee97821de2c9009abb7e93950eec396e66',
    files: DL_FILES.map(f => ({ ...f, present: initial === 'none' && dlJob?.status !== 'done' ? null : DL_FOLDER + '\\' + f.name })),
    missing: initial === 'none' ? DL_FILES.reduce((n, f) => n + f.size, 0) : 0 }],
  sources: { hf: 'Hugging Face', hfm: 'hf-mirror.com', gh: 'GitHub', ghp: 'gh-proxy.com' },
  drives: [{ drive: 'C:\\', free: 315e9, total: 2e12 }, { drive: 'D:\\', free: 1.2e12, total: 4e12 }],
  default_root: 'D:\\Rulith Models', ...dlStatus(),
})
const REPO = {
  source: 'hfm', owner: 'example', repo: 'Example-30B-A3B-GGUF', revision: 'main', commit: '4f2c8e1d9b7a6c5e3f1a2b4c6d8e0f1a2b3c4d5e',
  folder: 'D:\\Rulith Models\\example\\Example-30B-A3B-GGUF',
  groups: [
    { key: 'Example-30B-A3B-Q4_K_M', role: 'model', name: 'Example-30B-A3B-Q4_K_M', quant: 'Q4_K_M', size: 18.6e9, complete: true, shards: 1, fits: true, tuned: true, files: [{ name: 'Example-30B-A3B-Q4_K_M.gguf', path: 'Example-30B-A3B-Q4_K_M.gguf', size: 18.6e9, sha256: 'ab'.repeat(32), present: null }] },
    { key: 'Q8_0/Example-30B-A3B-Q8_0', role: 'model', name: 'Example-30B-A3B-Q8_0', quant: 'Q8_0', size: 32.5e9, complete: true, shards: 2, fits: true, files: [{ name: 'Example-30B-A3B-Q8_0-00001-of-00002.gguf', path: 'Q8_0/Example-30B-A3B-Q8_0-00001-of-00002.gguf', size: 16e9, sha256: 'cd'.repeat(32), present: null }, { name: 'Example-30B-A3B-Q8_0-00002-of-00002.gguf', path: 'Q8_0/Example-30B-A3B-Q8_0-00002-of-00002.gguf', size: 16.5e9, sha256: 'ef'.repeat(32), present: null }] },
    { key: 'BF16/Example-30B-A3B-BF16', role: 'model', name: 'Example-30B-A3B-BF16', quant: 'BF16', size: 61e9, complete: false, shards: 3, fits: true, files: [] },
    { key: 'mmproj-F16', role: 'projection', name: 'mmproj-F16', quant: 'F16', size: 0.9e9, complete: true, shards: 1, fits: null, files: [{ name: 'mmproj-F16.gguf', path: 'mmproj-F16.gguf', size: 0.9e9, sha256: null, present: null }] },
  ],
}

const shard305 = [10946624, 2062113760, 32640156864, 23927378976, 23707918976]
const RULITH_REPO = {
  source: 'hfm', owner: 'Rulith', repo: 'Qwen3.8-Flash-Next-Rulith-EXL3-GGUF', revision: 'main', commit: '4e3ec2101aa2056553f38168ab559c4b1707659f',
  folder: EXL3_FOLDER,
  groups: [
    { key: 'EXL3-3.05bpw/Qwen3.8-Flash-Next-EXL3-3.05bpw', role: 'model', name: 'Qwen3.8-Flash-Next-EXL3-3.05bpw', quant: 'EXL3 3.05 bpw',
      size: shard305.reduce((n, x) => n + x, 0), complete: true, shards: 5, fits: true, tuned: true,
      files: shard305.map((size, i) => ({ name: `Qwen3.8-Flash-Next-EXL3-3.05bpw-0000${i + 1}-of-00005.gguf`, path: `EXL3-3.05bpw/Qwen3.8-Flash-Next-EXL3-3.05bpw-0000${i + 1}-of-00005.gguf`, size, sha256: 'aa'.repeat(32), present: null })) },
    { key: 'EXL3-4.05bpw/Qwen3.8-Flash-Next-EXL3-4.05bpw', role: 'model', name: 'Qwen3.8-Flash-Next-EXL3-4.05bpw', quant: 'EXL3 4.05 bpw',
      size: EXL3_FILES.slice(0, 6).reduce((n, f) => n + f.size, 0), complete: true, shards: 6, fits: true, tuned: true,
      files: EXL3_FILES.slice(0, 6).map(f => ({ name: f.name, path: 'EXL3-4.05bpw/' + f.name, size: f.size, sha256: 'bb'.repeat(32), present: EXL3_FOLDER + '\\' + f.name })) },
    { key: 'mmproj-F16', role: 'projection', name: 'mmproj-F16', quant: 'F16', size: 904004000, complete: true, shards: 1, fits: null,
      files: [{ name: 'mmproj-F16.gguf', path: 'mmproj-F16.gguf', size: 904004000, sha256: 'cc'.repeat(32), present: EXL3_FOLDER + '\\mmproj-F16.gguf' }] },
    { key: 'mtp-Qwen3.8-Flash-Next-lr512-ownhead', role: 'draft', name: 'mtp-Qwen3.8-Flash-Next-lr512-ownhead', quant: '', size: 2566331680, complete: true, shards: 1, fits: null,
      files: [{ name: 'mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf', path: 'mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf', size: 2566331680, sha256: 'dd'.repeat(32), present: EXL3_FOLDER + '\\mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf' }] },
  ],
}

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
    case 'dl_catalog': return dlCatalog() as T
    case 'dl_status': return dlStatus() as T
    case 'dl_search':
      return { source: data.source, results: String(data.query).toLowerCase().includes('zzz') ? [] : [
        { id: 'example/Example-30B-A3B-GGUF', owner: 'example', repo: 'Example-30B-A3B-GGUF', downloads: 1234567, likes: 890, updated: '2026-10-01T00:00:00Z', task: 'text-generation', architecture: 'qwen3moe' },
        { id: 'unsloth/Qwen3.8-Flash-Next-GGUF', owner: 'unsloth', repo: 'Qwen3.8-Flash-Next-GGUF', downloads: 1100582, likes: 1243, updated: '2026-09-26T00:00:00Z', task: 'image-text-to-text', architecture: 'qwen4exp' },
      ] } as T
    case 'dl_info':
      return { source: data.source, info: { downloads: 1100582, likes: 1243, updated: '2026-10-06T05:31:22Z', task: 'image-text-to-text', license: 'qwen-community-1.0', base_model: 'Qwen/Qwen3.8-Flash-Next', architecture: 'qwen4exp', params: 176943899520, context: 262144 } } as T
    case 'dl_readme':
      return { text: '# ' + String(data.repo) + '\n\nA model card with **bold** text, a [link](https://huggingface.co) and a table:\n\n| quant | size |\n|---|---|\n| Q4_K_M | 18.6 GB |\n' } as T
    case 'dl_repo':
      if (!/\//.test(String(data.url))) throw JSON.stringify({ ok: false, error: 'Not a Hugging Face or hf-mirror repository link', code: 'dl_repo_invalid' })
      return (/Rulith\//.test(String(data.url)) ? RULITH_REPO : REPO) as T
    case 'dl_start':
      if (dlJob?.alive) throw JSON.stringify({ ok: false, error: 'A download is already running', code: 'dl_running' })
      dlJob = { id: '20261019-141500-def456', title: data.kind === 'repo' ? 'example/Example-30B-A3B-GGUF: Example-30B-A3B-Q4_K_M' : 'Qwen3.8 Flash Next (UD-IQ4_XS)', status: 'running', alive: true,
        files: (data.kind === 'repo' ? [{ name: 'Example-30B-A3B-Q4_K_M.gguf', size: 18.6e9 }] : DL_FILES).map(f => ({ name: f.name, size: f.size, bytes: 0 })) }
      return jobView(dlJob) as T
    case 'dl_pause': if (dlJob) { dlJob.alive = false; dlJob.status = 'stopped' } return (dlJob && jobView(dlJob)) as T
    case 'dl_resume': if (dlJob) { dlJob.alive = true; dlJob.status = 'running'; dlJob.error = undefined } return (dlJob && jobView(dlJob)) as T
    case 'dl_cancel': dlJob = null; return dlStatus() as T
    default: return {} as T
  }
}
