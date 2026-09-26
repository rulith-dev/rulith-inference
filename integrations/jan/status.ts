import { create } from 'zustand'
import { invoke } from '@tauri-apps/api/core'

// One IPC round trip to tools/manager.py. The helper reads a JSON request on stdin and answers on
// stdout; there is no network port and no shell in between. In the browser preview (`yarn dev:web`,
// no Tauri) the answers come from fixtures.ts instead, which production builds never include.
export const request = async <T,>(op: string, data: object = {}): Promise<T> => {
  if (import.meta.env.DEV && typeof (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ === 'undefined')
    return (await import('./fixtures')).mock<T>(op, data as Record<string, unknown>)
  return invoke<T>('strixllama_request', { request: { op, data } })
}

// The provider Jan's chat side talks to. There is exactly one in this build: the local server the
// manager starts, on the endpoint below. StrixLlamaSync registers it on first run.
export const PROVIDER = 'strixllama'
export const ENDPOINT = 'http://127.0.0.1:8080/v1'

export type Profile = { thinking: string; context: number; gpu_layers: number; threads: number; batch: number; ubatch: number; mtp: boolean; draft: string; draft_max: number; draft_min: number; ngram_spec: boolean; kv: string; flash_attention: string; qsa: boolean; trunk_decode_q6k: boolean; parallel: number; kv_pool: number; vision: boolean; mmproj: string; prompt_cache_disk: boolean; prompt_cache_disk_mib: number }
export type Model = { id: string; path: string; name: string; filename: string; architecture?: string; quant?: string; size: number; shards?: number; missing?: string[]; role: string; error?: string; context?: number }
// merged / merge_errors: set by a rescan, when a downloaded draft head was combined with its shared draft
export type Catalog = { models: Model[]; roots: string[]; scanned_at: string; merged?: string[]; merge_errors?: Record<string, string> }
// what the companion switches have to work with: the draft the profile would use (null when none is
// on disk), whether it is a merged *-head-* one, and whether the projector sits beside the model
export type Companions = { draft: string | null; draft_head: boolean; mmproj: boolean }
export type LogChunk = { text: string; offset: number; file: string; reset: boolean }
export type Status = {
  status: string; endpoint: string; runtime: string; runtime_available?: boolean; runtime_env?: Record<string, string>
  // the ROCm release the runtime was built and bundled with, and the GPU target it was compiled for
  runtime_info?: { rocm?: string; gfx?: string }
  identity?: { pid: number }; model_path?: string; model_name?: string; log?: string; command?: string; adopted?: boolean; profile?: Profile
  served_models?: { id: string }[]
  // set by the manager: the dedicated carve it saw, and why the last load ended if it ended badly
  dedicated_vram?: number | null; failure?: string; failure_code?: string
  // a ready server: Windows' commit limit (RAM + page file) and what is left of it, and whether that is too little
  commit_limit?: number; commit_available?: number; commit_low?: boolean
  // this server's log: answers stopped because an allocation failed (Windows commit ran out), and saves to the
  // disk tier skipped for the same reason, with the local time of the last of each
  memory_events?: { answers_stopped: number; saves_skipped: number; last_stop?: string; last_skip?: string }
}

// What the pages show: a process that answers health checks is ready, one that does not yet is
// loading, and without a process the last exit decides between stopped and failed.
export type ServerState = 'ready' | 'loading' | 'stopped' | 'failed'
export const serverState = (s?: Status): ServerState =>
  s?.identity ? (s.status === 'ready' ? 'ready' : 'loading') : s?.failure ? 'failed' : 'stopped'

export const gb = (n: number) => `${(n / 1e9).toFixed(1)} GB`
// "13:33" from the manager's local ISO time
export const clock = (iso?: string) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '')
// "Qwen3.8 Flash Next · IQ4_XS": the name alone does not tell two quantisations of one model apart
export const modelLabel = (m?: Model) => (m ? [m.name, m.quant].filter(Boolean).join(' · ') : '')
export const fileName = (path: string) => path.split(/[\\/]/).pop() || path
// "10.2.0a20260925" -> "ROCm 10.2"; the full string goes in a tooltip
export const rocmLabel = (s?: Status) => {
  const v = s?.runtime_info?.rocm
  return v ? `ROCm ${v.replace(/^(\d+\.\d+).*$/, '$1')}` : 'ROCm'
}

// A failed request arrives as the manager's whole error record, {error, code, params}, when it
// has a code (strixllama.rs passes it through as JSON), so the page can render `errors.<code>`
// from its own locale with the parameters filled in; the English text is the fallback. Anything
// else - the Rust side, the IPC - is a plain string.
export const describeError = (e: unknown, tr: (key: string, options?: Record<string, unknown>) => string) => {
  const text = String(e)
  if (text.startsWith('{')) {
    try {
      const parsed = JSON.parse(text)
      if (parsed && typeof parsed.code === 'string')
        return tr(`errors.${parsed.code}`, { defaultValue: String(parsed.error || parsed.code), ...(parsed.params || {}) })
    } catch { /* not the manager's record */ }
  }
  return text
}

type State = { status?: Status; error: string; refresh: () => Promise<Status | undefined> }

// The manager's status, polled once for the whole app by StrixLlamaSync and read by every view, the
// sidebar dot and the chat header, so they share one poll instead of each running its own.
export const useStrixLlamaStatus = create<State>((set) => ({
  status: undefined,
  error: '',
  refresh: async () => {
    try {
      const status = await request<Status>('status')
      set({ status, error: '' })
      return status
    } catch (e) {
      set({ error: String(e) })
      return undefined
    }
  },
}))

// Load the model when the app starts (Configuration > Startup). A view setting, so it lives with the
// web app's other preferences rather than in the manager's profile. Off unless turned on: loading on its own
// at startup took whichever model was loaded last, with its saved settings, and a user who meant to change
// them first had to wait out a load and unload it again (0.2.5's first builds had it on by default).
const AUTOLOAD_KEY = 'strixllama-autoload'
export const LAST_MODEL_KEY = 'strixllama-last-model'
export const autoloadEnabled = () => localStorage.getItem(AUTOLOAD_KEY) === '1'
export const setAutoloadEnabled = (on: boolean) => localStorage.setItem(AUTOLOAD_KEY, on ? '1' : '0')

// Before an update replaces the runtime: unload the model the way the Unload button does, so the server
// writes its conversations to the disk tier and lets go of runtime\bin\hip. Nothing loaded, nothing to do.
export const stopModelServer = async () => {
  try {
    const s = await request<Status>('status')
    if (s.identity) await request('stop')
  } catch { /* no manager to ask (the browser preview): the setup's own hook stops a server */ }
}

// One of the server's slots, from the manager's `slots` op: whether it is working, the tokens of the conversation it
// holds, how far the current prompt was processed, and the tokens generated for the current (or last) request
export type SlotInfo = { id: number; active: boolean; task?: number | null; context: number; prompt_processed: number; prompt_cached: number; generated: number }
