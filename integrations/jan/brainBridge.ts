import { Channel, invoke } from '@tauri-apps/api/core'
import { create } from 'zustand'

export const brainOrigin = 'https://brain-ui.rulith.invalid'
export const brainEntry = brainOrigin + '/index.html'
export const requiredCaps = ['ui.frame.v1', 'manager.python.v1', 'memory.stream.v1', 'ui.store.v1']
export const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const generationId = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i

export type Mode = {
  state: string
  selected: 'inference' | 'brain'
  active: 'inference' | 'brain' | 'none'
  target: 'inference' | 'brain' | null
  transition_id: string | null
  owner: { host: string; pid?: number; exe?: string } | null
  error: { code: string; params: object } | null
  can_retry: boolean
  can_restore: boolean
  previous_model_id: string | null
}
export type Runtime = {
  installed: { verified: boolean; capabilities: string[]; reason: string | null; version: string | null }
  pending: string | null
  job: { alive: boolean; bytes: number; total: number; status: string; error: { code: string } | null } | null
  available?: { newer: boolean; version: string }
}
export type Mount = { url: string; generation: string }
export const validMount = (m: Mount) => m.url === brainEntry && generationId.test(m.generation)
export const useMode = create<{ mode: Mode | null; error: string }>(() => ({ mode: null, error: '' }))
export const inferenceAllowed = (m: Mode | null) => !!m && m.selected === 'inference'
  && m.target === null && (m.state === 'idle' || m.state === 'inference_ready')
  && m.active !== 'brain' && (!m.owner || m.owner.host === 'inference')

let revision = 0
export async function readMode() {
  const ticket = ++revision
  try {
    const mode = await invoke<Mode>('brain_host_mode', { op: 'mode_status', data: {} })
    if (ticket === revision) useMode.setState({ mode, error: '' })
    return mode
  } catch (error) {
    if (ticket === revision) useMode.setState({ mode: null, error: errorCode(error) })
    throw error
  }
}
export async function changeMode(op: 'mode_switch_inference' | 'mode_switch_brain' | 'mode_retry', data: object) {
  ++revision
  const mode = await invoke<Mode>('brain_host_mode', { op, data: { ...data, request_id: crypto.randomUUID() } })
  ++revision
  useMode.setState({ mode, error: '' })
  return mode
}

type Obj = Record<string, unknown>
type Reason = 'navigate' | 'switch' | 'quit' | 'update'
type Closing = Reason | 'protocol'
export type Wire = Obj & { v: 1; type: string; id: string }
type Send = (request: Wire, reply: (event: Wire) => void) => Promise<void>
const object = (x: unknown): x is Obj => typeof x === 'object' && x !== null
  && Object.prototype.toString.call(x) === '[object Object]'
const uint = (x: unknown): x is number => Number.isSafeInteger(x) && Number(x) >= 0
const exact = (x: Obj, keys: string) => Object.keys(x).every(k => !!k && keys.split(' ').includes(k))
const utf8 = new TextEncoder()
const size = (x: unknown) => utf8.encode(JSON.stringify(x)).length
const MiB = 1024 * 1024
const reasons = ['navigate', 'switch', 'quit', 'update']
export const errorCode = (e: unknown) => object(e) && typeof e.code === 'string' && /^[a-z][a-z0-9_]{0,63}$/.test(e.code)
  ? e.code : 'host_unavailable'

// Each entry lists the fields the child may send. Native checks the operation's
// value types, path grants, ownership and projected result again before dispatch.
const fields: Record<string, string> = {
  brain_memory: 'method path body', brain_licence: 'refresh', brain_trial: '',
  brain_activate: 'key', brain_transfer: 'key', brain_upgrade: 'key to', brain_upgrade_finish: 'key',
  brain_create: 'name parent select', brain_runtime: '', brain_runtime_check: '',
  brain_runtime_install: '', brain_runtime_cancel: '', brain_app: 'refresh', brain_app_select: 'dir',
  brain_app_readonly: 'dir read_only', brain_room: 'dir', brain_app_start: '',
  brain_self: 'dir', brain_self_set: 'dir text', brain_capacity: 'dir', brain_capacity_set: 'dir tokens confirm_forget',
  brain_folders: 'sync', brain_folder_add: 'path', brain_folder_remove: 'id', brain_folder_set: 'id watch retry',
  brain_folder_scan: 'id', brain_web_settings: '', brain_web_set: 'provider searxng_url',
  brain_web_search: 'query lang', brain_web_fetch: 'url lang',
  status: '', logs: 'offset', stop: '', roots: 'roots', dl_start: 'model include source',
  dl_status: '', dl_pause: 'id', dl_resume: 'id', dl_cancel: 'id delete_parts',
  memory: 'method path body', memory_stream: 'method path body', open_link: 'url',
  pick_folder: 'title start', open_folder: 'path', app_info: '', set_theme: 'theme',
  ui_store_read: 'name', ui_store_write: 'name expected_rev value',
}
function jsonValue(x: unknown, ancestors = new Set<object>()): boolean {
  if (x === null || typeof x === 'string' || typeof x === 'boolean') return true
  if (typeof x === 'number') return Number.isFinite(x)
  if ((!object(x) && !Array.isArray(x)) || ancestors.has(x)) return false
  ancestors.add(x)
  const valid = Object.values(x).every(value => jsonValue(value, ancestors))
  ancestors.delete(x)
  return valid
}
function memoryPath(method: unknown, path: unknown) {
  if (typeof path !== 'string') return false
  if (method === 'GET' && path === '/slots') return true
  const [name, query, extra] = path.split('?')
  if (extra !== undefined || !name.startsWith('/v1/memory/')) return false
  if (query !== undefined && !/^[a-zA-Z0-9._~-]+(?:=[a-zA-Z0-9._~-]*)?(?:&[a-zA-Z0-9._~-]+(?:=[a-zA-Z0-9._~-]*)?)*$/.test(query)) return false
  return name.slice('/v1/memory/'.length).split('/').every(segment => {
    if (!/^(?:[A-Za-z0-9._~-]|%[A-Fa-f0-9]{2})+$/.test(segment)) return false
    try {
      const value = decodeURIComponent(segment)
      return value !== '.' && value !== '..' && !/[%/\\?#\u0000-\u001f\u007f]/.test(value)
    } catch { return false }
  })
}

const required: Record<string, string[]> = {
  brain_activate: ['key'], brain_transfer: ['key'], brain_upgrade: ['key','to'], brain_upgrade_finish: ['key'],
  brain_create: ['name'], brain_app_select: ['dir'], brain_folder_add: ['path'],
  brain_folder_remove: ['id'], brain_folder_set: ['id'], brain_web_search: ['query'], brain_web_fetch: ['url'],
  dl_start: ['model','include'], dl_pause: ['id'], dl_resume: ['id'], dl_cancel: ['id'], roots: ['roots'],
  open_link: ['url'], open_folder: ['path'], set_theme: ['theme'],
}
function managerShape(op: string, body: Obj) {
  if (required[op]?.some(key => !(key in body))) return false
  for (const [key, value] of Object.entries(body)) {
    if (['refresh','select','read_only','sync','watch','retry','delete_parts','confirm_forget'].includes(key)) {
      if (typeof value !== 'boolean') return false
    } else if (key === 'tokens') {
      if (value !== null && (!uint(value) || value === 0 || value % 1000000 !== 0)) return false
    } else if (key === 'offset') {
      if (!uint(value)) return false
    } else if (key === 'roots') {
      if (!Array.isArray(value) || !value.length || value.length > 12 || !value.every(x => typeof x === 'string')) return false
    } else if (key === 'include') {
      if (!Array.isArray(value) || value.length !== 1 || value[0] !== 'draft') return false
    } else if (value === null && ['title','start','text'].includes(key)) continue
    else if (typeof value !== 'string') return false
  }
  if (op === 'set_theme') return ['light','dark','system'].includes(String(body.theme))
  if (op === 'dl_start') return body.model === 'qwen3.8-flash-next-exl3-4.05bpw'
  if (op === 'brain_upgrade') return ['10M','50M','200M','unlimited'].includes(String(body.to))
  if (op === 'brain_create') return !!String(body.name).trim() && String(body.name).trim().length <= 80
  if (op === 'open_link') {
    const raw = String(body.url)
    if (!/^https:\/\//i.test(raw) || /[\s\\\u0000-\u001f\u007f]/.test(raw)) return false
    const rawPath = raw.split(/[?#]/)[0]
    if (/%(?:2f|5c|25)/i.test(rawPath) || /(?:^|\/)(?:\.|%2e){1,2}(?:\/|$)/i.test(rawPath)) return false
    try {
      const u = new URL(raw)
      if (u.username || u.password || u.port) return false
      return u.hostname === 'rulith.ai' || (u.hostname === 'github.com' && u.pathname.startsWith('/rulith-dev/'))
        || (u.hostname === 'huggingface.co' && u.pathname.startsWith('/Rulith/'))
        || (['creem.io','www.creem.io'].includes(u.hostname) && u.pathname.startsWith('/checkout/'))
    } catch { return false }
  }
  return true
}
function requestError(msg: Wire) {
  const { op, body } = msg
  if (typeof op !== 'string' || !object(body) || !exact(msg, 'v type id op body')) return 'request_invalid'
  if (size({ v: msg.v, type: msg.type, id: msg.id, op, method: body.method, path: body.path }) > 8192) return 'request_too_large'
  if (!Object.prototype.hasOwnProperty.call(fields, op)) return 'op_denied'
  if (!exact(body, fields[op]) || !jsonValue(body)) return 'request_invalid'
  if (op === 'memory' || op === 'memory_stream' || op === 'brain_memory') {
    if (!memoryPath(body.method, body.path) || !['GET', 'POST', 'DELETE'].includes(String(body.method))
      || ('body' in body && !(op === 'brain_memory' && body.body === null)
        && (body.method !== 'POST' || !object(body.body)))) return 'request_invalid'
    if (op === 'brain_memory') {
      const path = decodeURIComponent(String(body.path).split('?')[0])
      if (path === '/slots' || /^\/v1\/memory\/(?:input|chat)(?:\/|$)/.test(path)) return 'op_denied'
      if (size({ op, data: body }) > 65536) return 'request_too_large'
    } else if (size(body.body ?? {}) > 64 * MiB) return 'request_too_large'
  } else if (op.startsWith('ui_store_')) {
    if (!['settings', 'conversation', 'study', 'learn', 'activity'].includes(String(body.name))) return 'request_invalid'
    if (op === 'ui_store_write') {
      if (!uint(body.expected_rev) || !('value' in body)) return 'request_invalid'
      if (size(body.value) > 16 * MiB) return 'request_too_large'
    }
  } else {
    if (size({ op, data: body }) > 65536) return 'request_too_large'
    if (!managerShape(op, body)) return 'request_invalid'
    if (op === 'open_link' && !navigator.userActivation?.isActive) return 'op_denied'
  }
}

type Pending = {
  op: string; stream: boolean; head: boolean; next: number; ack: number
  acknowledging?: boolean; sizes: number[]; outstanding: number; cancelled: boolean; reply: (msg: Wire) => void
}
export function nativeSender(generation: string): Send {
  return (request, reply) => {
    // The stream keeps its original response channel. Native looks up controls
    // by generation and id; dropping a control channel cannot end that stream.
    const onEvent = new Channel<Wire>(request.type === 'request' ? reply : undefined)
    return invoke('brain_host_call', { generation, request, onEvent })
  }
}

// One instance per native mapping. The only document allowed to receive its
// port is the verified entry page in this particular iframe.
export class BrainBridge {
  private port?: MessagePort
  private loaded = false
  private hello = false
  private dead = false
  private seen = new Set<string>()
  private pending = new Map<string, Pending>()
  private flushing = new Map<string, (ok: boolean) => void>()

  constructor(private frame: HTMLIFrameElement, private caps: string[], private send: Send,
    private revoked: (reason: Closing) => void) {}
  get connected() { return !!this.port && !this.dead }

  helloFrom = (e: MessageEvent) => {
    if (this.dead || this.port || e.origin !== brainOrigin || e.source !== this.frame.contentWindow
      || e.ports.length || !object(e.data) || !exact(e.data, 'v type') || e.data.v !== 1 || e.data.type !== 'hello') return
    this.hello = true
    this.connect()
  }
  documentLoaded = () => {
    if (this.loaded) return this.close('navigate')
    this.loaded = true
    this.connect()
  }
  private connect() {
    if (this.dead || this.port || !this.loaded || !this.hello || !this.frame.contentWindow) return
    const { port1, port2 } = new MessageChannel()
    this.port = port1
    port1.onmessage = this.receive
    port1.onmessageerror = () => this.close('protocol')
    this.frame.contentWindow.postMessage({ v: 1, type: 'connect', host: 'inference',
      capabilities: requiredCaps.filter(cap => this.caps.includes(cap)) }, brainOrigin, [port2])
  }
  private post(msg: Obj) { this.port?.postMessage({ ...msg, v: 1 }) }
  private fail(id: string, code: string, params: Obj = {}) {
    this.post({ type: 'error', id, code, message: code, params })
  }
  private receive = async ({ data: msg, ports }: MessageEvent) => {
    if (this.dead) return
    if (!object(msg) || ports.length || msg.v !== 1) return this.close('protocol')
    if (msg.type === 'close' && exact(msg, 'v type reason') && [...reasons, 'protocol'].includes(String(msg.reason)))
      return this.close(msg.reason as Closing)
    if (typeof msg.id !== 'string' || !uuid.test(msg.id)) return this.close('protocol')
    const id = msg.id
    if (msg.type === 'flushed') {
      if (!exact(msg, 'v type id ok code') || typeof msg.ok !== 'boolean') return this.close('protocol')
      this.flushing.get(id)?.(msg.ok)
      return
    }
    const p = this.pending.get(id)
    if (msg.type === 'cancel' || msg.type === 'ack') {
      if (!exact(msg, msg.type === 'cancel' ? 'v type id' : 'v type id next_seq')) return this.close('protocol')
      if (!p?.stream || p.cancelled) return
      if (msg.type === 'cancel') p.cancelled = true
      else {
        if (!uint(msg.next_seq) || msg.next_seq < p.ack || msg.next_seq > p.next) return this.close('protocol')
        for (const n of p.sizes.splice(0, msg.next_seq - p.ack)) p.outstanding -= n
        p.ack = msg.next_seq
        // Native calls can run out of order. Keep one cumulative acknowledgement in flight per stream.
        if (p.acknowledging) return
        p.acknowledging = true
        try {
          let sent: number
          do {
            sent = p.ack
            await this.send({ v: 1, type: 'ack', id, next_seq: sent }, p.reply)
          } while (this.pending.get(id) === p && !p.cancelled && p.ack > sent)
        } catch {
          if (this.pending.get(id) === p && !p.cancelled) this.close('protocol')
        } finally { p.acknowledging = false }
        return
      }
      void this.send(msg as Wire, p.reply).catch(() => this.close('protocol'))
      return
    }
    if (msg.type !== 'request' || this.seen.has(id)) return this.close('protocol')
    this.seen.add(id)
    let error: string | undefined
    try { error = requestError(msg as Wire) } catch { error = 'request_invalid' }
    if (error) return this.fail(id, error)
    const job: Pending = { op: String(msg.op), stream: msg.op === 'memory_stream', head: false, next: 0, ack: 0,
      outstanding: 0, sizes: [], cancelled: false, reply: event => this.answer(id, job, event) }
    this.pending.set(id, job)
    void this.send(msg as Wire, job.reply).catch(e => {
      if (this.pending.get(id) !== job) return
      this.pending.delete(id)
      this.fail(id, errorCode(e), object(e) && object(e.params) ? e.params : {})
    })
  }
  private answer(id: string, job: Pending, msg: Wire) {
    if (this.dead || this.pending.get(id) !== job) return
    if (!object(msg) || msg.v !== 1 || msg.id !== id) return this.close('protocol')
    if (msg.type === 'error' && typeof msg.code === 'string' && typeof msg.message === 'string' && object(msg.params)) {
      this.pending.delete(id)
    } else if (!job.stream && msg.type === 'result' && 'body' in msg) {
      this.pending.delete(id)
      // Native checks memory response bytes before parsing or wrapping them.
      if (job.op !== 'memory' && size(msg.body) > 64 * MiB) return this.fail(id, 'response_too_large')
    } else if (job.stream && job.cancelled) {
      if (msg.type === 'end') {
        this.pending.delete(id)
        this.fail(id, 'cancelled')
      }
      return
    } else if (job.stream && msg.type === 'head' && !job.head && uint(msg.status) && object(msg.headers)) job.head = true
    else if (job.head && msg.type === 'chunk' && msg.seq === job.next && typeof msg.text === 'string') {
      const n = utf8.encode(msg.text).length
      if (n > 65536) return this.close('protocol')
      if (job.outstanding + n > MiB || job.sizes.length >= MiB) {
        this.pending.delete(id)
        void this.send({ v: 1, type: 'cancel', id }, job.reply).catch(() => this.close('protocol'))
        return this.fail(id, 'response_too_large')
      }
      job.next++
      job.sizes.push(n)
      job.outstanding += n
    } else if (job.head && msg.type === 'end' && msg.next_seq === job.next) this.pending.delete(id)
    else return this.close('protocol')
    this.post(msg)
  }
  async flush(reason: Reason, id: string = crypto.randomUUID()) {
    if (!this.port) return !this.dead
    const ok = await new Promise<boolean>(resolve => {
      const timer = setTimeout(() => finish(false), 10000)
      const finish = (result: boolean) => {
        clearTimeout(timer)
        this.flushing.delete(id)
        resolve(result)
      }
      this.flushing.set(id, finish)
      this.post({ type: 'event', id, op: 'closing', body: { reason } })
    })
    if (!ok) this.post({ type: 'event', id, op: 'closeFailed', body: { reason, code: 'store_flush_failed' } })
    return ok
  }
  async lifecycle(event: Wire) {
    if (event.v !== 1 || event.type !== 'event' || !uuid.test(event.id) || !object(event.body)) return
    if (event.op === 'closing' && reasons.includes(String(event.body.reason)))
      return this.flush(event.body.reason as Reason, event.id)
    if (event.op === 'closeFailed' || event.op === 'updateProgress') this.post(event)
  }
  close(reason: Closing) {
    if (this.dead) return
    this.dead = true
    this.post({ type: 'close', reason })
    for (const [id, job] of this.pending)
      if (job.stream && !job.cancelled) void this.send({ v: 1, type: 'cancel', id }, job.reply).catch(() => {})
    this.pending.clear()
    this.flushing.forEach(done => done(false))
    if (this.port) {
      this.port.onmessage = this.port.onmessageerror = null
      this.port.close()
    }
    this.port = undefined
    this.revoked(reason)
  }
}
