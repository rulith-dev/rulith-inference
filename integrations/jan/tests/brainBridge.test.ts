import { afterEach, describe, expect, it, vi } from 'vitest'
import { BrainBridge, brainOrigin, requiredCaps, inferenceAllowed, validMount, type Mode, type Wire } from './brainBridge'

const id = '897e58bb-5c06-47bd-a58f-360e8b5b8143'
const request = (op: string, body = {}) => ({ v: 1, type: 'request', id, op, body })
const clean: BrainBridge[] = []
afterEach(() => { clean.splice(0).forEach(b => b.close('navigate')); vi.unstubAllGlobals() })

function fixture() {
  const port = { onmessage: null as null | ((e: unknown) => void), onmessageerror: null, postMessage: vi.fn(), close: vi.fn() }
  vi.stubGlobal('MessageChannel', class { port1 = port; port2 = {} })
  const child = { postMessage: vi.fn() }
  const send = vi.fn(async (_: Wire, _reply: (e: Wire) => void) => {})
  const revoked = vi.fn()
  const bridge = new BrainBridge({ contentWindow: child } as unknown as HTMLIFrameElement, [...requiredCaps, 'desktop.quit'], send, revoked)
  clean.push(bridge)
  const hello = (source = child, origin = brainOrigin) =>
    bridge.helloFrom({ ports: [], data: { v: 1, type: 'hello' }, source, origin } as unknown as MessageEvent)
  const connect = () => { hello(); bridge.documentLoaded() }
  const receive = (data: unknown, ports: object[] = []) => port.onmessage?.({ data, ports })
  return { bridge, child, port, send, revoked, hello, connect, receive }
}
describe('Brain parent port', () => {
  it('waits for the expected document, filters capabilities and transfers a port once', () => {
    const f = fixture()
    f.hello({ postMessage: vi.fn() })
    f.hello(f.child, 'https://other.invalid')
    f.bridge.documentLoaded()
    expect(f.child.postMessage).not.toHaveBeenCalled()
    f.hello(); f.hello()
    expect(f.child.postMessage).toHaveBeenCalledTimes(1)
    expect(f.child.postMessage).toHaveBeenCalledWith({ v: 1, type: 'connect', host: 'inference', capabilities: requiredCaps }, brainOrigin, [{}])
  })
  it('closes on replacement navigation and discards late native responses', () => {
    const f = fixture(); f.connect(); f.receive(request('status'))
    const reply = f.send.mock.calls[0][1]
    f.bridge.documentLoaded()
    reply({ v: 1, type: 'result', id, body: {} })
    expect(f.revoked).toHaveBeenCalledWith('navigate')
    expect(f.port.postMessage).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'result' }))
  })
  it.each(['mode_switch_brain', 'network', 'manager', 'quit_app', 'brain', 'toString'])('denies %s', op => {
    const f = fixture(); f.connect(); f.receive(request(op))
    expect(f.send).not.toHaveBeenCalled()
    expect(f.port.postMessage).toHaveBeenCalledWith(expect.objectContaining({ code: 'op_denied', id }))
  })
  it('rejects transferred ports, unknown fields, cyclic values and duplicate requests', () => {
    const f = fixture(); f.connect(); f.receive(request('stop', { force: true }))
    expect(f.port.postMessage).toHaveBeenCalledWith(expect.objectContaining({ code: 'request_invalid' }))
    f.receive(request('stop'))
    expect(f.revoked).toHaveBeenCalledWith('protocol')
    const g = fixture(); g.connect(); g.receive(request('status'), [{}])
    expect(g.send).not.toHaveBeenCalled()
    const h = fixture(); h.connect()
    const circular: { value?: object } = {}; circular.value = circular
    h.receive(request('ui_store_write', { name: 'settings', expected_rev: 0, value: circular }))
    expect(h.send).not.toHaveBeenCalled()
  })
  it('binds ack and cancellation to the active request channel and delivers one terminal event', () => {
    const f = fixture(); f.connect(); f.receive(request('memory_stream', { method: 'GET', path: '/slots' }))
    const reply = f.send.mock.calls[0][1]
    reply({ v: 1, type: 'head', id, status: 200, headers: {} })
    reply({ v: 1, type: 'chunk', id, seq: 0, text: '帧\n\n' })
    f.receive({ v: 1, type: 'ack', id, next_seq: 1 })
    f.receive({ v: 1, type: 'cancel', id })
    f.receive({ v: 1, type: 'cancel', id })
    expect(f.send.mock.calls.map(([m, fn]) => [m.type, fn === reply])).toEqual([['request', true], ['ack', true], ['cancel', true]])
    reply({ v: 1, type: 'error', id, code: 'cancelled', params: {}, message: 'cancelled' })
    reply({ v: 1, type: 'end', id, next_seq: 1 })
    expect(f.port.postMessage.mock.calls.map(([m]) => m.type)).toEqual(['head','chunk','error'])
  })
  it('keeps a streaming chat connected when the first native acknowledgement is delayed', async () => {
    const f = fixture()
    let release!: () => void
    const first = new Promise<void>(resolve => { release = resolve })
    let applied = 0
    f.send.mockImplementation(async m => {
      if (m.type !== 'ack') return
      const next = Number(m.next_seq)
      if (next === 1) await first
      if (next < applied) throw new Error('bridge_protocol')
      applied = next
    })
    f.connect()
    f.receive(request('memory_stream', { method: 'POST', path: '/v1/memory/input' }))
    const reply = f.send.mock.calls[0][1]
    reply({ v: 1, type: 'head', id, status: 200, headers: {} })
    for (let seq = 0; seq < 3; seq++) {
      reply({ v: 1, type: 'chunk', id, seq, text: 'hello' })
      f.receive({ v: 1, type: 'ack', id, next_seq: seq + 1 })
    }
    release()
    await Promise.allSettled(f.send.mock.results.map(result => result.value))
    await vi.waitFor(() => expect(applied).toBe(3))
    expect(f.revoked).not.toHaveBeenCalled()
    reply({ v: 1, type: 'end', id, next_seq: 3 })
    expect(f.port.postMessage.mock.lastCall?.[0]).toMatchObject({ type: 'end', next_seq: 3 })
  })
  it('vetoes a failed flush and resumes editing on the same port', async () => {
    const f = fixture(); f.connect()
    const done = f.bridge.flush('switch', id)
    f.receive({ v: 1, type: 'flushed', id, ok: false, code: 'store_conflict' })
    await expect(done).resolves.toBe(false)
    expect(f.port.postMessage.mock.lastCall?.[0]).toMatchObject({ op: 'closeFailed', body: { code: 'store_flush_failed' } })
    f.receive(request('ui_store_read', { name: 'settings' }))
    expect(f.send).toHaveBeenCalledOnce()
  })
  it.each([
    ['brain_activate', { key: 'fixture', id: 'override' }],
    ['brain_create', { name: 'test', id: 'override' }],
    ['dl_start', { model: 'arbitrary', include: ['draft'] }],
    ['memory', { method: 'POST', path: '/v1/memory/%2e%2e/status' }],
    ['memory', { method: 'GET', path: '/v1/memory/%252f' }],
    ['open_link', { url: 'https://github.com/other/' }],
    ['set_theme', { theme: 'unknown' }],
  ])('rejects invalid projected arguments for %s', (op, body) => {
    const f = fixture(); f.connect(); f.receive(request(String(op), body as object))
    expect(f.send).not.toHaveBeenCalled()
    expect(f.port.postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'error', id }))
  })
  it('blocks a manager request over 64 KiB and accepts exactly 64 KiB', () => {
    const body = { dir: '', text: '' }
    const overhead = JSON.stringify({ op: 'brain_self_set', data: body }).length
    body.dir = 'a'.repeat(65536 - overhead)
    const f = fixture(); f.connect(); f.receive(request('brain_self_set', body))
    expect(f.send).toHaveBeenCalledOnce()
    const g = fixture(); g.connect(); g.receive(request('brain_self_set', { ...body, dir: body.dir + 'a' }))
    expect(g.port.postMessage).toHaveBeenCalledWith(expect.objectContaining({ code: 'request_too_large' }))
  })
  it('requires the native entry point and a reconciled Inference selection', () => {
    expect(validMount({ url: brainOrigin + '/index.html?origin=other', generation: id })).toBe(false)
    const idle = { state: 'idle', active: 'none', selected: 'inference', target: null } as Mode
    expect(inferenceAllowed(idle)).toBe(true)
    for (const m of [null, { ...idle, selected: 'brain' }, { ...idle, state: 'recovering' }, { ...idle, target: 'brain' }])
      expect(inferenceAllowed(m as Mode | null)).toBe(false)
  })
})
