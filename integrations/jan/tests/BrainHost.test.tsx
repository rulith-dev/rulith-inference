import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { invoke } from '@tauri-apps/api/core'
import BrainHost from './BrainHost'
import { BrainBridge, brainEntry, requiredCaps, useMode, type Mode } from './brainBridge'

const navigate = vi.hoisted(() => vi.fn())
vi.mock('@tanstack/react-router', () => ({ useNavigate: () => navigate, useBlocker: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(), Channel: class {} }))
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}), emitTo: vi.fn() }))
vi.mock('@/containers/HeaderPage', () => ({ default: ({ children }: { children: React.ReactNode }) => children }))
vi.mock('./parts', () => ({ useTr: () => (key: string, vars?: { code?: string }) => vars?.code || key }))
const mode: Mode = { selected: 'inference', active: 'inference', target: null, state: 'inference_ready',
  can_retry: false, can_restore: false, owner: null, transition_id: null, previous_model_id: null, error: null }
const generation = '88f31d58-e5e5-4f34-b34c-73ba376aa07b'
let verified = false
let next: Mode
beforeEach(() => {
  vi.clearAllMocks()
  verified = false
  next = mode
  useMode.setState({ mode })
  vi.mocked(invoke).mockImplementation(async (cmd, args) => {
    const op = (args as { op?: string }).op
    if (cmd === 'brain_host_mode') return next
    if (cmd === 'brain_host_mount') return { url: brainEntry, generation }
    if (cmd === 'brain_host_unmount') return null
    if (cmd === 'brain_host_setup') return { installed: { verified, capabilities: requiredCaps },
      pending: null, job: op === 'brain_runtime_install' ? { alive: true, bytes: 12, total: 24 } : null }
    throw new Error(cmd)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })
describe('Brain route', () => {
  it('can install and cancel on first navigation while Inference serves', async () => {
    render(<BrainHost />)
    fireEvent.click(await screen.findByText('brain.install'))
    fireEvent.click(await screen.findByText('brain.cancel'))
    await screen.findByText('brain.install')
    expect(invoke).toHaveBeenCalledWith('brain_host_setup', { op: 'brain_runtime_cancel', data: {} })
    expect(invoke).not.toHaveBeenCalledWith('brain_host_mode', expect.anything())
    expect(document.querySelector('iframe')).toBeNull()
  })
  it('mounts a sandboxed frame, shows save failure, and retries the recorded transition', async () => {
    verified = true
    render(<BrainHost />)
    const frame = await screen.findByTitle('Rulith Brain')
    expect(frame).toHaveAttribute('src', brainEntry)
    expect(frame).toHaveAttribute('sandbox', 'allow-scripts allow-same-origin')
    next = { ...mode, state: 'save_failed', target: 'brain', transition_id: generation, can_retry: true,
      error: { code: 'persist_failed', params: {} } }
    fireEvent.click(screen.getByText('brain.switch'))
    await screen.findByText('persist_failed')
    expect(invoke).toHaveBeenCalledWith('brain_host_unmount', { generation })
    fireEvent.click(screen.getByText('brain.retry'))
    await waitFor(() => expect(invoke).toHaveBeenCalledWith('brain_host_mode', {
      op: 'mode_retry', data: { request_id: expect.any(String), transition_id: generation, action: 'retry' },
    }))
  })
  it('keeps a stopped Brain selected and returns to model selection only when the coordinator commits', async () => {
    useMode.setState({ mode: { ...mode, selected: 'brain', active: 'none', state: 'idle' } })
    next = { ...mode, selected: 'brain', active: 'none', state: 'checking', target: 'inference' }
    render(<BrainHost />)
    fireEvent.click(await screen.findByText('brain.return'))
    await screen.findByText('brain.states.checking')
    expect(navigate).not.toHaveBeenCalled()
    act(() => useMode.setState({ mode: { ...mode, active: 'none', state: 'idle' } }))
    await waitFor(() => expect(navigate).toHaveBeenCalledWith({ to: '/strixllama/models' }))
  })
  it('keeps the frame and engine in place when the UI cannot flush', async () => {
    verified = true
    vi.spyOn(BrainBridge.prototype, 'flush').mockResolvedValue(false)
    render(<BrainHost />)
    const frame = await screen.findByTitle('Rulith Brain')
    fireEvent.click(screen.getByText('brain.switch'))
    await screen.findByText('store_flush_failed')
    expect(screen.getByTitle('Rulith Brain')).toBe(frame)
    expect(invoke).not.toHaveBeenCalledWith('brain_host_mode', expect.anything())
    expect(invoke).not.toHaveBeenCalledWith('brain_host_unmount', expect.anything())
  })
})
