import { useEffect, useRef, useState } from 'react'
import { useBlocker, useNavigate } from '@tanstack/react-router'
import { invoke } from '@tauri-apps/api/core'
import { emitTo, listen } from '@tauri-apps/api/event'
import HeaderPage from '@/containers/HeaderPage'
import { Button } from '@/components/ui/button'
import { useTr } from './parts'
import {
  BrainBridge, changeMode, errorCode, nativeSender, requiredCaps, useMode, validMount,
  type Mount, type Runtime, type Wire,
} from './brainBridge'
import './strixllama.css'

const runtimeOp = (op: string) => invoke<Runtime>('brain_host_setup', { op, data: {} })
const unmount = (generation: string) => invoke('brain_host_unmount', { generation })
// Route changes and strict-mode remounts can overlap an asynchronous native mount.
// Finish releasing the previous lease before asking native for a new one.
let mapping = Promise.resolve()

export default function BrainHost() {
  const tr = useTr()
  const navigate = useNavigate()
  const mode = useMode(s => s.mode)
  const modeError = useMode(s => s.error)
  const [runtime, setRuntime] = useState<Runtime>()
  const [setupError, setSetupError] = useState('')
  const [actionError, setActionError] = useState('')
  const [mountError, setMountError] = useState('')
  const [busy, setBusy] = useState(false)
  const [paused, setPaused] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [available, setAvailable] = useState('')
  const container = useRef<HTMLDivElement>(null)
  const bridge = useRef<BrainBridge | null>(null)
  const working = useRef(false)
  const returning = useRef(false)
  const setupVersion = useRef(0)
  const ready = !!runtime?.installed.verified && requiredCaps.every(c => runtime.installed.capabilities.includes(c))
  const transitioning = !!mode && !['idle', 'brain_ready', 'inference_ready', 'blocked', 'launch_failed', 'save_failed'].includes(mode.state)
  const mapped = ready && !paused && !mountError

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      const rev = setupVersion.current
      try {
        const result = await runtimeOp('brain_runtime')
        if (!stopped && rev === setupVersion.current) { setRuntime(result); setSetupError('') }
      } catch (e) { if (!stopped && rev === setupVersion.current) setSetupError(errorCode(e)) }
      if (!stopped) timer = setTimeout(poll, 1500)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [])

  useEffect(() => {
    if (!mapped) return
    let disposed = false
    let mounted: Mount | undefined
    let current: BrainBridge | undefined
    let frame: HTMLIFrameElement | undefined
    let handshakeTimer: ReturnType<typeof setTimeout> | undefined
    let unlisten: (() => void) | undefined
    let release: () => void
    const lifetime = new Promise<void>(resolve => { release = resolve })
    mapping = mapping.catch(() => {}).then(async () => {
      if (disposed || !container.current) return
      try {
        mounted = await invoke<Mount>('brain_host_mount', {})
        if (!validMount(mounted)) throw { code: 'runtime_verification' }
        if (disposed) return
        const generation = mounted.generation
        frame = document.createElement('iframe')
        frame.title = 'Rulith Brain'
        frame.setAttribute('sandbox', 'allow-scripts allow-same-origin')
        frame.referrerPolicy = 'no-referrer'
        current = new BrainBridge(frame, requiredCaps, nativeSender(generation), reason => {
          if (!disposed) setMountError(reason === 'protocol' ? 'bridge_protocol' : 'bridge_closed')
          release()
        })
        const connection = current
        unlisten = await listen<{ generation: string; event: Wire }>('brain-host-lifecycle', async ({ payload }) => {
          if (disposed || payload.generation !== generation) return
          const ok = await connection.lifecycle(payload.event)
          if (typeof ok === 'boolean') await emitTo('main', 'brain-host-flushed', {
            generation, v: 1, type: 'flushed', id: payload.event.id, ok, ...(!ok && { code: 'store_flush_failed' }),
          })
        })
        if (disposed) return
        bridge.current = current
        window.addEventListener('message', current.helloFrom)
        frame.addEventListener('load', current.documentLoaded)
        frame.onerror = () => connection.close('protocol')
        frame.src = mounted.url
        container.current?.appendChild(frame)
        handshakeTimer = setTimeout(() => {
          if (!connection.connected) connection.close('protocol')
        }, 30000)
        await lifetime
      } catch (e) { if (!disposed) setMountError(errorCode(e)) }
      finally {
        clearTimeout(handshakeTimer)
        if (current) {
          window.removeEventListener('message', current.helloFrom)
          frame?.removeEventListener('load', current.documentLoaded)
          current.close('navigate')
        }
        if (bridge.current === current) bridge.current = null
        frame?.remove()
        unlisten?.()
        if (mounted) await unmount(mounted.generation)
      }
    })
    void mapping.catch(e => { if (!disposed) setMountError(errorCode(e)) })
    return () => {
      disposed = true
      current?.close('navigate')
      release()
    }
  }, [mapped, attempt])

  async function flush(reason: 'switch' | 'navigate' | 'update') {
    const current = bridge.current
    if (current && !(await current.flush(reason))) {
      setActionError('store_flush_failed')
      return false
    }
    return true
  }
  useBlocker({
    enableBeforeUnload: false,
    shouldBlockFn: async ({ current, next }) => current.pathname !== next.pathname && !(await flush('navigate')),
  })
  useEffect(() => {
    if (returning.current && mode?.selected === 'inference' && mode.target === null
      && ['idle', 'inference_ready'].includes(mode.state)) {
      returning.current = false
      void navigate({ to: '/strixllama/models' })
    }
  }, [mode, navigate])
  useEffect(() => {
    if (paused && !busy && !transitioning) setPaused(false)
  }, [paused, busy, transitioning])
  const lastSelection = useRef(mode?.selected)
  useEffect(() => {
    const failed = () => {
      void bridge.current?.lifecycle({ v: 1, type: 'event', id: crypto.randomUUID(),
        op: 'closeFailed', body: { reason: 'update', code: 'update_failed' } })
      setActionError('update_failed')
      setAttempt(n => n + 1)
    }
    window.addEventListener('brain-host-update-failed', failed)
    return () => window.removeEventListener('brain-host-update-failed', failed)
  }, [])
  useEffect(() => {
    if (!mode) return
    if (lastSelection.current && lastSelection.current !== mode.selected) {
      // A Start from inside the frame also goes through the native coordinator.
      // Once its mode commits, release that generation and show the new session.
      setAttempt(n => n + 1)
    }
    lastSelection.current = mode.selected
  }, [mode])

  async function act(fn: () => Promise<void>) {
    if (working.current) return
    working.current = true
    setBusy(true)
    setActionError('')
    try { await fn() } catch (e) { setActionError(errorCode(e)) }
    finally { working.current = false; setBusy(false) }
  }
  async function setup(op: string) {
    ++setupVersion.current
    try {
      const result = await runtimeOp(op)
      setSetupError('')
      setRuntime(result)
      if (result.available) setAvailable(result.available.newer ? result.available.version : '')
      if (op === 'brain_runtime_install') setAvailable('')
    } finally { ++setupVersion.current }
  }
  async function switchTo(destination: 'brain' | 'inference' | 'retry' | 'restore') {
    if (!mode || !(await flush('switch'))) return
    returning.current = destination === 'inference' || (destination === 'retry' && mode.target === 'inference')
      || (destination === 'restore' && mode.selected === 'inference')
    setPaused(true)
    // Invalidate the port synchronously, before the native worker can stop its source.
    bridge.current?.close('switch')
    setMountError('')
    try {
      await mapping
      if (destination === 'retry' || destination === 'restore')
        await changeMode('mode_retry', { transition_id: mode.transition_id, action: destination })
      else await changeMode(destination === 'brain' ? 'mode_switch_brain' : 'mode_switch_inference', {})
    } catch (e) { returning.current = false; throw e }
  }
  const job = runtime?.job
  const code = actionError || mountError || setupError || modeError || mode?.error?.code || job?.error?.code || runtime?.installed.reason
  return (
    <div className="brain-host">
      <HeaderPage><h1 className="px-4 text-[15px] font-semibold">{tr('brain.title')}</h1></HeaderPage>
      <section className="brain-controls" aria-label={tr('brain.title')}>
        <p role="status">{tr(mode ? 'brain.states.' + mode.state : 'brain.connecting')}</p>
        {mode?.owner && mode.owner.host !== 'inference' &&
          <p>{tr('brain.owner', { owner: mode.owner.exe || mode.owner.host, pid: mode.owner.pid ?? '—' })}</p>}
        {code && <p role="alert">{tr('brain.problem', { code })}</p>}
        {(!ready || runtime?.pending || job?.alive || available) && <div>
          {!ready && <p>{tr('brain.setup')}</p>}
          {available && <p>{tr('brain.available', { version: available })}</p>}
          {runtime?.pending && <p>{tr('brain.pending', { version: runtime.pending })}</p>}
          {job?.alive && <label>{tr('brain.downloading', { got: job.bytes.toLocaleString(), total: job.total.toLocaleString() })}
            <progress value={job.bytes} max={Math.max(1, job.total)} /></label>}
        </div>}
        <div className="brain-actions">
          {runtime && !job?.alive && (!ready || available || mountError) &&
            <Button disabled={busy} onClick={() => void act(() => setup('brain_runtime_install'))}>{tr('brain.install')}</Button>}
          {runtime && !job?.alive && <Button variant="outline" disabled={busy}
            onClick={() => void act(() => setup('brain_runtime_check'))}>{tr('brain.check')}</Button>}
          {job?.alive && <Button variant="outline" disabled={busy}
            onClick={() => void act(() => setup('brain_runtime_cancel'))}>{tr('brain.cancel')}</Button>}
          {runtime?.pending && ready && !transitioning && mode?.active !== 'brain' && <Button disabled={busy} onClick={() => void act(async () => {
            if (!(await flush('update'))) return
            setPaused(true)
            bridge.current?.close('update')
            setMountError('')
            try { await mapping; await setup('brain_runtime') } finally { setPaused(false); setAttempt(v => v + 1) }
          })}>{tr('brain.apply')}</Button>}
          {mountError && ready && <Button disabled={busy} onClick={() => { setMountError(''); setAttempt(v => v + 1) }}>{tr('brain.reopen')}</Button>}
          {ready && mode?.selected === 'inference' && ['idle','inference_ready'].includes(mode.state) &&
            <Button disabled={busy} onClick={() => void act(() => switchTo('brain'))}>{tr('brain.switch')}</Button>}
          {mode?.selected === 'brain' && ['idle','brain_ready'].includes(mode.state) &&
            <Button disabled={busy} onClick={() => void act(() => switchTo('inference'))}>{tr('brain.return')}</Button>}
          {mode?.can_retry && <Button disabled={busy} onClick={() => void act(() => switchTo('retry'))}>{tr('brain.retry')}</Button>}
          {mode?.can_restore && <Button variant="outline" disabled={busy} onClick={() => void act(() => switchTo('restore'))}>{tr('brain.restore')}</Button>}
        </div>
      </section>
      <div className="brain-viewport" ref={container} />
    </div>
  )
}
