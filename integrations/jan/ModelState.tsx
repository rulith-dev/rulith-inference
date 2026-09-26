import { Link } from '@tanstack/react-router'
import { useState } from 'react'
import { LoaderCircle, Play, RotateCw, TriangleAlert, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { clock, modelLabel, serverState, useStrixLlamaStatus } from './status'
import { useModelStore, useNextModel } from './store'
import { useTitlebarRightInset, useTr } from './parts'

const MEMORY_DISMISSED = 'strixllama-memory-dismissed'
const EDGE = { loading: 'border-l-[var(--rl-amber)]', stopped: 'border-l-[var(--rl-blue)]', failed: 'border-l-destructive' }

// Above the chat, as Rulith's desktop app has it ("Chat is ready. Start the Worker when you need..."): a
// strip that says why the model cannot answer yet, with the action that fixes it. Nothing while it is
// ready. The strip lies under the window's drag area, so only its buttons are lifted above it.
export function ModelBanner() {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const catalog = useModelStore(s => s.catalog)
  const next = useNextModel()
  const inset = useTitlebarRightInset()
  const busy = useModelStore(s => s.busy)
  const load = useModelStore(s => s.load)
  const state = serverState(status)
  const [dismissed, setDismissed] = useState(() => sessionStorage.getItem(MEMORY_DISMISSED) || '')
  if (!status) return null
  if (state === 'ready') {
    // a ready server that stopped answers for lack of memory: say so above the chat until dismissed
    const events = status.memory_events
    const key = `${status.identity?.pid}:${events?.answers_stopped}`
    if (!events?.answers_stopped || dismissed === key) return null
    return (
      <div className="flex min-h-12 shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-l-2 border-l-[var(--rl-amber)] bg-[var(--rl-banner)] px-4 py-2 text-[13px]" style={inset ? { paddingRight: inset } : undefined}>
        <TriangleAlert className="size-4 shrink-0 text-[var(--rl-amber)]" />
        <span className="text-foreground">{tr('memory.banner', { time: clock(events.last_stop), count: events.answers_stopped })}</span>
        <Button size="sm" variant="outline" className="relative z-30 h-7" asChild><Link to="/strixllama/configuration">{tr('actions.details')}</Link></Button>
        <Button size="icon-xs" variant="ghost" className="relative z-30" aria-label={tr('actions.dismiss')}
          onClick={() => { sessionStorage.setItem(MEMORY_DISMISSED, key); setDismissed(key) }}><X /></Button>
      </div>
    )
  }
  const loading = status.identity ? catalog?.models.find(m => m.path === status.model_path) : undefined
  const model = modelLabel(loading) || status.model_name || modelLabel(next) || tr('banner.theModel')
  const reason = status.failure_code ? tr(`errors.${status.failure_code}`) : status.failure
  return (
    <div className={`flex min-h-12 shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-l-2 bg-[var(--rl-banner)] px-4 py-2 text-[13px] ${EDGE[state]}`} style={inset ? { paddingRight: inset } : undefined}>
      {state === 'loading' && <LoaderCircle className="size-4 shrink-0 animate-spin text-[var(--rl-amber)]" />}
      <span className="text-foreground">
        {state === 'loading' ? tr('banner.loading', { model }) : state === 'failed' ? tr('banner.failed', { reason }) : tr('banner.stopped', { model })}
      </span>
      {state === 'stopped' && (
        <Button size="sm" className="relative z-30 h-7" disabled={busy || !status.runtime_available || !next} onClick={() => void load(next?.id)}>
          <Play className="size-3.5" />{tr('actions.load')}
        </Button>
      )}
      {state === 'failed' && <>
        <Button size="sm" className="relative z-30 h-7" disabled={busy || !status.runtime_available || !next} onClick={() => void load(next?.id)}>
          <RotateCw className="size-3.5" />{tr('actions.retry')}
        </Button>
        <Button size="sm" variant="outline" className="relative z-30 h-7" asChild><Link to="/strixllama/logs">{tr('actions.viewLog')}</Link></Button>
      </>}
    </div>
  )
}
