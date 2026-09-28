import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ChevronDown, Copy, Download, Pause, Play, Search } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import { describeError, request, rocmLabel, serverState, useStrixLlamaStatus, type LogChunk, type SlotInfo } from './status'
import { Choice, Label, StatePill, useTr } from './parts'

// a server log line's level is the letter after its timestamp (`2.13.732.236 E srv ...`); a line without one -
// an assert, a ROCm error, a fused kernel's one-time note - is an error only by its words
const logLevel = (l: string): 'error' | 'warning' | '' => {
  const m = /^\s*\d+\.\d+\.\d+\.\d+ ([EW]) /.exec(l)
  if (m) return m[1] === 'E' ? 'error' : 'warning'
  return /^\s*\d+\.\d+\.\d+\.\d+ /.test(l) ? '' : /GGML_ASSERT|\berror\b|\bfailed\b|\babort/i.test(l) ? 'error' : ''
}

export default function LogsView() {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const [text, setText] = useState('')
  const [query, setQuery] = useState('')
  const [level, setLevel] = useState('all')
  const [paused, setPaused] = useState(false)
  const [follow, setFollow] = useState(true)
  // long lines scroll sideways by default, as in a terminal; wrapping is a choice that is remembered
  const [wrap, setWrap] = useState(() => localStorage.getItem(WRAP_KEY) === '1')
  const [error, setError] = useState('')
  const [params, setParams] = useState(false)
  const [slots, setSlots] = useState<SlotInfo[]>([])
  const cursor = useRef({ offset: 0, file: '' })
  const box = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (paused) return
    let disposed = false, pending = false
    const poll = async () => {
      if (pending) return
      pending = true
      try {
        let r = await request<LogChunk>('logs', { offset: cursor.current.offset })
        if (disposed) return
        if (r.file !== cursor.current.file) { cursor.current = { file: r.file, offset: 0 }; setText(''); r = await request<LogChunk>('logs', { offset: 0 }) }
        if (disposed) return
        cursor.current = { offset: r.offset, file: r.file }
        setText(t => ((r.reset ? '' : t) + r.text).split('\n').slice(-2500).join('\n'))
        setError('')
      } catch (e) { if (!disposed) setError(describeError(e, tr)) } finally { pending = false }
    }
    void poll()
    const timer = setInterval(poll, 1200)
    return () => { disposed = true; clearInterval(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused])
  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight }, [text, follow, query, level])

  // what each conversation slot is doing, while a server runs: generating (tokens so far), reading a prompt, or idle
  // with its conversation still in the cache. A busy server can answer late; the last answer stays on screen then.
  const running = !!status?.identity
  // the server's throughput summed over its conversations, from the same polls (see Throughput)
  const meter = useRef(new Throughput())
  const [rate, setRate] = useState<{ answers: number; prompts: number } | null>(null)
  // requests the server holds in its queue: no free slot, or no room in the KV pool yet (0.3.3 runtimes say)
  const [queued, setQueued] = useState(0)
  useEffect(() => {
    meter.current = new Throughput()
    setRate(null)
    if (!running) { setSlots([]); return }
    let disposed = false, pending = false
    const poll = async () => {
      if (pending) return
      pending = true
      try {
        const r = await request<{ slots: SlotInfo[] | null; waiting?: number }>('slots')
        if (!disposed && r.slots) {
          setSlots(r.slots)
          setQueued(r.waiting ?? 0)
          setRate(meter.current.add(r.slots, performance.now()))
        }
      } catch { /* the next poll tries again */ } finally { pending = false }
    }
    void poll()
    const timer = setInterval(poll, 1500)
    return () => { disposed = true; clearInterval(timer) }
  }, [running])

  const lines = text.split('\n').filter(l => l.toLowerCase().includes(query.toLowerCase())
    && (level === 'all' || (level === 'error' ? logLevel(l) === 'error' : logLevel(l) !== '')))
  const copy = (value: string, done: string) => navigator.clipboard.writeText(value).then(() => toast.success(done), e => setError(String(e)))
  const exportLog = () => {
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' }))
    const a = document.createElement('a')
    a.href = url
    a.download = 'strixllama-server.log'
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const state = serverState(status)
  const apiModel = status?.identity ? status.served_models?.[0]?.id : undefined
  const qsa = status?.runtime_env?.LLAMA_QSA_SPARSE
  const runtime = [rocmLabel(status), status?.runtime_info?.gfx, status?.identity && qsa !== undefined && tr('logs.qsa', { state: tr(qsa === '0' ? 'state.off' : 'state.on') })].filter(Boolean).join(' · ')

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-5">
      <section className="shrink-0">
        <Label>{tr('logs.server')}</Label>
        <div className="rounded-lg border bg-card px-4 py-3">
        <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-[13px] lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(0,1.2fr)_auto]">
          <Fact label={tr('logs.state')}><StatePill state={state} /></Fact>
          <Fact label={tr('logs.model')}>
            <div className="min-w-0">
              <div className="truncate">{status?.identity ? status.model_name : '—'}</div>
              {apiModel && (
                // the name API clients send as "model" (llama-server's /v1/models id), as the endpoint beside it
                <button type="button" className="mt-0.5 flex min-w-0 max-w-full items-center gap-1.5 font-mono text-xs text-muted-foreground hover:text-foreground"
                  title={tr('logs.modelIdHint')} onClick={() => void copy(apiModel, tr('logs.modelIdCopied'))}>
                  <span className="truncate">{apiModel}</span><Copy className="size-3.5 shrink-0" />
                </button>
              )}
            </div>
          </Fact>
          <Fact label={tr('logs.runtime')}><span title={status?.runtime_info?.rocm ? `ROCm ${status.runtime_info.rocm}` : undefined}>{runtime}</span></Fact>
          <Fact label={tr('logs.endpoint')}>
            <button type="button" className="flex min-w-0 items-center gap-1.5 font-mono text-xs hover:text-foreground" title={tr('actions.copy')}
              onClick={() => void copy(status?.endpoint || '', tr('logs.endpointCopied'))}>
              <span className="whitespace-nowrap">{status?.endpoint || 'http://127.0.0.1:8080/v1'}</span><Copy className="size-3.5 shrink-0" />
            </button>
          </Fact>
        </div>
        {slots.length > 0 && (
          <div className="mt-3 border-t border-border/40 pt-3">
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <div className="text-[11px] uppercase tracking-[0.06em] text-[var(--rl-label)]">
                {tr('logs.conversations', { active: slots.filter(x => x.active).length, total: slots.length })}
                {queued > 0 && <span title={tr('logs.queuedTitle')}> · {tr('logs.queued', { count: queued })}</span>}
              </div>
              {rate && (
                <div className="font-mono text-xs text-muted-foreground" title={tr('logs.throughputTitle', { seconds: THROUGHPUT_WINDOW_MS / 1000 })}>
                  {tr('logs.throughput', { answers: perSecond(rate.answers), prompts: perSecond(rate.prompts) })}
                </div>
              )}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">{slots.map(x => <SlotChip key={x.id} slot={x} />)}</div>
          </div>
        )}
        <Collapsible open={params} onOpenChange={setParams} className="mt-3 border-t border-border/40 pt-3">
          <CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
            <ChevronDown className={cn('size-3.5 transition-transform', params && 'rotate-180')} />
            {tr('logs.launchParams')}{status?.identity?.pid ? ` · PID ${status.identity.pid}` : ''}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-md bg-secondary/50 p-3 font-mono text-xs text-muted-foreground">
              {status?.command || tr('logs.notStarted')}
              {status?.runtime_env && `\n\n${Object.entries(status.runtime_env).map(([k, v]) => `${k}=${v}`).join('\n')}`}
            </pre>
          </CollapsibleContent>
        </Collapsible>
        </div>
      </section>

      <section className="flex min-h-0 flex-1 flex-col">
        <Label>{tr('logs.log')}</Label>
        <div className="flex min-h-0 flex-1 flex-col gap-2 rounded-lg border bg-card p-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-48 flex-1">
            <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input aria-label={tr('logs.searchLabel')} className="h-8 pl-8 text-[13px]" placeholder={tr('logs.search')} value={query} onChange={e => setQuery(e.target.value)} />
          </div>
          <Choice ariaLabel={tr('logs.levelLabel')} value={level} onChange={setLevel} options={[
            { value: 'all', label: tr('logs.levelAll') }, { value: 'warning', label: tr('logs.levelWarning') }, { value: 'error', label: tr('logs.levelError') }]} />
          <Button size="sm" variant="outline" className="h-8" onClick={() => setPaused(!paused)}>
            {paused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}{paused ? tr('logs.resume') : tr('logs.pause')}
          </Button>
          <Button size="sm" variant="outline" className="h-8" onClick={() => void copy(lines.join('\n'), tr('logs.copied'))}><Copy className="size-3.5" />{tr('actions.copy')}</Button>
          <Button size="sm" variant="outline" className="h-8" onClick={exportLog}><Download className="size-3.5" />{tr('logs.export')}</Button>
        </div>
        {error && <div className="text-xs text-destructive">{error}</div>}
        <pre ref={box} tabIndex={0} role="log" aria-label={tr('logs.logLabel')} aria-live="off"
          onScroll={e => { const b = e.currentTarget; setFollow(b.scrollHeight - b.clientHeight - b.scrollTop < 24) }}
          className="strixllama-logs min-h-0 min-w-0 flex-1 rounded-md border bg-[#0e0e0e] p-3 font-mono text-xs leading-5 text-zinc-300">
          {lines.length && text ? lines.map((l, i) => (
            <div key={i} className={cn(wrap ? 'whitespace-pre-wrap break-words' : 'w-max min-w-full whitespace-pre', logLevel(l) === 'error' ? 'text-red-400' : logLevel(l) === 'warning' ? 'text-amber-400' : /tokens per second|acceptance|tg =/.test(l) ? 'text-emerald-400' : '')}>{l || ' '}</div>
          )) : <span className="text-zinc-500">{tr('logs.waiting')}</span>}
        </pre>
        <div className="flex items-center justify-between gap-4 text-xs text-muted-foreground">
          <span className="min-w-0 truncate font-mono" title={cursor.current.file}>{cursor.current.file}</span>
          <label className="ml-auto flex shrink-0 items-center gap-2">
            {tr('logs.wrap')}
            <Switch checked={wrap} onCheckedChange={v => { setWrap(v); localStorage.setItem(WRAP_KEY, v ? '1' : '0') }} />
          </label>
          <label className="flex shrink-0 items-center gap-2">
            {tr('logs.autoscroll')}
            <Switch checked={follow} onCheckedChange={setFollow} />
          </label>
        </div>
        </div>
      </section>
    </div>
  )
}

const WRAP_KEY = 'strixllama-log-wrap'

const kTokens = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n))

const THROUGHPUT_WINDOW_MS = 10000

const perSecond = (n: number) => (n < 10 ? n.toFixed(1) : Math.round(n).toLocaleString())

// Answer and prompt tokens a second, summed over the server's conversations, over the last THROUGHPUT_WINDOW_MS. A slot's
// counters (llama-server's /slots: n_decoded, n_prompt_tokens_processed) grow while one task runs and start over with the
// next, so each poll adds what they grew by since the last one - all of a new task's, none of a slot seen the first time.
// Tokens a task makes between the last poll and its end are missed: at most one poll interval's worth.
export class Throughput {
  private last = new Map<number, { task: number | null | undefined; answers: number; prompts: number }>()
  private total = { answers: 0, prompts: 0 }
  private history: { t: number; answers: number; prompts: number }[] = []

  add(slots: SlotInfo[], now: number): { answers: number; prompts: number } | null {
    for (const s of slots) {
      const prev = this.last.get(s.id)
      if (prev) {
        const same = prev.task === s.task
        this.total.answers += Math.max(0, s.generated - (same ? prev.answers : 0))
        this.total.prompts += Math.max(0, s.prompt_processed - (same ? prev.prompts : 0))
      }
      this.last.set(s.id, { task: s.task, answers: s.generated, prompts: s.prompt_processed })
    }
    const h = this.history
    h.push({ t: now, ...this.total })
    while (h.length > 1 && now - h[0].t > THROUGHPUT_WINDOW_MS) h.shift()
    const dt = (now - h[0].t) / 1000
    // a first figure only after a couple of polls: one interval's rate swings with every token burst
    return dt >= 2 ? { answers: (this.total.answers - h[0].answers) / dt, prompts: (this.total.prompts - h[0].prompts) / dt } : null
  }
}

// One slot, as LM Studio shows a loaded model's sequences: what it is doing and how far, and how long the
// conversation it holds is
function SlotChip({ slot }: { slot: SlotInfo }) {
  const tr = useTr()
  const generating = slot.active && slot.generated > 0
  const reading = slot.active && !generating
  const what = generating ? tr('logs.slotGenerating', { n: slot.generated.toLocaleString() })
    : reading ? tr('logs.slotPrefill', { n: slot.prompt_processed.toLocaleString() })
    : slot.context > 0 ? tr('logs.slotIdle') : tr('logs.slotEmpty')
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-xs',
      generating ? 'border-[var(--rl-green)]/40 text-foreground' : reading ? 'border-[var(--rl-amber)]/40 text-foreground' : 'text-muted-foreground')}
      title={tr('logs.slotTitle', { id: slot.id, context: slot.context.toLocaleString(), generated: slot.generated.toLocaleString(), cached: slot.prompt_cached.toLocaleString() })}>
      <span className={cn('size-1.5 rounded-full', generating ? 'animate-pulse bg-[var(--rl-green)]' : reading ? 'animate-pulse bg-[var(--rl-amber)]' : 'bg-muted-foreground/40')} />
      <span className="text-muted-foreground">#{slot.id}</span>
      <span>{what}</span>
      {slot.context > 0 && <span className="text-muted-foreground">· {kTokens(slot.context)}</span>}
    </span>
  )
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] uppercase tracking-[0.06em] text-[var(--rl-label)]">{label}</div>
      <div className="mt-1 flex min-w-0 items-center text-foreground">{children}</div>
    </div>
  )
}
