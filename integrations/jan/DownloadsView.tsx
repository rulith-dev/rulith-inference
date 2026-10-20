import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Link } from '@tanstack/react-router'
import { revealItemInDir } from '@tauri-apps/plugin-opener'
import { CheckCircle2, CircleAlert, CirclePause, Download, FolderOpen, LoaderCircle, Search } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useServiceHub } from '@/hooks/useServiceHub'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { RenderMarkdown } from '@/containers/RenderMarkdown'
import { cn } from '@/lib/utils'
import { describeError, request } from './status'
import { useModelStore } from './store'
import { Label, Notice, RowAction, Segments, openExternal, useTr } from './parts'

// Model downloads (tools/downloader.py, the manager's dl_* ops): the models this build is tested with, and any GGUF
// repository found by searching Hugging Face (or opened by its link), with its model card. One download runs at a time, in a worker process
// that outlives this page and the app; the page polls it while it runs.

type DlFile = { name: string; role: string; required: boolean; size: number; present: string | null }
type DlModel = { id: string; name: string; variant: string; owner: string; repo: string; license: string; capabilities: string[]; files: DlFile[]; missing: number; fits: boolean | null; commit?: string }
type JobFile = { name: string; size: number; bytes: number; status: string; verified?: boolean | null; source: number }
type JobError = { code: string; params: Record<string, unknown> }
export type Job = { id: string; kind: string; title: string; model: string | null; folder: string; source: string; created_at: string; status: string; alive: boolean; error?: JobError | null; files: JobFile[]; bytes: number; total: number; speed: number; eta: number | null }
type Drive = { drive: string; free: number; total: number }
type DlCatalog = { models: DlModel[]; sources: Record<string, string>; drives: Drive[]; default_root: string | null; jobs: Job[]; active: string | null }
type RepoFile = { name: string; path: string; size: number; sha256: string | null; present: string | null }
type RepoGroup = { key: string; role: string; name: string; quant: string; size: number; complete: boolean; shards: number; files: RepoFile[]; fits: boolean | null; tuned?: boolean }
type SearchHit = { id: string; owner: string; repo: string; downloads: number; likes: number; updated: string | null; task: string | null; architecture: string | null }
// what the right-hand pane shows: a recommended model, or a repository from the search
type Pick = { kind: 'catalog'; id: string } | { kind: 'repo'; id: string }
type RepoListing = { source: string; owner: string; repo: string; commit: string; groups: RepoGroup[]; folder: string | null }
// what the start dialog sends to dl_start, less the source and folder it asks for
type Plan = { kind: 'catalog'; model: string; include: string[] } | { kind: 'repo'; url: string; groups: string[] }
type Start = { plan: Plan; title: string; bytes: number; owner: string; repo: string }

// 11 MB, 1.9 GB, 1.2 TB: shards and projectors are small next to the model
const gb = (n: number) => (n < 1e9 ? `${Math.max(1, Math.round(n / 1e6))} MB` : n < 1e12 ? `${(n / 1e9).toFixed(1)} GB` : `${(n / 1e12).toFixed(2)} TB`)
const sep = (folder: string) => (folder.includes('/') && !folder.includes('\\') ? '/' : '\\')
const driveOf = (path: string) => (/^[a-z]:/i.test(path) ? path.slice(0, 2).toUpperCase() + '\\' : '')

function useDuration() {
  const tr = useTr()
  return (s: number) => (s < 90 ? tr('dl.time.s', { n: Math.max(1, Math.round(s)) })
    : s < 90 * 60 ? tr('dl.time.min', { n: Math.round(s / 60) })
    : tr('dl.time.h', { h: Math.floor(s / 3600), m: Math.round((s % 3600) / 60) }))
}

export default function DownloadsView() {
  const tr = useTr()
  const loadLibrary = useModelStore(s => s.loadCatalog)
  const [cat, setCat] = useState<DlCatalog>()
  const [error, setError] = useState<unknown>('')
  const [start, setStart] = useState<Start | null>(null)

  const refresh = useCallback(async () => {
    try { setCat(await request<DlCatalog>('dl_catalog')) } catch (e) { setError(String(e)) }
  }, [])
  useEffect(() => { void refresh() }, [refresh])

  // while a download runs: its progress every second; when it ends, the files and the Library change
  const active = cat?.active
  useEffect(() => {
    if (!active) return
    const timer = setInterval(async () => {
      try {
        const s = await request<{ jobs: Job[]; active: string | null }>('dl_status')
        setCat(c => (c ? { ...c, ...s } : c))
        if (!s.active) {
          const job = s.jobs.find(j => j.id === active)
          if (job?.status === 'done') toast.success(tr('dl.finished', { name: job.title }))
          await refresh()
          await loadLibrary()
        }
      } catch (e) { setError(String(e)) }
    }, 1000)
    return () => clearInterval(timer)
  }, [active, refresh, loadLibrary, tr])

  const act = async (op: string, data: object) => {
    setError('')
    try {
      await request(op, data)
      await refresh()
    } catch (e) { setError(String(e)) }
  }

  return (
    <>
      {error ? <Notice tone="error">{describeError(error, tr)}</Notice> : null}
      {!!cat?.jobs.length && (
        <section>
          <Label count={cat.jobs.length}>{tr('dl.jobs')}</Label>
          <div className="divide-y divide-border rounded-lg border bg-card">
            {cat.jobs.slice(0, 5).map(j => <JobRow key={j.id} job={j} busy={!!cat.active && cat.active !== j.id} onAct={act} />)}
          </div>
        </section>
      )}

      <BrowseSection cat={cat} onDownload={setStart} />

      {cat && start && (
        <StartDialog start={start} catalog={cat} onClose={() => setStart(null)}
          onStarted={async () => { setStart(null); await refresh() }} />
      )}
    </>
  )
}

function JobRow({ job, busy, onAct }: { job: Job; busy: boolean; onAct: (op: string, data: object) => Promise<void> }) {
  const tr = useTr()
  const duration = useDuration()
  const { catalog: library, select, load } = useModelStore()
  const [files, setFiles] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const pct = job.total ? Math.min(100, (job.bytes / job.total) * 100) : 0
  const running = job.alive
  const icon = running ? <LoaderCircle className="size-4 shrink-0 animate-spin text-muted-foreground" />
    : job.status === 'done' ? <CheckCircle2 className="size-4 shrink-0 text-foreground" />
    : job.status === 'failed' ? <CircleAlert className="size-4 shrink-0 text-destructive" />
    : <CirclePause className="size-4 shrink-0 text-muted-foreground" />
  const state = running ? (job.files.some(f => f.status === 'verifying') ? 'verifying' : job.status === 'starting' ? 'starting' : 'running') : job.status
  const failed = job.error ? tr(`dl.err.${job.error.code}`, { defaultValue: tr('dl.err.other', { code: job.error.code }), ...job.error.params,
    file: job.error.params.file ?? '', reason: job.error.params.reason ?? '' }) : ''
  // the model this download completed, as the Library lists it now
  const first = job.files[0]?.name
  const ready = job.status === 'done' ? library?.models.find(m => m.role === 'model' && m.filename === first) : undefined
  const meta = [tr('dl.progress', { done: gb(job.bytes), total: gb(job.total) }),
    running && job.speed > 0 ? tr('dl.speed', { speed: (job.speed / 1e6).toFixed(1) }) : null,
    running && job.eta ? tr('dl.eta', { time: duration(job.eta) }) : null].filter(Boolean).join(' · ')

  return (
    <div className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        {icon}
        <div className="min-w-48 flex-1">
          <div className="truncate text-[13px] font-medium text-foreground" title={job.title}>{job.title}</div>
          <div className="mt-0.5 truncate text-xs text-muted-foreground" title={job.folder}>
            {tr(`dl.status.${state}`)} · {meta}
          </div>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-4">
          {running && <RowAction onClick={() => void onAct('dl_pause', { id: job.id })}>{tr('dl.pause')}</RowAction>}
          {!running && (job.status === 'stopped' || job.status === 'failed') &&
            <RowAction disabled={busy} onClick={() => void onAct('dl_resume', { id: job.id })}>{tr(job.status === 'failed' ? 'dl.retry' : 'dl.resume')}</RowAction>}
          {ready && <RowAction onClick={() => void select(ready.id).then(() => load(ready.id))}>{tr('dl.loadNow')}</RowAction>}
          {job.status === 'done' && <RowAction onClick={() => void revealItemInDir(job.folder).catch(() => openExternal(job.folder))}>{tr('dl.showFolder')}</RowAction>}
          <RowAction onClick={() => setFiles(!files)}>{tr(files ? 'dl.hideFiles' : 'dl.files')}</RowAction>
          {job.status === 'done'
            ? <RowAction onClick={() => void onAct('dl_cancel', { id: job.id, delete_parts: false })}>{tr('dl.remove')}</RowAction>
            : <RowAction onClick={() => setConfirm(true)}>{tr('dl.cancel')}</RowAction>}
        </div>
      </div>
      {job.status !== 'done' && <Progress value={pct} className="mt-2 h-1.5" aria-label={tr('dl.progressLabel')} />}
      {failed && <p className="mt-2 text-xs text-destructive">{failed}</p>}
      {files && (
        <div className="mt-2 space-y-1 rounded-md border px-3 py-2">
          {job.files.map(f => (
            <div key={f.name} className="flex flex-wrap items-center gap-x-3 text-xs">
              <span className="min-w-0 flex-1 truncate font-mono" title={f.name}>{f.name}</span>
              <span className="text-muted-foreground">{f.status === 'done' ? gb(f.size) : `${gb(f.bytes)} / ${gb(f.size)}`}</span>
              <span className="w-28 text-right text-muted-foreground">
                {f.status === 'done' ? tr(f.verified ? 'dl.verified' : 'dl.sizeOnly') : tr(`dl.status.${f.status === 'downloading' && !running ? 'stopped' : f.status}`, { defaultValue: f.status })}
              </span>
            </div>
          ))}
        </div>
      )}
      <Dialog open={confirm} onOpenChange={setConfirm}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-[15px]">{tr('dl.cancelTitle')}</DialogTitle>
            <DialogDescription className="text-xs">{tr('dl.cancelBody')}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button size="sm" variant="outline" className="h-8" onClick={() => setConfirm(false)}>{tr('dl.keep')}</Button>
            <Button size="sm" variant="destructive" className="h-8" onClick={() => { setConfirm(false); void onAct('dl_cancel', { id: job.id, delete_parts: true }) }}>{tr('dl.confirmCancel')}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function FitNote({ fits }: { fits: boolean | null }) {
  const tr = useTr()
  if (fits === null) return null
  return fits
    ? <span className="inline-flex items-center gap-1 text-xs text-muted-foreground"><CheckCircle2 className="size-3.5" />{tr('dl.fits')}</span>
    : <span className="inline-flex items-center gap-1 text-xs text-destructive"><CircleAlert className="size-3.5" />{tr('dl.tooLarge')}</span>
}

function ModelCard({ model, running, onDownload }: { model: DlModel; running: boolean; onDownload: (include: string[], bytes: number) => void }) {
  const tr = useTr()
  const optional = model.files.filter(f => !f.required).map(f => f.role)
  const [include, setInclude] = useState<string[]>(optional)
  const wanted = model.files.filter(f => f.required || include.includes(f.role))
  const bytes = wanted.filter(f => !f.present).reduce((n, f) => n + f.size, 0)
  const shards = model.files.filter(f => f.role === 'model').length
  let shard = 0
  return (
    <div className="mb-3 rounded-lg border bg-card px-5 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-semibold text-foreground">{model.name}</span>
        <span className="rounded border px-1.5 font-mono text-[11px] leading-5 text-muted-foreground">{model.variant}</span>
        {model.capabilities.map(c => <span key={c} className="rounded border px-1.5 text-[11px] leading-5 text-muted-foreground">{tr(`dl.caps.${c}`, { defaultValue: c })}</span>)}
        <span className="ml-auto"><FitNote fits={model.fits} /></span>
      </div>
      <div className="mt-1 text-xs text-muted-foreground">
        {model.owner}/{model.repo}
        {model.license && <> · <button type="button" className="text-[var(--rl-blue)] hover:underline" onClick={() => openExternal(model.license)}>{tr('dl.license')}</button></>}
      </div>
      <div className="mt-3 divide-y divide-border rounded-md border">
        {model.files.map(f => {
          if (f.role === 'model') shard += 1
          const on = f.required || include.includes(f.role)
          return (
            <label key={f.name} className={cn('flex items-center gap-3 px-3 py-2 text-xs', !f.required && 'cursor-pointer')}>
              <input type="checkbox" className="size-3.5 accent-[var(--primary)]" checked={on} disabled={f.required || running}
                onChange={e => setInclude(e.target.checked ? [...include, f.role] : include.filter(r => r !== f.role))} />
              <span className="min-w-0 flex-1">
                <span className="text-foreground">{f.role === 'model' && shards > 1 ? tr('dl.shard', { n: shard, of: shards }) : tr(`dl.role.${f.role}`)}</span>
                <span className="ml-2 font-mono text-muted-foreground">{f.name}</span>
              </span>
              <span className="text-muted-foreground">{gb(f.size)}</span>
              <span className="w-24 text-right text-muted-foreground">{f.present ? tr('dl.present') : f.required ? tr('dl.required') : ''}</span>
            </label>
          )
        })}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        {bytes > 0 ? <>
          <Button size="sm" className="h-8" disabled={running} onClick={() => onDownload(include, bytes)}><Download className="size-3.5" />{tr('dl.download')}</Button>
          <span className="text-xs text-muted-foreground">{tr('dl.toDownload', { size: gb(bytes) })}</span>
        </> : <>
          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground"><CheckCircle2 className="size-3.5" />{tr('dl.allPresent')}</span>
          <Link to="/strixllama/models" className="text-xs text-[var(--rl-blue)] hover:underline">{tr('tabs.models')}</Link>
        </>}
      </div>
    </div>
  )
}

function BrowseSection({ cat, onDownload }: { cat: DlCatalog | undefined; onDownload: (s: Start) => void }) {
  const tr = useTr()
  const { i18n } = useTranslation()
  const [query, setQuery] = useState('')
  const [source, setSource] = useState(i18n.language?.startsWith('zh') ? 'hfm' : 'hf')
  const [results, setResults] = useState<SearchHit[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>('')
  const [pick, setPick] = useState<Pick | null>(null)
  const running = !!cat?.active

  // the first recommended model is shown until something else is picked
  const shown: Pick | null = pick || (cat?.models.length ? { kind: 'catalog', id: cat.models[0].id } : null)

  const search = async () => {
    const q = query.trim()
    if (!q) { setResults(null); return }
    setError('')
    // a repository link or owner/repo opens that repository directly
    const link = q.match(/^https?:\/\/(?:huggingface\.co|hf\.co|hf-mirror\.com)\/(?!api\/|datasets\/|spaces\/)([\w.-]+\/[\w.-]+)/)
    if (link || /^[\w.-]+\/[\w.-]+$/.test(q)) {
      if (link?.[0].includes('hf-mirror.com')) setSource('hfm')
      setPick({ kind: 'repo', id: (link ? link[1] : q).replace(/\.git$/, '') })
      return
    }
    setBusy(true)
    try { setResults((await request<{ results: SearchHit[] }>('dl_search', { query: q, source })).results) }
    catch (e) { setError(String(e)); setResults(null) } finally { setBusy(false) }
  }

  return (
    <section>
      <Label>{tr('dl.browse')}</Label>
      <div className="flex min-h-[560px] overflow-hidden rounded-lg border bg-card">
        <div className="flex w-[340px] shrink-0 flex-col border-r">
          <div className="space-y-2 border-b p-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input aria-label={tr('dl.searchLabel')} className="h-8 pl-8 text-xs" placeholder={tr('dl.searchPlaceholder')} value={query}
                onChange={e => { setQuery(e.target.value); if (!e.target.value.trim()) setResults(null) }}
                onKeyDown={e => { if (e.key === 'Enter') void search() }} />
              {busy && <LoaderCircle className="absolute right-3 top-1/2 size-3.5 -translate-y-1/2 animate-spin text-muted-foreground" />}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>{tr('dl.listFrom')}</span>
              <Segments ariaLabel={tr('dl.start.source')} value={source} onChange={setSource}
                options={[{ value: 'hf', label: 'Hugging Face' }, { value: 'hfm', label: 'hf-mirror.com' }]} />
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto" role="listbox" aria-label={tr('dl.searchLabel')}>
            {error ? <Notice tone="error" className="m-3">{describeError(error, tr)}</Notice> : null}
            {results === null ? (
              <>
                <div className="px-3 pb-1 pt-3 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('dl.picks')}</div>
                {!cat ? <div className="px-3 py-6 text-center text-xs text-muted-foreground">{tr('dl.reading')}</div>
                  : cat.models.map(m => (
                    <ListRow key={m.id} active={shown?.kind === 'catalog' && shown.id === m.id} onClick={() => setPick({ kind: 'catalog', id: m.id })}
                      title={`${m.name} · ${m.variant}`} sub={`${m.owner}/${m.repo}`}
                      badge={<span className="rounded border border-[var(--rl-blue)] px-1.5 text-[10px] leading-4 text-[var(--rl-blue)]">{tr('dl.tested')}</span>} />
                  ))}
                <p className="px-3 py-3 text-xs text-muted-foreground">{tr('dl.searchHint')}</p>
              </>
            ) : results.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-muted-foreground">{tr('dl.noResults')}</div>
            ) : (
              <>
                <div className="px-3 pb-1 pt-3 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('dl.results', { count: results.length })}</div>
                {results.map(r => (
                  <ListRow key={r.id} active={shown?.kind === 'repo' && shown.id === r.id} onClick={() => setPick({ kind: 'repo', id: r.id })}
                    title={r.repo} sub={r.owner}
                    meta={<>{count(r.downloads)} ↓ · {count(r.likes)} ♥{r.updated ? <> · {r.updated.slice(0, 10)}</> : null}</>}
                    badge={r.architecture ? <span className="rounded border px-1.5 font-mono text-[10px] leading-4 text-muted-foreground">{r.architecture}</span> : null} />
                ))}
              </>
            )}
          </div>
        </div>
        <div className="min-w-0 flex-1 overflow-y-auto">
          {!shown ? <div className="px-6 py-10 text-center text-xs text-muted-foreground">{tr('dl.pickHint')}</div>
            : shown.kind === 'catalog'
              ? (() => {
                  const m = cat?.models.find(x => x.id === shown.id)
                  return m ? <PickDetail key={m.id} model={m} running={running} onDownload={onDownload} source={source} /> : null
                })()
              : <RepoDetail key={`${shown.id}@${source}`} id={shown.id} source={source} running={running} onDownload={onDownload} />}
        </div>
      </div>
    </section>
  )
}

function ListRow({ active, onClick, title, sub, meta, badge }: { active: boolean; onClick: () => void; title: string; sub: string; meta?: ReactNode; badge?: ReactNode }) {
  return (
    <button type="button" role="option" aria-selected={active} onClick={onClick}
      className={cn('block w-full border-l-2 px-3 py-2.5 text-left transition-colors',
        active ? 'border-l-[var(--rl-blue)] bg-[rgba(127,127,127,0.2)]' : 'border-l-transparent hover:bg-[rgba(127,127,127,0.1)]')}>
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-foreground">{title}</span>
        {badge}
      </div>
      <div className="mt-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
        <span className="min-w-0 truncate">{sub}</span>
        {meta && <span className="ml-auto shrink-0 tabular-nums">{meta}</span>}
      </div>
    </button>
  )
}

// a recommended model: its tested file set, then the repository's model card
function PickDetail({ model, running, onDownload, source }: { model: DlModel; running: boolean; onDownload: (s: Start) => void; source: string }) {
  return (
    <div className="px-6 py-5">
      <ModelCard model={model} running={running}
        onDownload={(include, bytes) => onDownload({ plan: { kind: 'catalog', model: model.id, include }, title: `${model.name} (${model.variant})`, bytes, owner: model.owner, repo: model.repo })} />
      {model.commit && <Readme owner={model.owner} repo={model.repo} commit={model.commit} source={source} />}
    </div>
  )
}

function RepoDetail({ id, source, running, onDownload }: { id: string; source: string; running: boolean; onDownload: (s: Start) => void }) {
  const tr = useTr()
  const [listing, setListing] = useState<RepoListing>()
  const [error, setError] = useState<unknown>('')
  const [model, setModel] = useState('')
  const [extras, setExtras] = useState<string[]>([])

  useEffect(() => {
    let live = true
    request<RepoListing>('dl_repo', { url: id, source }).then(l => {
      if (!live) return
      setListing(l)
      const models = l.groups.filter(g => g.role === 'model' && g.complete)
      const first = models.find(g => g.tuned) || models[0]
      setModel(first?.key || '')
      setExtras(companionsFor(first, l.groups))
    }).catch(e => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [id, source])

  if (error) return <div className="px-6 py-5"><Notice tone="error">{describeError(error, tr)}</Notice></div>
  if (!listing) return <div className="flex items-center justify-center gap-2 px-6 py-10 text-xs text-muted-foreground"><LoaderCircle className="size-3.5 animate-spin" />{tr('dl.loadingRepo')}</div>

  const models = listing.groups.filter(g => g.role === 'model')
  const others = listing.groups.filter(g => g.role !== 'model')
  const current = models.find(g => g.key === model)
  const chosen = listing.groups.filter(g => g.key === model || extras.includes(g.key))
  const bytes = chosen.flatMap(g => g.files).filter(f => !f.present).reduce((n, f) => n + f.size, 0)
  const tested = models.some(g => g.tuned)
  const choose = (key: string) => { setModel(key); setExtras(companionsFor(models.find(g => g.key === key), listing.groups)) }

  return (
    <div className="px-6 py-5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[15px] font-semibold text-foreground">{listing.owner}/{listing.repo}</span>
        <button type="button" className="text-xs text-[var(--rl-blue)] hover:underline" onClick={() => openExternal(`https://huggingface.co/${listing.owner}/${listing.repo}`)}>{tr('dl.openOnHf')}</button>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span className="font-mono">{tr('dl.commit', { commit: listing.commit.slice(0, 7) })}</span>
        {others.some(g => g.role === 'projection') && <span className="rounded border px-1.5 text-[11px] leading-5">{tr('dl.caps.images')}</span>}
        {others.some(g => g.role === 'draft') && <span className="rounded border px-1.5 text-[11px] leading-5">{tr('dl.caps.mtp')}</span>}
        {!tested && <span className="inline-flex items-center gap-1"><CircleAlert className="size-3.5" />{tr('dl.notTested')}</span>}
      </div>

      {models.length > 0 && (
        <>
          <div className="mb-1.5 mt-4 text-xs font-medium text-foreground">{tr('dl.options')}</div>
          <div className="divide-y divide-border rounded-md border" role="radiogroup" aria-label={tr('dl.options')}>
            {models.map(g => (
              <label key={g.key} className={cn('flex items-center gap-3 px-3 py-2 text-xs', g.complete ? 'cursor-pointer' : 'opacity-60')}>
                <input type="radio" name="repo-model" className="size-3.5 accent-[var(--primary)]" checked={model === g.key} disabled={!g.complete} onChange={() => choose(g.key)} />
                <span className="min-w-0 flex-1 truncate">
                  <span className="text-foreground">{g.quant || g.name}</span>
                  {g.shards > 1 && <span className="ml-2 text-muted-foreground">{tr('dl.shards', { count: g.shards })}</span>}
                  {g.tuned && <span className="ml-2 rounded border border-[var(--rl-blue)] px-1.5 text-[10px] leading-4 text-[var(--rl-blue)]" title={tr('dl.tunedHint')}>{tr('dl.tuned')}</span>}
                </span>
                {!g.complete && <span className="text-destructive">{tr('dl.incomplete')}</span>}
                {g.files.length > 0 && g.files.every(f => f.present) && <span className="text-muted-foreground">{tr('dl.present')}</span>}
                <FitNote fits={g.fits} />
                <span className="w-20 text-right tabular-nums text-muted-foreground">{gb(g.size)}</span>
              </label>
            ))}
          </div>
        </>
      )}
      {others.length > 0 && (
        <>
          <div className="mb-1.5 mt-4 text-xs font-medium text-foreground">{tr('dl.companions')}</div>
          <div className="divide-y divide-border rounded-md border">
            {others.map(g => (
              <label key={g.key} className="flex cursor-pointer items-center gap-3 px-3 py-2 text-xs">
                <input type="checkbox" className="size-3.5 accent-[var(--primary)]" checked={extras.includes(g.key)}
                  onChange={e => setExtras(e.target.checked ? [...extras, g.key] : extras.filter(k => k !== g.key))} />
                <span className="min-w-0 flex-1 truncate"><span className="text-foreground">{tr(`dl.role.${g.role}`)}</span><span className="ml-2 font-mono text-muted-foreground">{g.name}</span></span>
                {g.files.every(f => f.present) && <span className="text-muted-foreground">{tr('dl.present')}</span>}
                <span className="w-20 text-right tabular-nums text-muted-foreground">{gb(g.size)}</span>
              </label>
            ))}
          </div>
        </>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Button size="sm" className="h-8" disabled={running || !current || bytes === 0}
          onClick={() => onDownload({ plan: { kind: 'repo', url: `${listing.owner}/${listing.repo}`, groups: chosen.map(g => g.key) }, title: `${listing.owner}/${listing.repo}`, bytes, owner: listing.owner, repo: listing.repo })}>
          <Download className="size-3.5" />{tr('dl.download')}
        </Button>
        {current && <span className="text-xs text-muted-foreground">{bytes ? tr('dl.toDownload', { size: gb(bytes) }) : tr('dl.allPresent')}</span>}
      </div>
      <Readme owner={listing.owner} repo={listing.repo} commit={listing.commit} source={source} />
    </div>
  )
}

// the files that go with a model: its vision projector, and an MTP draft of the right kind (an EXL3 model takes the
// draft with its own head, the others never do)
function companionsFor(model: RepoGroup | undefined, groups: RepoGroup[]) {
  if (!model) return []
  const exl3 = model.quant.startsWith('EXL3')
  return groups.filter(g => g.role === 'projection' || (g.role === 'draft' && g.name.toLowerCase().includes('ownhead') === exl3)).map(g => g.key)
}

function Readme({ owner, repo, commit, source }: { owner: string; repo: string; commit: string; source: string }) {
  const tr = useTr()
  const [text, setText] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    setText(null)
    request<{ text: string }>('dl_readme', { owner, repo, commit, source })
      .then(r => { if (live) setText(r.text) }).catch(() => { if (live) setText('') })
    return () => { live = false }
  }, [owner, repo, commit, source])
  return (
    <div className="mt-6 border-t pt-4">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('dl.readme')}</div>
      {text === null ? <div className="text-xs text-muted-foreground">{tr('dl.reading')}</div>
        : text ? <div className="text-[13px]"><RenderMarkdown content={text} /></div>
        : <div className="text-xs text-muted-foreground">{tr('dl.readmeNone')}</div>}
    </div>
  )
}

const count = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n))

function StartDialog({ start, catalog, onClose, onStarted }: { start: Start; catalog: DlCatalog; onClose: () => void; onStarted: () => Promise<void> }) {
  const tr = useTr()
  const { i18n } = useTranslation()
  const serviceHub = useServiceHub()
  const root = catalog.default_root || ''
  const [folder, setFolder] = useState(root ? [root, start.owner, start.repo].join(sep(root)) : '')
  const [source, setSource] = useState(i18n.language?.startsWith('zh') ? 'hfm' : 'hf')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>('')
  const drive = catalog.drives.find(d => d.drive.toUpperCase() === driveOf(folder))
  const short = !!drive && drive.free < start.bytes

  const browse = async () => {
    const chosen = await serviceHub.dialog().open({ multiple: false, directory: true })
    if (typeof chosen === 'string') setFolder(chosen)
  }
  const go = async () => {
    setBusy(true); setError('')
    try {
      await request('dl_start', { ...start.plan, source, folder: folder === [root, start.owner, start.repo].join(sep(root)) ? '' : folder })
      await onStarted()
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }

  return (
    <Dialog open onOpenChange={o => { if (!o) onClose() }}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="text-[15px]">{tr('dl.start.title', { name: start.title })}</DialogTitle>
          <DialogDescription className="text-xs">{tr('dl.toDownload', { size: gb(start.bytes) })}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <div className="mb-1.5 text-xs font-medium text-foreground">{tr('dl.start.source')}</div>
            <Segments ariaLabel={tr('dl.start.source')} value={source} onChange={setSource}
              options={[{ value: 'hfm', label: 'hf-mirror.com', hint: tr('dl.start.china') }, { value: 'hf', label: 'Hugging Face' }]} />
          </div>
          <div>
            <div className="mb-1.5 text-xs font-medium text-foreground">{tr('dl.start.folder')}</div>
            <div className="flex gap-2">
              <Input className="h-8 font-mono text-xs" value={folder} onChange={e => setFolder(e.target.value)} aria-label={tr('dl.start.folder')} />
              <Button size="sm" variant="outline" className="h-8" onClick={() => void browse()}><FolderOpen className="size-3.5" />{tr('dl.start.browse')}</Button>
            </div>
            {drive && <p className={cn('mt-1 text-xs', short ? 'text-destructive' : 'text-muted-foreground')}>{tr('dl.start.free', { free: gb(drive.free), drive: drive.drive })}</p>}
          </div>
          <p className="text-xs text-muted-foreground">{tr('dl.start.note')}</p>
          {error ? <Notice tone="error">{describeError(error, tr)}</Notice> : null}
        </div>
        <DialogFooter>
          <Button size="sm" variant="outline" className="h-8" onClick={onClose}>{tr('actions.cancel')}</Button>
          <Button size="sm" className="h-8" disabled={busy || !folder.trim()} onClick={() => void go()}>
            {busy ? <LoaderCircle className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}{tr('dl.start.go')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
