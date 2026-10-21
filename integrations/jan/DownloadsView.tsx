import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { revealItemInDir } from '@tauri-apps/plugin-opener'
import { CheckCircle2, ChevronDown, ChevronRight, CircleAlert, CirclePause, Download, FolderOpen, Heart, LoaderCircle, Search } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Switch } from '@/components/ui/switch'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useServiceHub } from '@/hooks/useServiceHub'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { RenderMarkdown } from '@/containers/RenderMarkdown'
import { cn } from '@/lib/utils'
import { describeError, request } from './status'
import { useModelStore } from './store'
import { Label, Notice, RowAction, Segments, openExternal, useTr } from './parts'

// Model downloads (tools/downloader.py, the manager's dl_* ops), laid out as a model search: the models this build is
// tested with and any GGUF repository found on Hugging Face (or opened by its link) on the left; on the right the
// chosen model with its download options, one button, the files that go with it, its model card and more from its
// owner; the downloads below. One download runs at a time, in a worker process that outlives this page and the app;
// the page polls it while it runs.

type DlFile = { name: string; role: string; required: boolean; size: number; present: string | null }
type DlModel = { id: string; name: string; variant: string; owner: string; repo: string; license: string; capabilities: string[]; files: DlFile[]; missing: number; fits: boolean | null; commit?: string }
type JobFile = { name: string; size: number; bytes: number; status: string; verified?: boolean | null; source: number }
type JobError = { code: string; params: Record<string, unknown> }
export type Job = { id: string; kind: string; title: string; model: string | null; folder: string; source: string; created_at: string; status: string; alive: boolean; error?: JobError | null; files: JobFile[]; bytes: number; total: number; speed: number; eta: number | null }
type Drive = { drive: string; free: number; total: number }
type DlCatalog = { models: DlModel[]; sources: Record<string, string>; drives: Drive[]; default_root: string | null; jobs: Job[]; active: string | null }
type RepoFile = { name: string; path: string; size: number; sha256: string | null; present: string | null }
type RepoGroup = { key: string; role: string; name: string; quant: string; size: number; complete: boolean; shards: number; files: RepoFile[]; fits: boolean | null; tuned?: boolean }
type SearchHit = { id: string; owner: string; repo: string; downloads: number; likes: number; updated: string | null; task: string | null; architecture: string | null; params?: number | null }
// a repository's numbers, from the hub (downloader.info_of)
type RepoInfo = { downloads: number; likes: number; updated: string | null; task: string | null; license: string | null; base_model: string | null; architecture: string | null; params: number | null; context: number | null }
// what the right-hand pane shows: a recommended model, or a repository from the search
type Pick = { kind: 'catalog'; id: string } | { kind: 'repo'; id: string; path?: string }
type RepoListing = { source: string; owner: string; repo: string; commit: string; groups: RepoGroup[]; folder: string | null; info?: RepoInfo }
// what the start dialog sends to dl_start, less the source and folder it asks for
type Plan = { kind: 'catalog'; model: string; include: string[] } | { kind: 'repo'; url: string; groups: string[] }
type Start = { plan: Plan; title: string; bytes: number; owner: string; repo: string }
type Source = { source: string; setSource: (s: string) => void }

// 11 MB, 1.9 GB, 1.2 TB: shards and projectors are small next to the model
const gb = (n: number) => (n < 1e9 ? `${Math.max(1, Math.round(n / 1e6))} MB` : n < 1e12 ? `${(n / 1e9).toFixed(1)} GB` : `${(n / 1e12).toFixed(2)} TB`)
const sep = (folder: string) => (folder.includes('/') && !folder.includes('\\') ? '/' : '\\')
const driveOf = (path: string) => (/^[a-z]:/i.test(path) ? path.slice(0, 2).toUpperCase() + '\\' : '')
const count = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n))
// 177B, 9.5B, 600M parameters; 256K context
const params = (n: number) => (n >= 1e10 ? `${Math.round(n / 1e9)}B` : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${Math.round(n / 1e6)}M`)
const tokens = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}K` : String(n))
const SOURCE_KEY = 'rulith-dl-source'
const BADGE = 'rounded border px-1.5 text-[10px] leading-4'
const BLUE_BADGE = cn(BADGE, 'border-[var(--rl-blue)] text-[var(--rl-blue)]')

function useDuration() {
  const tr = useTr()
  return (s: number) => (s < 90 ? tr('dl.time.s', { n: Math.max(1, Math.round(s)) })
    : s < 90 * 60 ? tr('dl.time.min', { n: Math.round(s / 60) })
    : tr('dl.time.h', { h: Math.floor(s / 3600), m: Math.round((s % 3600) / 60) }))
}

export default function DownloadsView() {
  const tr = useTr()
  const { i18n } = useTranslation()
  const loadLibrary = useModelStore(s => s.loadCatalog)
  const [cat, setCat] = useState<DlCatalog>()
  const [error, setError] = useState<unknown>('')
  const [start, setStart] = useState<Start | null>(null)
  // where lists, model cards and downloads come from: the source last used (or the one that answered when the other
  // could not be reached), else by language. Not a control on the page: the download dialog offers the choice.
  const [source, setSourceState] = useState(() => localStorage.getItem(SOURCE_KEY) || (i18n.language?.startsWith('zh') ? 'hfm' : 'hf'))
  const setSource = useCallback((s: string) => { setSourceState(s); localStorage.setItem(SOURCE_KEY, s) }, [])
  // the downloads panel: open while something is unfinished, unless the user closed it
  const [jobsOpen, setJobsOpen] = useState<boolean | null>(null)

  const refresh = useCallback(async () => {
    try { setCat(await request<DlCatalog>('dl_catalog')) } catch (e) { setError(String(e)) }
  }, [])
  useEffect(() => { void refresh() }, [refresh])

  // while a download runs: its progress every second; when it ends, the files and the Library change
  const active = cat?.active
  useEffect(() => {
    if (!active) return
    // one tick at a time: a manager call can take longer than the period, and ticks that overlapped all saw the
    // job end and raised its toast several times
    let busy = false, ended = false
    const timer = setInterval(async () => {
      if (busy || ended) return
      busy = true
      try {
        const s = await request<{ jobs: Job[]; active: string | null }>('dl_status')
        setCat(c => (c ? { ...c, ...s } : c))
        if (!s.active) {
          ended = true
          clearInterval(timer)
          const job = s.jobs.find(j => j.id === active)
          if (job?.status === 'done') toast.success(tr('dl.finished', { name: job.title }))
          await refresh()
          await loadLibrary()
        }
      } catch (e) { setError(String(e)) } finally { busy = false }
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

  const jobs = cat?.jobs || []
  const open = jobsOpen ?? jobs.some(j => j.status !== 'done')
  return (
    <>
      {error ? <Notice tone="error">{describeError(error, tr)}</Notice> : null}
      <BrowseSection cat={cat} onDownload={setStart} source={source} setSource={setSource} />

      {jobs.length > 0 && (
        <section>
          <Label count={jobs.length} action={
            <button type="button" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground" aria-expanded={open}
              onClick={() => setJobsOpen(!open)}>
              {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}{tr(open ? 'dl.hideJobs' : 'dl.showJobs')}
            </button>}>{tr('dl.jobs')}</Label>
          {open && (
            <div className="divide-y divide-border rounded-lg border bg-card">
              {jobs.slice(0, 5).map(j => <JobRow key={j.id} job={j} busy={!!cat?.active && cat.active !== j.id} onAct={act} />)}
            </div>
          )}
        </section>
      )}

      {cat && start && (
        <StartDialog start={start} catalog={cat} source={source} setSource={setSource} onClose={() => setStart(null)}
          onStarted={async () => { setStart(null); setJobsOpen(true); await refresh() }} />
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
  // the action on its way: stopping the worker can take seconds, the row says so meanwhile
  const [working, setWorking] = useState('')
  const run = (op: string, data: object) => { setWorking(op); void onAct(op, data).finally(() => setWorking('')) }
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
            {working ? tr(`dl.working.${working}`) : tr(`dl.status.${state}`)} · {meta}
          </div>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-4">
          {running && <RowAction disabled={!!working} onClick={() => run('dl_pause', { id: job.id })}>{tr('dl.pause')}</RowAction>}
          {!running && (job.status === 'stopped' || job.status === 'failed') &&
            <RowAction disabled={busy || !!working} onClick={() => run('dl_resume', { id: job.id })}>{tr(job.status === 'failed' ? 'dl.retry' : 'dl.resume')}</RowAction>}
          {ready && <RowAction onClick={() => void select(ready.id).then(() => load(ready.id))}>{tr('dl.loadNow')}</RowAction>}
          {job.status === 'done' && <RowAction onClick={() => void revealItemInDir(job.folder).catch(() => openExternal(job.folder))}>{tr('dl.showFolder')}</RowAction>}
          <RowAction onClick={() => setFiles(!files)}>{tr(files ? 'dl.hideFiles' : 'dl.files')}</RowAction>
          {job.status === 'done'
            ? <RowAction disabled={!!working} onClick={() => run('dl_cancel', { id: job.id, delete_parts: false })}>{tr('dl.remove')}</RowAction>
            : <RowAction disabled={!!working} onClick={() => setConfirm(true)}>{tr('dl.cancel')}</RowAction>}
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
            <Button size="sm" variant="destructive" className="h-8" onClick={() => { setConfirm(false); run('dl_cancel', { id: job.id, delete_parts: true }) }}>{tr('dl.confirmCancel')}</Button>
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

function BrowseSection({ cat, onDownload, source, setSource }: { cat: DlCatalog | undefined; onDownload: (s: Start) => void } & Source) {
  const tr = useTr()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchHit[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>('')
  const [pick, setPick] = useState<Pick | null>(null)
  const running = !!cat?.active
  // finished downloads so far: the repository pane reads its files again when this changes (In the Library)
  const doneJobs = cat?.jobs.filter(j => j.status === 'done').length ?? 0

  // recommended models by model, their variants in catalog order (the first is the recommended one)
  const groups = groupModels(cat?.models || [])
  // the first recommended model is shown until something else is picked
  const shown: Pick | null = pick || (groups.length ? { kind: 'catalog', id: groups[0].key } : null)
  const openRepo = (id: string) => setPick({ kind: 'repo', id })

  const search = async () => {
    const q = query.trim()
    if (!q) { setResults(null); return }
    setError('')
    // a repository link or owner/repo opens that repository directly
    const link = q.match(/^https?:\/\/(?:huggingface\.co|hf\.co|hf-mirror\.com)\/(?!api\/|datasets\/|spaces\/)([\w.-]+\/[\w.-]+)(?:\/(?:tree|blob|resolve)\/[\w.-]+\/([^?#]+))?/)
    if (link || /^[\w.-]+\/[\w.-]+$/.test(q)) {
      if (link?.[0].includes('hf-mirror.com')) setSource('hfm')
      // a link into a folder or file (…/tree/main/UD-Q4_K_XL) preselects that download option
      setPick({ kind: 'repo', id: (link ? link[1] : q).replace(/\.git$/, ''), path: link?.[2] })
      return
    }
    setBusy(true)
    try {
      const r = await request<{ results: SearchHit[]; source: string }>('dl_search', { query: q, source })
      if (r.source && r.source !== source) setSource(r.source)
      setResults(r.results)
    }
    catch (e) { setError(String(e)); setResults(null) } finally { setBusy(false) }
  }

  return (
    <section>
      <Label>{tr('dl.browse')}</Label>
      {/* the window's height less the title bar, the page's padding and this label */}
      <div className="flex h-[calc(100vh-150px)] min-h-[520px] overflow-hidden rounded-lg border bg-card">
        <div className="flex w-[320px] shrink-0 flex-col border-r">
          <div className="border-b p-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input aria-label={tr('dl.searchLabel')} className="h-8 pl-8 text-xs" placeholder={tr('dl.searchPlaceholder')} value={query}
                onChange={e => { setQuery(e.target.value); if (!e.target.value.trim()) setResults(null) }}
                onKeyDown={e => { if (e.key === 'Enter') void search() }} />
              {busy && <LoaderCircle className="absolute right-3 top-1/2 size-3.5 -translate-y-1/2 animate-spin text-muted-foreground" />}
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto" role="listbox" aria-label={tr('dl.searchLabel')}>
            {error ? <Notice tone="error" className="m-3">{describeError(error, tr)}</Notice> : null}
            {results === null ? (
              <>
                <ListLabel>{tr('dl.picks')}</ListLabel>
                {!cat ? <div className="px-3 py-6 text-center text-xs text-muted-foreground">{tr('dl.reading')}</div>
                  : groups.map(g => (
                    <ListRow key={g.key} active={shown?.kind === 'catalog' && shown.id === g.key} onClick={() => setPick({ kind: 'catalog', id: g.key })}
                      title={g.name} sub={`${g.owner} · ${g.models.map(m => m.variant).join(' · ')}`}
                      badge={<span className={BLUE_BADGE}>{tr(g === groups[0] ? 'dl.recommendedOne' : 'dl.tested')}</span>} />
                  ))}
                <p className="px-3 py-3 text-xs text-muted-foreground">{tr('dl.searchHint')}</p>
              </>
            ) : results.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-muted-foreground">{tr('dl.noResults')}</div>
            ) : (
              <>
                <ListLabel>{tr('dl.results', { count: results.length })}</ListLabel>
                {results.map(r => (
                  <ListRow key={r.id} active={shown?.kind === 'repo' && shown.id === r.id} onClick={() => openRepo(r.id)}
                    title={r.repo} sub={r.owner}
                    meta={<>{count(r.downloads)} ↓ · {count(r.likes)} ♥{r.updated ? <> · {r.updated.slice(0, 10)}</> : null}</>}
                    badge={r.architecture ? <span className={cn(BADGE, 'font-mono text-muted-foreground')}>{r.architecture}</span> : null} />
                ))}
              </>
            )}
          </div>
        </div>
        <div className="min-w-0 flex-1 overflow-y-auto">
          {!shown ? <div className="px-6 py-10 text-center text-xs text-muted-foreground">{tr('dl.pickHint')}</div>
            : shown.kind === 'catalog'
              ? (() => {
                  const g = groups.find(x => x.key === shown.id)
                  return g ? <PickDetail key={g.key} group={g} top={g === groups[0]} running={running} onDownload={onDownload} source={source} setSource={setSource} onOpen={openRepo} /> : null
                })()
              : <RepoDetail key={`${shown.id}@${shown.path || ''}@${doneJobs}`} id={shown.id} path={shown.path} source={source} setSource={setSource}
                  running={running} onDownload={onDownload} onOpen={openRepo} />}
        </div>
      </div>
    </section>
  )
}

const ListLabel = ({ children }: { children: ReactNode }) =>
  <div className="px-3 pb-1 pt-3 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{children}</div>

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

// ---- the model pane: the same layout for a recommended model and a repository from the search ----

// one line of the "Download options" menu
type Opt = { key: string; label: string; sub?: string; size: number; present: boolean; complete: boolean; fits: boolean | null; recommended?: boolean; tuned?: boolean }
// a file that goes with the model, as a switch under the button
type Companion = { role: string; size: number; present: boolean; on: boolean; toggle: (on: boolean) => void }
// a line of the file list under "Files": checkable when it is optional
type FileLine = { key: string; label: string; name: string; size: number; present: boolean; on: boolean; toggle?: (on: boolean) => void }

type PaneProps = {
  title: string; owner: string; repo: string; info?: RepoInfo; caps: string[]; tested: boolean; license?: string
  options: Opt[]; option: string; onOption: (key: string) => void
  companions: Companion[]; files: FileLine[]; bytes: number; firstFile?: string
  running: boolean; onDownload: () => void; readme: { commit: string; source: string }; setSource: (s: string) => void; onOpen: (id: string) => void
}

function ModelPane(p: PaneProps) {
  const tr = useTr()
  const [filesOpen, setFilesOpen] = useState(false)
  const { catalog: library, select, load, loadCatalog } = useModelStore()
  const current = p.options.find(o => o.key === p.option)
  // every file here already: the model loads from the Library entry of its first shard
  const ready = p.firstFile ? library?.models.find(m => m.role === 'model' && m.filename === p.firstFile) : undefined
  // The page checks the disk; the Library lists what its last scan found. Files put there since (copied by hand,
  // or downloaded by another app) are on disk but not in the Library: scan it once, then they can be loaded.
  const [scanned, setScanned] = useState<string[]>([])
  const scanning = !!p.firstFile && p.bytes === 0 && !!library && !ready && !scanned.includes(p.firstFile)
  useEffect(() => {
    if (!scanning || !p.firstFile) return
    const file = p.firstFile
    void loadCatalog(true).finally(() => setScanned(s => [...s, file]))
  }, [scanning, p.firstFile, loadCatalog])
  const i = p.info
  const stats = i ? [
    i.downloads ? <span key="d" className="inline-flex items-center gap-1"><Download className="size-3" />{count(i.downloads)}</span> : null,
    i.likes ? <span key="l" className="inline-flex items-center gap-1"><Heart className="size-3" />{count(i.likes)}</span> : null,
    i.updated ? <span key="u">{tr('dl.updated', { date: i.updated.slice(0, 10) })}</span> : null,
    i.architecture ? <span key="a" className="font-mono">{i.architecture}</span> : null,
    i.params ? <span key="p">{tr('dl.params', { n: params(i.params) })}</span> : null,
    i.context ? <span key="c">{tr('dl.context', { n: tokens(i.context) })}</span> : null,
  ].filter(Boolean) : []

  return (
    <div className="px-6 py-5">
      {/* the header: name, owner, numbers, what it can do */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[17px] font-semibold text-foreground">{p.title}</h3>
        <button type="button" className="text-xs text-[var(--rl-blue)] hover:underline" onClick={() => openExternal(`https://huggingface.co/${p.owner}/${p.repo}`)}>
          {p.owner}/{p.repo} ↗
        </button>
      </div>
      {stats.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs tabular-nums text-muted-foreground">
          {stats.map((s, n) => <span key={n} className="inline-flex items-center gap-3">{n > 0 && <span aria-hidden>·</span>}{s}</span>)}
        </div>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-muted-foreground">
        {p.tested ? <span className={BLUE_BADGE}>{tr('dl.tested')}</span>
          : <span className="inline-flex items-center gap-1 text-xs"><CircleAlert className="size-3.5" />{tr('dl.notTested')}</span>}
        {p.caps.map(c => <span key={c} className={BADGE}>{tr(`dl.caps.${c}`, { defaultValue: c })}</span>)}
        {(p.license || i?.license) && (
          p.license
            ? <button type="button" className={cn(BADGE, 'hover:text-foreground')} onClick={() => openExternal(p.license!)}>{tr('dl.license')}</button>
            : <span className={BADGE}>{i!.license}</span>
        )}
      </div>

      {/* the download options, one per quantization / variant */}
      {p.options.length > 0 && (
        <>
          <div className="mb-1.5 mt-5 text-xs font-medium text-foreground">{tr('dl.options')}</div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button type="button" aria-label={tr('dl.options')}
                className="flex w-full items-center gap-3 rounded-md border border-input bg-[rgba(127,127,127,0.12)] px-3 py-2 text-left text-xs hover:bg-[rgba(127,127,127,0.2)]">
                {current ? <OptionLine o={current} /> : <span className="flex-1 text-muted-foreground">{tr('dl.pickOption')}</span>}
                <ChevronDown className="size-4 shrink-0 text-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-(--radix-dropdown-menu-trigger-width) max-h-80">
              <DropdownMenuRadioGroup value={p.option} onValueChange={p.onOption}>
                {p.options.map(o => (
                  <DropdownMenuRadioItem key={o.key} value={o.key} disabled={!o.complete} className="gap-3 text-xs">
                    <OptionLine o={o} />
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      )}

      {/* one button: what is missing, or load what is here */}
      <div className="mt-3 flex flex-wrap items-center gap-3">
        {p.bytes > 0 ? (
          <Button className="h-9 px-5" disabled={p.running || !current?.complete} onClick={p.onDownload}>
            <Download className="size-4" />{tr('dl.downloadSize', { size: gb(p.bytes) })}
          </Button>
        ) : (
          <Button className="h-9 px-5" disabled={!ready || p.running} onClick={() => ready && void select(ready.id).then(() => load(ready.id))}>
            {scanning ? <LoaderCircle className="size-4 animate-spin" /> : <CheckCircle2 className="size-4" />}{tr('dl.loadNow')}
          </Button>
        )}
        <span className="text-xs text-muted-foreground">
          {p.running && p.bytes > 0 ? tr('dl.busyOther') : p.bytes === 0 ? tr(scanning ? 'dl.addingToLibrary' : ready ? 'dl.allPresent' : 'dl.notInLibrary')
            : current ? <FitNote fits={current.fits} /> : null}
        </span>
      </div>
      {p.bytes === 0 && <p className="mt-2 text-xs text-muted-foreground">{tr(p.options.length > 1 ? 'dl.otherDownloads' : 'dl.otherDownloadsSearch')}</p>}

      {/* the files that go with it, as switches */}
      {p.companions.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
          {p.companions.map(c => (
            <label key={c.role} className="inline-flex cursor-pointer items-center gap-2 text-xs">
              <Switch checked={c.on} disabled={p.running} onCheckedChange={c.toggle} aria-label={tr(`dl.with.${c.role}`, { defaultValue: c.role })} />
              <span className="text-foreground">{tr(`dl.with.${c.role}`, { defaultValue: c.role })}</span>
              <span className="text-muted-foreground">{c.present ? tr('dl.present') : gb(c.size)}</span>
            </label>
          ))}
        </div>
      )}

      {/* every file, folded */}
      {p.files.length > 0 && (
        <div className="mt-4">
          <button type="button" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground" aria-expanded={filesOpen}
            onClick={() => setFilesOpen(!filesOpen)}>
            {filesOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}{tr('dl.filesCount', { count: p.files.length })}
          </button>
          {filesOpen && (
            <div className="mt-1.5 divide-y divide-border rounded-md border">
              {p.files.map(f => (
                <label key={f.key} className={cn('flex items-center gap-3 px-3 py-1.5 text-xs', f.toggle && 'cursor-pointer')}>
                  <input type="checkbox" className="size-3.5 accent-[var(--primary)]" checked={f.on} disabled={!f.toggle || p.running}
                    onChange={e => f.toggle?.(e.target.checked)} />
                  <span className="min-w-0 flex-1 truncate">
                    <span className="text-foreground">{f.label}</span>
                    <span className="ml-2 font-mono text-muted-foreground" title={f.name}>{f.name}</span>
                  </span>
                  <span className="text-muted-foreground">{gb(f.size)}</span>
                  <span className="w-24 text-right text-muted-foreground">{f.present ? tr('dl.present') : ''}</span>
                </label>
              ))}
            </div>
          )}
        </div>
      )}

      <Readme owner={p.owner} repo={p.repo} commit={p.readme.commit} source={p.readme.source} onSource={p.setSource} />
      <MoreFrom owner={p.owner} repo={p.repo} source={p.readme.source} onOpen={p.onOpen} />
    </div>
  )
}

function OptionLine({ o }: { o: Opt }) {
  const tr = useTr()
  return (
    <>
      <span className="min-w-0 flex-1 truncate">
        <span className="font-mono text-foreground">{o.label}</span>
        {o.recommended && <span className={cn(BLUE_BADGE, 'ml-2')}>{tr('dl.recommendedOne')}</span>}
        {o.tuned && <span className={cn(BLUE_BADGE, 'ml-2')} title={tr('dl.tunedHint')}>{tr('dl.tuned')}</span>}
        {o.sub && <span className="ml-2 text-muted-foreground">{o.sub}</span>}
      </span>
      {!o.complete ? <span className="text-destructive">{tr('dl.incomplete')}</span>
        : o.present ? <span className="text-muted-foreground">{tr('dl.present')}</span>
        : <FitNote fits={o.fits} />}
      <span className="w-16 shrink-0 text-right tabular-nums text-muted-foreground">{gb(o.size)}</span>
    </>
  )
}

// a recommended repository: its quantizations (UD-Q4_K_XL, UD-IQ4_XS, ...) as the download options, each with its tested
// file set. One entry per repository, as on Hugging Face: the same model from another provider is another entry.
type ModelGroup = { key: string; name: string; owner: string; models: DlModel[] }

function groupModels(models: DlModel[]): ModelGroup[] {
  const out: ModelGroup[] = []
  for (const m of models) {
    const key = `${m.owner}/${m.repo}`
    const g = out.find(x => x.key === key)
    if (g) g.models.push(m); else out.push({ key, name: m.name, owner: m.owner, models: [m] })
  }
  return out
}

function useRepoInfo(owner: string, repo: string, source: string, setSource: (s: string) => void) {
  const [info, setInfo] = useState<RepoInfo>()
  useEffect(() => {
    let live = true
    setInfo(undefined)
    // the numbers are a nicety: without them the pane shows the rest
    request<{ info: RepoInfo; source: string }>('dl_info', { owner, repo, source })
      .then(r => { if (!live) return; setInfo(r.info); if (r.source && r.source !== source) setSource(r.source) }).catch(() => {})
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [owner, repo])
  return info
}

function PickDetail({ group, top, running, onDownload, source, setSource, onOpen }: { group: ModelGroup; top: boolean; running: boolean; onDownload: (s: Start) => void; onOpen: (id: string) => void } & Source) {
  const tr = useTr()
  const [id, setId] = useState(group.models[0].id)
  const model = group.models.find(m => m.id === id) || group.models[0]
  // the optional files (projector, draft, draft head) are wanted unless switched off
  const [off, setOff] = useState<string[]>([])
  const info = useRepoInfo(model.owner, model.repo, source, setSource)
  const need = (m: DlModel) => m.files.filter(f => f.required).reduce((n, f) => n + f.size, 0)
  const wanted = model.files.filter(f => f.required || !off.includes(f.role))
  const bytes = wanted.filter(f => !f.present).reduce((n, f) => n + f.size, 0)
  const toggle = (role: string) => (on: boolean) => setOff(on ? off.filter(r => r !== role) : [...off, role])
  const roles = [...new Set(model.files.filter(f => !f.required).map(f => f.role))]
  const shards = model.files.filter(f => f.role === 'model').length
  let shard = 0

  return (
    <ModelPane title={group.name} owner={model.owner} repo={model.repo} info={info} caps={model.capabilities} tested license={model.license}
      options={group.models.map((m, i) => ({ key: m.id, label: m.variant,
        size: need(m), present: m.files.filter(f => f.required).every(f => f.present), complete: true, fits: m.fits, recommended: top && i === 0 }))}
      option={model.id} onOption={setId}
      companions={roles.map(role => {
        const files = model.files.filter(f => f.role === role)
        return { role, size: files.reduce((n, f) => n + f.size, 0), present: files.every(f => f.present), on: !off.includes(role), toggle: toggle(role) }
      })}
      files={model.files.map(f => {
        if (f.role === 'model') shard += 1
        return { key: f.name, name: f.name, size: f.size, present: !!f.present, on: f.required || !off.includes(f.role),
          label: f.role === 'model' && shards > 1 ? tr('dl.shard', { n: shard, of: shards }) : tr(`dl.role.${f.role}`),
          toggle: f.required ? undefined : toggle(f.role) }
      })}
      bytes={bytes} firstFile={model.files.find(f => f.role === 'model')?.name}
      running={running} onOpen={onOpen} setSource={setSource} readme={{ commit: 'main', source }}
      onDownload={() => onDownload({ plan: { kind: 'catalog', model: model.id, include: roles.filter(r => !off.includes(r)) },
        title: `${model.name} (${model.variant})`, bytes, owner: model.owner, repo: model.repo })} />
  )
}

function RepoDetail({ id, path, source, setSource, running, onDownload, onOpen }: { id: string; path?: string; running: boolean; onDownload: (s: Start) => void; onOpen: (id: string) => void } & Source) {
  const tr = useTr()
  const [listing, setListing] = useState<RepoListing>()
  const [error, setError] = useState<unknown>('')
  const [model, setModel] = useState('')
  const [extras, setExtras] = useState<string[]>([])

  useEffect(() => {
    let live = true
    request<RepoListing>('dl_repo', { url: id, source }).then(l => {
      if (!live) return
      // the other source answered: follow it (the downloads then come from there too)
      if (l.source && l.source !== source) setSource(l.source)
      setListing(l)
      const models = l.groups.filter(g => g.role === 'model' && g.complete)
      // the group's folder (UD-Q4_K_XL/...) or the group itself (a shard's path starts with its key)
      const dir = (k: string) => k.slice(0, Math.max(0, k.lastIndexOf('/')))
      const linked = path ? models.find(g => path.startsWith(g.key) || (!!dir(g.key) && (path === dir(g.key) || path.startsWith(dir(g.key) + '/')))) : undefined
      const first = linked || models.find(g => g.tuned) || models[0]
      setModel(first?.key || '')
      setExtras(companionsFor(first, l.groups))
    }).catch(e => { if (live) setError(String(e)) })
    return () => { live = false }
    // setSource is stable; the pane follows a source change without reading the repository again
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, path])

  if (error) return <div className="px-6 py-5"><Notice tone="error">{describeError(error, tr)}</Notice></div>
  if (!listing) return <div className="flex items-center justify-center gap-2 px-6 py-10 text-xs text-muted-foreground"><LoaderCircle className="size-3.5 animate-spin" />{tr('dl.loadingRepo')}</div>

  const groups = listing.groups
  const models = groups.filter(g => g.role === 'model')
  const others = groups.filter(g => g.role !== 'model')
  const current = models.find(g => g.key === model)
  const chosen = groups.filter(g => g.key === model || extras.includes(g.key))
  const bytes = chosen.flatMap(g => g.files).filter(f => !f.present).reduce((n, f) => n + f.size, 0)
  const present = (g: RepoGroup) => g.files.length > 0 && g.files.every(f => f.present)
  const choose = (key: string) => { setModel(key); setExtras(companionsFor(models.find(g => g.key === key), groups)) }
  const flip = (key: string) => (on: boolean) => setExtras(on ? [...extras, key] : extras.filter(k => k !== key))
  // a switch per kind: on when any file of the kind is picked; switching it on picks the one companionsFor would
  const roles = [...new Set(others.map(g => g.role))].filter(r => !!bestCompanion(r, current, groups))
  const caps = [others.some(g => g.role === 'projection') ? 'images' : '', others.some(g => g.role === 'draft') ? 'mtp' : ''].filter(Boolean)

  return (
    <ModelPane title={listing.repo} owner={listing.owner} repo={listing.repo} info={listing.info} caps={caps} tested={models.some(g => g.tuned)}
      options={models.map(g => ({ key: g.key, label: g.quant || g.name, sub: g.shards > 1 ? tr('dl.shards', { count: g.shards }) : undefined,
        size: g.size, present: present(g), complete: g.complete, fits: g.fits, tuned: g.tuned }))}
      option={model} onOption={choose}
      companions={roles.map(role => {
        const picked = chosen.filter(g => g.role === role)
        const best = bestCompanion(role, current, groups)!
        const shown = picked.length ? picked : [best]
        return { role, size: shown.reduce((n, g) => n + g.size, 0), present: shown.every(present), on: picked.length > 0,
          toggle: (on: boolean) => setExtras(on ? [...extras, best.key] : extras.filter(k => !others.some(g => g.key === k && g.role === role))) }
      })}
      files={[
        ...(current?.files || []).map((f, n) => ({ key: f.path, name: f.name, size: f.size, present: !!f.present, on: true,
          label: current!.shards > 1 ? tr('dl.shard', { n: n + 1, of: current!.shards }) : tr('dl.role.model') })),
        ...others.map(g => ({ key: g.key, name: g.name, size: g.size, present: present(g), on: extras.includes(g.key),
          label: tr(`dl.role.${g.role}`, { defaultValue: g.role }), toggle: flip(g.key) })),
      ]}
      bytes={bytes} firstFile={current?.files[0]?.name}
      running={running} onOpen={onOpen} setSource={setSource} readme={{ commit: listing.commit, source: listing.source || source }}
      onDownload={() => onDownload({ plan: { kind: 'repo', url: `${listing.owner}/${listing.repo}`, groups: chosen.map(g => g.key) },
        title: current ? `${listing.owner}/${listing.repo}: ${current.quant || current.name}` : `${listing.owner}/${listing.repo}`, bytes, owner: listing.owner, repo: listing.repo })} />
  )
}

// the file of a kind that goes with a model: a vision projector (F16 first: what the app looks for by name) or an
// MTP draft of the right kind (an EXL3 model takes the draft with its own head, the others never do; among the rest
// the shared Q4_K_M draft the app is tuned with, then any Q4_K_M, then the smallest). A repository often carries
// several of each (BF16 / F16 / F32 projectors, drafts per type); the others are under Files for the user to add.
function bestCompanion(role: string, model: RepoGroup | undefined, groups: RepoGroup[]) {
  if (!model) return undefined
  const exl3 = model.quant.startsWith('EXL3')
  const best = (list: RepoGroup[], score: (g: RepoGroup) => number) =>
    list.slice().sort((a, b) => score(a) - score(b) || a.size - b.size)[0]
  if (role === 'projection')
    return best(groups.filter(g => g.role === 'projection'), g => (/f16/i.test(g.name) && !/bf16/i.test(g.name) ? 0 : /bf16/i.test(g.name) ? 1 : 2))
  if (role === 'draft')
    return best(groups.filter(g => g.role === 'draft' && g.name.toLowerCase().includes('ownhead') === exl3),
      g => (/shared-q4_k_m/i.test(g.name) ? 0 : /q4_k_m/i.test(g.name) ? 1 : 2))
  return undefined
}

function companionsFor(model: RepoGroup | undefined, groups: RepoGroup[]) {
  return ['projection', 'draft'].map(r => bestCompanion(r, model, groups)).filter((g): g is RepoGroup => !!g).map(g => g.key)
}

function Readme({ owner, repo, commit, source, onSource }: { owner: string; repo: string; commit: string; source: string; onSource?: (s: string) => void }) {
  const tr = useTr()
  const [text, setText] = useState<string | null>(null)
  const [failed, setFailed] = useState<unknown>('')
  useEffect(() => {
    let live = true
    setText(null); setFailed('')
    request<{ text: string; source: string }>('dl_readme', { owner, repo, commit, source })
      .then(r => { if (!live) return; setText(r.text); if (r.source && r.source !== source) onSource?.(r.source) })
      .catch(e => { if (live) { setText(''); setFailed(String(e)) } })
    return () => { live = false }
    // read once per card: a source change it caused itself must not read it again
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [owner, repo, commit])
  return (
    <div className="mt-6 border-t pt-4">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('dl.readme')}</div>
      {text === null ? <div className="text-xs text-muted-foreground">{tr('dl.readmeLoading')}</div>
        : text ? <div className="text-[13px]"><RenderMarkdown content={text} /></div>
        : failed ? <Notice tone="error">{describeError(failed, tr)}</Notice>
        : <div className="text-xs text-muted-foreground">{tr('dl.readmeNone')}</div>}
    </div>
  )
}

// the owner's other GGUF repositories, most downloaded first
function MoreFrom({ owner, repo, source, onOpen }: { owner: string; repo: string; source: string; onOpen: (id: string) => void }) {
  const tr = useTr()
  const [hits, setHits] = useState<SearchHit[]>([])
  useEffect(() => {
    let live = true
    setHits([])
    request<{ results: SearchHit[] }>('dl_search', { author: owner, source })
      .then(r => { if (live) setHits(r.results.filter(h => h.repo !== repo).slice(0, 6)) }).catch(() => {})
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [owner, repo])
  if (!hits.length) return null
  return (
    <div className="mt-6 border-t pt-4">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('dl.moreFrom', { owner })}</div>
      <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
        {hits.map(h => (
          <button key={h.id} type="button" onClick={() => onOpen(h.id)}
            className="rounded-md border px-3 py-2 text-left transition-colors hover:bg-[rgba(127,127,127,0.1)]">
            <div className="truncate text-xs font-medium text-foreground">{h.repo}</div>
            <div className="mt-0.5 flex items-center gap-2 text-[11px] tabular-nums text-muted-foreground">
              <span>{count(h.downloads)} ↓</span>
              {h.params ? <span>· {params(h.params)}</span> : null}
              {h.architecture && <span className="ml-auto font-mono">{h.architecture}</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function StartDialog({ start, catalog, source: initial, setSource: remember, onClose, onStarted }: { start: Start; catalog: DlCatalog; onClose: () => void; onStarted: () => Promise<void> } & Source) {
  const tr = useTr()
  const serviceHub = useServiceHub()
  const root = catalog.default_root || ''
  const [folder, setFolder] = useState(root ? [root, start.owner, start.repo].join(sep(root)) : '')
  const [source, setSource] = useState(initial)
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
      remember(source)
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
