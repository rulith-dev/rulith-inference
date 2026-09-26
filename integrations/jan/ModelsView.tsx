import { useEffect, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Search, RefreshCw, FolderOpen, FolderPlus, Trash2, Circle, CircleDot } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useServiceHub } from '@/hooks/useServiceHub'
import { cn } from '@/lib/utils'
import { fileName, gb, modelLabel, serverState, useStrixLlamaStatus, type Model } from './status'
import { useModelStore } from './store'
import { Label, RowAction, Segments, StatePill, openExternal, useTr } from './parts'

// Where the README lists the model files, pinned to the revision this build was tested with
export const MODEL_FILES_URL = 'https://github.com/rulith-dev/strixllama#model-files'

export default function ModelsView() {
  const tr = useTr()
  const navigate = useNavigate()
  const status = useStrixLlamaStatus(s => s.status)
  const { catalog, busy, loadCatalog, select, load } = useModelStore()
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState('model')
  const [folders, setFolders] = useState(false)
  const [scanning, setScanning] = useState(false)

  const rescan = async () => {
    setScanning(true)
    const c = await loadCatalog(true)
    setScanning(false)
    // the one thing on these pages that writes a file: say what was combined
    if (c?.merged?.length) toast.success(tr('notice.headMerged', { files: c.merged.map(fileName).join(', ') }))
    else if (c) toast(tr('notice.rescanned', { count: c.models.length }))
  }
  const configure = async (m: Model) => {
    await select(m.id)
    void navigate({ to: '/strixllama/configuration' })
  }
  // with another model running this switches: load() unloads it first
  const loadModel = async (m: Model) => {
    const switching = !!status?.identity
    await select(m.id)
    if (await load(m.id)) toast(tr(switching ? 'notice.switching' : 'notice.loading', { model: modelLabel(m) }))
  }

  const models = catalog?.models || []
  const count = (k: string) => models.filter(m => k === 'all' || m.role === k).length
  const kinds = [
    { value: 'model', label: tr('models.kindModel'), hint: String(count('model')) },
    { value: 'draft', label: tr('models.kindDraft'), hint: String(count('draft')) },
    { value: 'projection', label: tr('models.kindProjection'), hint: String(count('projection')) },
    { value: 'all', label: tr('models.kindAll'), hint: String(count('all')) },
  ]
  const filtered = models.filter(m => (kind === 'all' || m.role === kind)
    && `${m.name} ${m.filename} ${m.architecture} ${m.quant}`.toLowerCase().includes(query.toLowerCase()))
  const running = status?.identity ? status.model_path : undefined

  if (catalog && !models.some(m => m.role === 'model')) return <>
    <EmptyCatalog onFolders={() => setFolders(true)} onRescan={() => void rescan()} scanning={scanning} />
    <ModelFolders open={folders} onOpenChange={setFolders} />
  </>

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-48 flex-1">
          <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input aria-label={tr('models.searchLabel')} className="h-8 pl-8 text-[13px]" placeholder={tr('models.search')} value={query} onChange={e => setQuery(e.target.value)} />
        </div>
        <Button size="sm" variant="outline" className="h-8" onClick={() => setFolders(true)}><FolderOpen className="size-3.5" />{tr('models.folders')}</Button>
        <Button size="sm" variant="outline" className="h-8" disabled={scanning} onClick={() => void rescan()}>
          <RefreshCw className={cn('size-3.5', scanning && 'animate-spin')} />{tr('models.rescan')}
        </Button>
      </div>
      <Segments ariaLabel={tr('models.kindLabel')} value={kind} options={kinds} onChange={setKind} />

      <section>
        <Label count={filtered.length}>{kinds.find(k => k.value === kind)?.label}</Label>
        <div className="divide-y divide-border rounded-lg border bg-card">
          {!catalog ? (
            <div className="px-4 py-8 text-center text-xs text-muted-foreground">{tr('models.loadingCatalog')}</div>
          ) : !filtered.length ? (
            <div className="px-4 py-8 text-center text-xs text-muted-foreground">{tr('models.noMatch')}</div>
          ) : filtered.map(m => {
            const broken = !!(m.error || m.missing?.length)
            const isRunning = running === m.path
            return (
              <div key={m.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
                {isRunning ? <CircleDot className="size-3.5 shrink-0 text-[var(--rl-green)]" /> : <Circle className={cn('size-3.5 shrink-0', broken ? 'text-destructive' : 'text-muted-foreground/60')} />}
                <div className="min-w-48 flex-1">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="truncate text-[13px] font-medium text-foreground" title={m.name}>{m.name}</span>
                    {m.quant && <span className="shrink-0 rounded border px-1.5 font-mono text-[11px] leading-5 text-muted-foreground">{m.quant}</span>}
                  </div>
                  <div className="mt-0.5 truncate text-xs text-muted-foreground" title={m.path}>
                    {[gb(m.size), (m.shards ?? 1) > 1 ? tr('models.shards', { count: m.shards }) : null, m.filename].filter(Boolean).join(' · ')}
                  </div>
                </div>
                <div className="ml-auto flex shrink-0 items-center gap-4">
                  {broken ? <span className="text-xs text-destructive" title={m.error || m.missing?.join(', ')}>{tr('models.incomplete')}</span>
                    : isRunning ? <StatePill state={serverState(status)} />
                    : m.role !== 'model' && <span className="text-xs text-muted-foreground">{tr(`models.role.${m.role}`)}</span>}
                  {m.role === 'model' && !broken && <>
                    {!isRunning && <RowAction disabled={busy || !status?.runtime_available} onClick={() => void loadModel(m)}>{status?.identity ? tr('models.switch') : tr('actions.load')}</RowAction>}
                    <RowAction onClick={() => void configure(m)}>{tr('models.configure')}</RowAction>
                  </>}
                </div>
              </div>
            )
          })}
        </div>
        {catalog && (
          <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
            <span>{tr('models.summary', { count: filtered.length, size: gb(filtered.reduce((n, m) => n + m.size, 0)) })}</span>
            <span>{tr('models.readOnly')}</span>
          </div>
        )}
      </section>
      <ModelFolders open={folders} onOpenChange={setFolders} />
    </>
  )
}

// No model file anywhere: Rulith's "Start or continue work" card
function EmptyCatalog({ onFolders, onRescan, scanning }: { onFolders: () => void; onRescan: () => void; scanning: boolean }) {
  const tr = useTr()
  return (
    <section className="rounded-lg border bg-card px-5 py-4">
      <div className="text-[13px] font-semibold text-foreground">{tr('models.emptyTitle')}</div>
      <p className="mt-1 text-xs text-muted-foreground">{tr('models.emptyBody')}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button size="sm" className="h-8" onClick={onFolders}><FolderOpen className="size-3.5" />{tr('models.folders')}</Button>
        <Button size="sm" variant="outline" className="h-8" disabled={scanning} onClick={onRescan}><RefreshCw className={cn('size-3.5', scanning && 'animate-spin')} />{tr('models.rescan')}</Button>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        {tr('models.getFilesLead')}{' '}
        <button type="button" className="text-[var(--rl-blue)] hover:underline" onClick={() => openExternal(MODEL_FILES_URL)}>{tr('models.getFiles')}</button>
      </p>
    </section>
  )
}

// The folders the catalog scans, edited as a list: add with the system folder picker (or a typed
// path where there is none, as in the browser preview), remove per row, save to rescan.
export function ModelFolders({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const tr = useTr()
  const serviceHub = useServiceHub()
  const setRoots = useModelStore(s => s.setRoots)
  const busy = useModelStore(s => s.busy)
  const [roots, setList] = useState<string[]>([])
  const [typed, setTyped] = useState('')
  // start from the saved list each time the dialog opens
  useEffect(() => {
    if (open) { setList(useModelStore.getState().catalog?.roots || []); setTyped('') }
  }, [open])

  const add = (path: string) => {
    const p = path.trim()
    if (p && !roots.includes(p)) setList([...roots, p])
  }
  const pick = async () => {
    const chosen = await serviceHub.dialog().open({ multiple: false, directory: true })
    if (typeof chosen === 'string') add(chosen)
  }
  const save = async () => {
    if (await setRoots(roots)) {
      onOpenChange(false)
      toast.success(tr('models.foldersSaved'))
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="text-[15px]">{tr('models.folders')}</DialogTitle>
          <DialogDescription className="text-xs">{tr('models.foldersHelp')}</DialogDescription>
        </DialogHeader>
        <div className="divide-y divide-border rounded-lg border">
          {roots.map(r => (
            <div key={r} className="flex items-center gap-2 px-3 py-2">
              <FolderOpen className="size-3.5 shrink-0 text-muted-foreground" />
              <span className="min-w-0 flex-1 truncate font-mono text-xs" title={r}>{r}</span>
              <Button size="icon-xs" variant="ghost" aria-label={tr('models.removeFolder')} onClick={() => setList(roots.filter(x => x !== r))}><Trash2 /></Button>
            </div>
          ))}
          {!roots.length && <p className="px-3 py-3 text-xs text-muted-foreground">{tr('models.noFolders')}</p>}
        </div>
        <div className="flex gap-2">
          <Input className="h-8 font-mono text-xs" placeholder={tr('models.folderPath')} value={typed} onChange={e => setTyped(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') { add(typed); setTyped('') } }} />
          <Button size="sm" variant="outline" className="h-8" onClick={() => { if (typed.trim()) { add(typed); setTyped('') } else void pick() }}>
            <FolderPlus className="size-3.5" />{typed.trim() ? tr('models.addFolder') : tr('models.browse')}
          </Button>
        </div>
        <DialogFooter>
          <Button size="sm" variant="outline" className="h-8" onClick={() => onOpenChange(false)}>{tr('actions.cancel')}</Button>
          <Button size="sm" className="h-8" disabled={busy || !roots.length} onClick={() => void save()}>{tr('models.foldersSave')}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
