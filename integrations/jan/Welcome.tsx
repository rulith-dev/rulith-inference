import { useEffect, useState } from 'react'
import upperFirst from 'lodash/upperFirst'
import { Link } from '@tanstack/react-router'
import { FolderOpen, LoaderCircle, Play, RefreshCw, RotateCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { gb, describeError, serverState, useStrixLlamaStatus } from './status'
import { useModelStore } from './store'
import { Label, Notice, PageFooter, openExternal, useTr } from './parts'
import { MODEL_FILES_URL, ModelFolders } from './ModelsView'

// In place of Jan's setup screen, which offers to download Jan's own model: shown until the chat has a model
// to talk to. It finds the model files, loads one, and gets out of the way - the home route renders the chat
// as soon as the provider lists the loaded model. Laid out as Rulith's console: one centred column, a card
// that says what to do next with one white button.
export default function Welcome() {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const { catalog, selected, busy, error, loadCatalog, load } = useModelStore()
  const [folders, setFolders] = useState(false)
  const [scanning, setScanning] = useState(false)
  useEffect(() => { if (!catalog) void loadCatalog() }, [catalog, loadCatalog])

  const state = serverState(status)
  const model = catalog?.models.find(m => m.id === selected)
  const hasModel = !!catalog?.models.some(m => m.role === 'model')
  const rescan = async () => { setScanning(true); await loadCatalog(true); setScanning(false) }

  return (
    <div className="flex h-full w-full flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-[640px] flex-1 flex-col justify-center px-6 py-10">
        <div className="mb-6 flex items-center gap-3">
          <img src="/images/jan-logo.png" alt="" className="size-10 rounded-lg" />
          <div>
            <h1 className="text-lg font-semibold text-foreground">{tr('welcome.title')}</h1>
            <p className="text-xs text-muted-foreground">{tr('welcome.subtitle')}</p>
          </div>
        </div>

        <Label>{hasModel ? tr('welcome.modelLabel') : tr('welcome.setupLabel')}</Label>
        <section className="rounded-lg border bg-card px-5 py-4">
          {!catalog ? (
            <div className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
              <LoaderCircle className="size-3.5 animate-spin" />{tr('welcome.searching')}
            </div>
          ) : !hasModel ? (
            <>
              <div className="text-[13px] font-semibold text-foreground">{tr('welcome.noModelTitle')}</div>
              <p className="mt-1 text-xs text-muted-foreground">{tr('welcome.noModelBody')}</p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button size="sm" className="h-8" onClick={() => setFolders(true)}><FolderOpen className="size-3.5" />{tr('models.folders')}</Button>
                <Button size="sm" variant="outline" className="h-8" disabled={scanning} onClick={() => void rescan()}>
                  <RefreshCw className={scanning ? 'size-3.5 animate-spin' : 'size-3.5'} />{tr('models.rescan')}
                </Button>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                {tr('models.getFilesLead')}{' '}
                <button type="button" className="text-[var(--rl-blue)] hover:underline" onClick={() => openExternal(MODEL_FILES_URL)}>{tr('models.getFiles')}</button>
              </p>
            </>
          ) : (
            <>
              <div className="text-[13px] font-semibold text-foreground">{model?.name}</div>
              {model && <div className="mt-0.5 text-xs text-muted-foreground">{[model.quant, gb(model.size), model.filename].filter(Boolean).join(' · ')}</div>}
              {state === 'loading' || state === 'ready' ? (
                // ready: the chat replaces this screen as soon as the provider lists the model, a poll away
                <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                  <LoaderCircle className="size-3.5 shrink-0 animate-spin text-[var(--rl-amber)]" />
                  <span className="flex-1">{tr(state === 'ready' ? 'welcome.ready' : 'welcome.loading')}</span>
                  {state === 'loading' && <Link to="/strixllama/logs" className="text-[var(--rl-blue)] hover:underline">{tr('actions.viewLog')}</Link>}
                </div>
              ) : (
                <>
                  {state === 'failed' && (
                    <Notice tone="error" className="mt-3" title={tr('failureTitle')}>{status?.failure_code ? upperFirst(tr(`errors.${status.failure_code}`)) : status?.failure}</Notice>
                  )}
                  {error ? <Notice tone="error" className="mt-3">{describeError(error, tr)}</Notice> : null}
                  {status && !status.runtime_available && <Notice tone="error" className="mt-3" title={tr('runtimeMissing')} />}
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <Button size="sm" className="h-8" disabled={busy || !model || !status?.runtime_available} onClick={() => void load()}>
                      {state === 'failed' ? <RotateCw className="size-3.5" /> : <Play className="size-3.5" />}{state === 'failed' ? tr('actions.retry') : tr('actions.load')}
                    </Button>
                    <Link to="/strixllama/configuration" className="text-xs text-[var(--rl-blue)] hover:underline">{tr('welcome.reviewSettings')}</Link>
                  </div>
                  <p className="mt-3 text-xs text-muted-foreground">{tr('welcome.loadHint')}</p>
                </>
              )}
            </>
          )}
        </section>
        <PageFooter />
      </div>
      <ModelFolders open={folders} onOpenChange={setFolders} />
    </div>
  )
}
