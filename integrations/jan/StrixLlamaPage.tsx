import { useEffect } from 'react'
import upperFirst from 'lodash/upperFirst'
import { Link } from '@tanstack/react-router'
import { X } from 'lucide-react'
import HeaderPage from '@/containers/HeaderPage'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { clock, describeError, serverState, useStrixLlamaStatus } from './status'
import { useModelStore } from './store'
import { Notice, PageFooter, useTr } from './parts'
import ModelsView from './ModelsView'
import ConfigurationView from './ConfigurationView'
import LogsView from './LogsView'
import './strixllama.css'

export type View = 'models' | 'configuration' | 'logs'

// A model page in the layout of Rulith's console: the page's name in a strip across the top (Jan's header,
// which also brings the sidebar toggle back when the sidebar is closed), then one centred column. Problems
// that persist - a failed load, little Windows commit left, a manager that does not answer - lead it.
export default function StrixLlamaPage({ view }: { view: View }) {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const pollError = useStrixLlamaStatus(s => s.error)
  const catalog = useModelStore(s => s.catalog)
  const loadCatalog = useModelStore(s => s.loadCatalog)
  const error = useModelStore(s => s.error)
  const clearError = useModelStore(s => s.clearError)
  useEffect(() => { if (!catalog) void loadCatalog() }, [catalog, loadCatalog])
  const state = serverState(status)

  return (
    <div className="flex h-svh w-full flex-col">
      <div className="shrink-0 border-b">
        <HeaderPage>
          <div className="mx-auto w-full max-w-[1280px] px-6 text-[15px] font-semibold">{tr(`tabs.${view}`)}</div>
        </HeaderPage>
      </div>
      <main className={cn('min-h-0 flex-1', view === 'logs' ? 'flex flex-col overflow-hidden' : 'overflow-y-auto')}>
        <div className={cn('mx-auto flex w-full max-w-[1280px] flex-col gap-5 px-6 py-6', view === 'logs' && 'min-h-0 flex-1')}>
          {status && !status.runtime_available && <Notice tone="error" title={tr('runtimeMissing')} />}
          {pollError && <Notice tone="error">{describeError(pollError, tr)}</Notice>}
          {error ? (
            <Notice tone="error" action={<Button size="icon-xs" variant="ghost" aria-label={tr('actions.dismiss')} onClick={clearError}><X /></Button>}>
              {describeError(error, tr)}
            </Notice>
          ) : null}
          {state === 'failed' && (
            <Notice tone="error" title={tr('failureTitle')}
              action={view !== 'logs' && <Button size="sm" variant="outline" className="h-7" asChild><Link to="/strixllama/logs">{tr('actions.viewLog')}</Link></Button>}>
              {status?.failure_code ? upperFirst(tr(`errors.${status.failure_code}`)) : status?.failure}
            </Notice>
          )}
          {status?.memory_events && (
            // answers that ended mid-sentence, and why: the server ran out of Windows commit
            <Notice tone="warning" title={tr('memory.title')}
              action={view !== 'configuration' && <Button size="sm" variant="outline" className="h-7" asChild><Link to="/strixllama/configuration">{tr('memory.settings')}</Link></Button>}>
              {[status.memory_events.answers_stopped > 0 && tr('memory.stopped', { time: clock(status.memory_events.last_stop), count: status.memory_events.answers_stopped }),
                status.memory_events.saves_skipped > 0 && tr('memory.skipped', { time: clock(status.memory_events.last_skip) }),
                tr('memory.why')].filter(Boolean).join('\n')}
            </Notice>
          )}
          {status?.commit_low && !status.memory_events && (
            <Notice tone="warning" title={tr('commitLowTitle')}>
              {tr('commitLow', { available: ((status.commit_available ?? 0) / 2 ** 30).toFixed(1), limit: ((status.commit_limit ?? 0) / 2 ** 30).toFixed(1) })}
            </Notice>
          )}
          {view === 'models' && <ModelsView />}
          {view === 'configuration' && <ConfigurationView />}
          {view === 'logs' && <LogsView />}
          {view !== 'logs' && <PageFooter />}
        </div>
      </main>
    </div>
  )
}
