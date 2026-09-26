import { useEffect } from 'react'
import { Link, useRouterState } from '@tanstack/react-router'
import { ArrowDownToLine, Check, ChevronsUpDown, Database, ScrollText, SlidersHorizontal, Settings, Play, Square, RotateCw } from 'lucide-react'
import { toast } from 'sonner'
import { SidebarGroup, SidebarMenu, SidebarMenuButton, SidebarMenuItem } from '@/components/ui/sidebar'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import { useAppUpdater } from '@/hooks/useAppUpdater'
import { gb, modelLabel, serverState, useStrixLlamaStatus, type Model } from './status'
import { useModelStore, useNextModel } from './store'
import { StateDot, StatePill, openExternal, useTr } from './parts'

// Jan's sidebar in the shape of Rulith's: the product's mark at the top, the model pages as their own group
// under the chat entries, and at the foot the loaded model with the one action that matters for it
// (Rulith's "Selected agent" panel), a menu to run another model, then settings and the guide.

export function SidebarBrand() {
  return (
    <div className="flex min-w-0 items-center gap-2.5 pl-1.5">
      <img src="/images/jan-logo.png" alt="" className="size-6 shrink-0 rounded-md" />
      <div className="min-w-0 leading-tight">
        <div className="truncate text-[13px] font-semibold uppercase tracking-wide">Strix Llama</div>
        <div className="truncate text-[11px] text-muted-foreground">by Rulith</div>
      </div>
    </div>
  )
}

export function ModelNav() {
  const tr = useTr()
  const pathname = useRouterState({ select: s => s.location.pathname })
  const items = [
    { key: 'models', to: '/strixllama/models', icon: Database },
    { key: 'configuration', to: '/strixllama/configuration', icon: SlidersHorizontal },
    { key: 'logs', to: '/strixllama/logs', icon: ScrollText },
  ] as const
  return (
    <SidebarGroup className="px-0 pb-0">
      <div className="px-2 pb-1 text-xs font-semibold text-muted-foreground">{tr('title')}</div>
      <SidebarMenu>
        {items.map(item => (
          <SidebarMenuItem key={item.key}>
            <SidebarMenuButton asChild isActive={pathname.startsWith(item.to)}>
              <Link to={item.to}>
                <item.icon className="text-foreground/70" />
                <span>{tr(`tabs.${item.key}`)}</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        ))}
      </SidebarMenu>
    </SidebarGroup>
  )
}

export function ModelPanel() {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const catalog = useModelStore(s => s.catalog)
  const loadCatalog = useModelStore(s => s.loadCatalog)
  const busy = useModelStore(s => s.busy)
  const select = useModelStore(s => s.select)
  const load = useModelStore(s => s.load)
  const unload = useModelStore(s => s.unload)
  const pathname = useRouterState({ select: s => s.location.pathname })
  const { updateState, setRemindMeLater } = useAppUpdater()
  // the panel names the model it would load, so it reads the catalog once, as the model pages do
  useEffect(() => { if (!catalog) void loadCatalog() }, [catalog, loadCatalog])
  const state = serverState(status)
  const next = useNextModel()
  const models = catalog?.models.filter(m => m.role === 'model') ?? []
  const running = status?.identity ? models.find(m => m.path === status.model_path) : undefined
  const shown = running ?? next
  const label = shown ? modelLabel(shown) : status?.identity ? status.model_name : undefined
  // picking a model runs it: the one running is unloaded first, and the model pages follow the choice
  const pick = async (m: Model) => {
    await select(m.id)
    if (running?.id === m.id) return
    if (await load(m.id)) toast(tr(status?.identity ? 'notice.switching' : 'notice.loading', { model: modelLabel(m) }))
  }
  return (
    <div className="border-t px-2 pb-2 pt-3">
      <div className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('panel.title')}</div>
      <div className="flex flex-wrap items-center gap-1.5 px-1">
        <StatePill state={state} />
        {status?.identity
          ? <Button size="xs" variant="outline" className="h-6" disabled={busy} onClick={() => void unload()}><Square className="size-3" />{tr('actions.unload')}</Button>
          : <Button size="xs" variant="outline" className="h-6" disabled={busy || !status?.runtime_available || !next} onClick={() => void load(next?.id)}>
              {state === 'failed' ? <RotateCw className="size-3" /> : <Play className="size-3" />}{state === 'failed' ? tr('actions.retry') : tr('actions.load')}
            </Button>}
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild disabled={busy || !models.length}>
          <button type="button" aria-label={tr('panel.choose')} title={label}
            className="mt-1.5 flex w-full min-w-0 items-center gap-1 rounded-md px-1 py-1 text-left text-xs text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground disabled:pointer-events-none">
            <span className="truncate">{label ?? tr('panel.noModel')}</span>
            {models.length > 0 && <ChevronsUpDown className="ml-auto size-3 shrink-0" />}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent side="top" align="start" className="w-72">
          <DropdownMenuLabel className="text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">{tr('panel.switchLabel')}</DropdownMenuLabel>
          {models.map(m => (
            <DropdownMenuItem key={m.id} className="cursor-pointer items-start" onClick={() => void pick(m)}>
              <Check className={cn('mt-0.5 size-3.5 shrink-0', m.id === shown?.id ? 'opacity-100' : 'opacity-0')} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px]">{m.name}</div>
                <div className="truncate text-xs text-muted-foreground">{[m.quant, gb(m.size)].filter(Boolean).join(' · ')}</div>
              </div>
              {m.id === running?.id && <StateDot state={state} className="mt-1.5" />}
            </DropdownMenuItem>
          ))}
          {status?.identity && <>
            <DropdownMenuSeparator />
            <div className="px-2 py-1.5 text-xs text-muted-foreground">{tr('panel.switchNote')}</div>
          </>}
        </DropdownMenuContent>
      </DropdownMenu>
      <div className="mt-2 flex items-center justify-between gap-2 px-1 text-xs">
        <Link to="/settings/general" className={`flex items-center gap-1.5 hover:text-foreground ${pathname.startsWith('/settings') ? 'text-foreground' : 'text-muted-foreground'}`}>
          <Settings className="size-3.5" />{tr('panel.settings')}
        </Link>
        <span className="flex items-center gap-2 text-muted-foreground">
          <button type="button" className="hover:text-foreground" onClick={() => openExternal('https://github.com/rulith-dev/strixllama/blob/main/docs/getting-started.md')}>{tr('panel.guide')} ↗</button>
          {updateState.isUpdateAvailable ? (
            // a newer release verified by the updater: open its prompt (release notes, Update) again
            <button type="button" title={tr('panel.updateTitle', { version: updateState.updateInfo?.version ?? '' })}
              disabled={updateState.isDownloading} onClick={() => setRemindMeLater(false)}
              className="flex items-center gap-1 rounded-md border border-[var(--rl-blue)]/40 px-1.5 py-0.5 text-[var(--rl-blue)] hover:bg-[var(--rl-blue)]/10 disabled:opacity-80">
              <ArrowDownToLine className="size-3" />
              {updateState.isDownloading ? tr('panel.updating', { pct: Math.round(updateState.downloadProgress * 100) }) : tr('panel.update')}
            </button>
          ) : <span className="text-muted-foreground/60">v{VERSION}</span>}
        </span>
      </div>
    </div>
  )
}
