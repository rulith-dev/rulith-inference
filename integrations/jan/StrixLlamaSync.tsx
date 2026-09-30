import { useEffect, useRef } from 'react'
import cloneDeep from 'lodash/cloneDeep'
import { toast } from 'sonner'
import { useModelProvider } from '@/hooks/useModelProvider'
import { openAIProviderSettings } from '@/constants/providers'
import { localStorageKey } from '@/constants/localStorage'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { ENDPOINT, LAST_MODEL_KEY, PROVIDER, autoloadEnabled, request, serverState, useStrixLlamaStatus, type ServerState } from './status'
// Rulith's palette and type for the whole app, loaded with the root route
import './rulith-theme.css'
// downloads from the page (a code block's button, a table's CSV) go through the app: see downloads.ts
import './downloads'

// Mounted once at the root. Polls the manager, and keeps the chat side of Jan pointed at it.
//
// Jan's chat talks to whatever it calls a provider. This build has exactly one, the local server
// tools/manager.py starts, so it is registered here on first run instead of asking for a name, a
// URL and a key that could only ever be one thing. Its model list follows the running server:
// what /v1/models reports, shown under the name the catalog has for the file. Nothing here is
// specific to this machine: the endpoint is the manager's fixed loopback port.
//
// It also loads the last model when the app starts (Configuration > Startup), once per app
// session, and says so wherever the user is when a load finishes or fails.
let autoloadTried = false

export function StrixLlamaSync() {
  const { t } = useTranslation()
  const refresh = useStrixLlamaStatus((s) => s.refresh)
  const status = useStrixLlamaStatus((s) => s.status)
  const providers = useModelProvider((s) => s.providers)
  const previous = useRef<ServerState | undefined>(undefined)
  const stopped = useRef<number | undefined>(undefined)

  useEffect(() => {
    let pending = false
    const tick = async () => {
      if (pending) return
      pending = true
      try { await refresh() } finally { pending = false }
    }
    void tick()
    const timer = setInterval(tick, 3000)
    return () => clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    if (!status || autoloadTried) return
    autoloadTried = true
    const last = localStorage.getItem(LAST_MODEL_KEY)
    // only a clean start: not over a running server, not straight into a load that just failed
    if (!autoloadEnabled() || !last || status.identity || status.failure || !status.runtime_available) return
    request('start', { id: last }).then(() => refresh(), () => { /* a model that is gone: the welcome screen shows the choice */ })
  }, [status, refresh])

  useEffect(() => {
    const state = serverState(status)
    if (previous.current === 'loading' && state === 'ready')
      toast.success(t('strixllama:notice.loaded', { model: status?.model_name ?? '' }))
    else if (previous.current === 'loading' && state === 'failed')
      toast.error(t('strixllama:notice.loadFailed'))
    previous.current = status ? state : previous.current
  }, [status, t])

  // answers the server stopped because Windows ran out of commit: said wherever you are, since the chat
  // itself only shows an answer that ends mid-sentence
  useEffect(() => {
    if (!status) return
    const n = status.identity ? status.memory_events?.answers_stopped ?? 0 : 0
    if (stopped.current !== undefined && n > stopped.current)
      toast.error(t('strixllama:memory.toast'), { description: t('strixllama:memory.why'), duration: 20000 })
    stopped.current = n
  }, [status, t])

  useEffect(() => {
    // Wait for Jan's own list before adding to it: the persisted store hydrates over whatever is
    // in memory, and an entry added before that would be lost and added twice.
    if (!providers.length || providers.some((p) => p.provider === PROVIDER)) return
    const settings = cloneDeep(openAIProviderSettings) as ProviderSetting[]
    for (const s of settings) {
      if (s.key === 'base-url') (s.controller_props as { value?: string }).value = ENDPOINT
      if (s.key === 'api-key') (s.controller_props as { value?: string }).value = ''
    }
    useModelProvider.getState().addProvider({
      provider: PROVIDER,
      active: true,
      models: [],
      settings,
      base_url: ENDPOINT,
      api_key: '',
    })
  }, [providers])

  // Models this provider listed before 0.3.6 carry no capabilities, and the list below is refreshed only from a running
  // server: until a model had loaded, the chat box offered no web search (and, in 0.3.6, no documents). Whatever it
  // serves takes tools, so they get 'tools' as soon as the store has hydrated.
  useEffect(() => {
    const p = providers.find((p) => p.provider === PROVIDER)
    if (!p || p.models.every((m) => m.capabilities?.includes('tools'))) return
    useModelProvider.getState().updateProvider(PROVIDER, {
      models: p.models.map((m) => ({ ...m, capabilities: ['tools', ...(m.capabilities ?? []).filter((c) => c !== 'tools')] })),
    })
  }, [providers])

  useEffect(() => {
    if (status?.status !== 'ready' || !status.served_models?.length) return
    const p = providers.find((p) => p.provider === PROVIDER)
    if (!p) return
    // What the chat box offers depends on these: 'tools' for documents and Jan's web search (off until turned on,
    // apply.py web_tools()), 'vision' for adding, pasting and dropping images - only when this load has the projector.
    const capabilities = ['tools', ...(status.profile?.vision ? ['vision'] : [])]
    const models = status.served_models.map((m) => ({
      ...p.models.find((old) => old.id === m.id),
      id: m.id,
      displayName: status.model_name || m.id,
      capabilities,
    }))
    if (JSON.stringify(p.models) !== JSON.stringify(models)) {
      useModelProvider.getState().updateProvider(PROVIDER, { models })
    }
    // Nothing chosen yet (a fresh install, or the picker mounted before the server answered):
    // pick the model that is actually loaded, and make it the one new chats start with.
    const store = useModelProvider.getState()
    if (!store.selectedModel && models.length) {
      store.selectModelProvider(PROVIDER, models[0].id)
      localStorage.setItem(localStorageKey.lastUsedModel, JSON.stringify({ provider: PROVIDER, model: models[0].id }))
    }
  }, [status, providers])

  return null
}
