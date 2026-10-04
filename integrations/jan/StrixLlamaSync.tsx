import { useEffect, useRef } from 'react'
import cloneDeep from 'lodash/cloneDeep'
import { toast } from 'sonner'
import { useModelProvider } from '@/hooks/useModelProvider'
import { getServiceHub } from '@/hooks/useServiceHub'
import { openAIProviderSettings } from '@/constants/providers'
import { localStorageKey } from '@/constants/localStorage'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { LAST_MODEL_KEY, PROVIDER, autoloadEnabled, endpointOf, request, serverState, useStrixLlamaNetwork, useStrixLlamaStatus, type ServerState } from './status'
// Rulith's palette and type for the whole app, loaded with the root route
import './rulith-theme.css'
// downloads from the page (a code block's button, a table's CSV) go through the app: see downloads.ts
import './downloads'

// Mounted once at the root. Polls the manager, and keeps the chat side of Jan pointed at it.
//
// Jan's chat talks to whatever it calls a provider. This build has exactly one, the local server
// tools/manager.py starts, so it is registered here on first run instead of asking for a name, a
// URL and a key that could only ever be one thing. Its model list follows the running server:
// what /v1/models reports, shown under the name the catalog has for the file. Its URL and key follow
// Configuration › Network: the running server's while one runs, else the next load's, always on this
// PC's loopback - the address other devices use is for them.
//
// It also loads the last model when the app starts (Configuration > Startup), once per app
// session, and says so wherever the user is when a load finishes or fails.
let autoloadTried = false

// Jan's OpenAI-compatible settings with the server's URL and key in place
const providerSettings = (settings: ProviderSetting[], endpoint: string, key: string) =>
  settings.map((s) =>
    s.key === 'base-url' ? { ...s, controller_props: { ...s.controller_props, value: endpoint } }
    : s.key === 'api-key' ? { ...s, controller_props: { ...s.controller_props, value: key } }
    : s)

export function StrixLlamaSync() {
  const { t } = useTranslation()
  const refresh = useStrixLlamaStatus((s) => s.refresh)
  const status = useStrixLlamaStatus((s) => s.status)
  const providers = useModelProvider((s) => s.providers)
  const network = useStrixLlamaNetwork((s) => s.network)
  const refreshNetwork = useStrixLlamaNetwork((s) => s.refresh)
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

  // The server's address and key, read again when the status says they may have changed: another endpoint, another
  // process (a load starts with the saved settings), a key added or removed. The page's own saves update them too.
  const endpoint = status?.endpoint
  const pid = status?.identity?.pid
  const keyed = status?.api_key_set
  useEffect(() => { void refreshNetwork() }, [endpoint, pid, keyed, refreshNetwork])

  useEffect(() => {
    // Wait for Jan's own list before adding to it: the persisted store hydrates over whatever is
    // in memory, and an entry added before that would be lost and added twice.
    if (!providers.length || providers.some((p) => p.provider === PROVIDER)) return
    const server = useStrixLlamaNetwork.getState().network?.server
    const url = server?.endpoint || endpointOf(useStrixLlamaStatus.getState().status)
    const key = server?.api_key ?? ''
    useModelProvider.getState().addProvider({
      provider: PROVIDER,
      active: true,
      models: [],
      settings: providerSettings(cloneDeep(openAIProviderSettings) as ProviderSetting[], url, key),
      base_url: url,
      api_key: key,
    })
  }, [providers])

  // ...and kept pointed at the server whenever what it holds differs, not only when it is created: a port or key
  // changed under Configuration › Network reaches the chat once the server listens with it. Jan keeps a provider's
  // key in the OS keyring and puts it back at the next start, so a removed one is deleted there too.
  useEffect(() => {
    const server = network?.server
    const p = providers.find((p) => p.provider === PROVIDER)
    if (!server || !p) return
    const url = p.settings.find((s) => s.key === 'base-url')?.controller_props.value
    if (p.base_url === server.endpoint && url === server.endpoint && (p.api_key ?? '') === server.api_key) return
    useModelProvider.getState().updateProvider(PROVIDER, {
      base_url: server.endpoint,
      api_key: server.api_key,
      api_key_fallbacks: [],
      settings: providerSettings(p.settings, server.endpoint, server.api_key),
    })
    if (!server.api_key && p.api_key) {
      try { void getServiceHub().providers().deleteProviderKeys(PROVIDER) } catch { /* no services yet: nothing stored */ }
    }
  }, [network, providers])

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
