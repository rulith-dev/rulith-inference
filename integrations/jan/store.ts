import { create } from 'zustand'
import { request, useStrixLlamaStatus, LAST_MODEL_KEY, type Catalog, type Companions, type Profile } from './status'

// The catalog and the profile being edited, shared by the model pages, the welcome screen and the
// header's load button. Actions return whether they succeeded; a failure is kept raw in `error`
// (the manager's JSON record or a plain string) for the page to render in its own language.
type Store = {
  catalog?: Catalog
  selected: string
  profile?: Profile          // as edited on the page
  saved?: Profile            // as saved in the manager's settings
  companions?: Companions
  busy: boolean
  error: unknown
  loadCatalog: (refresh?: boolean) => Promise<Catalog | undefined>
  select: (id: string) => Promise<void>
  setField: <K extends keyof Profile>(key: K, value: Profile[K]) => void
  discard: () => void
  save: () => Promise<boolean>
  load: (id?: string) => Promise<boolean>
  unload: () => Promise<boolean>
  reload: () => Promise<boolean>
  setRoots: (roots: string[]) => Promise<boolean>
  clearError: () => void
}

const SELECTED_KEY = 'strixllama-selected'
const refreshStatus = () => useStrixLlamaStatus.getState().refresh()

// two profiles are the same settings whatever order their keys came back in
export const sameProfile = (a?: Profile, b?: Profile) => {
  if (!a || !b) return false
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]) as Set<keyof Profile>
  for (const k of keys) if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) return false
  return true
}

export const useModelStore = create<Store>((set, get) => {
  // one action at a time; the flag also disables the buttons that would start another
  const act = async (fn: () => Promise<void>) => {
    if (get().busy) return false
    set({ busy: true, error: '' })
    try {
      await fn()
      return true
    } catch (e) {
      set({ error: String(e) })
      return false
    } finally {
      set({ busy: false })
    }
  }

  return {
    selected: '',
    busy: false,
    error: '',
    loadCatalog: async (refresh = false) => {
      try {
        const catalog = await request<Catalog>('catalog', { refresh })
        set({ catalog })
        // a rescan may have combined a downloaded draft head with its shared draft and failed to
        // combine another; the page says both, the first as news and the second as an error
        const failed = Object.entries(catalog.merge_errors || {})
        if (failed.length) set({ error: failed.map(([p, e]) => `${p.split(/[\\/]/).pop()}: ${e}`).join('\n') })
        // keep the selection if it still exists, else the running model, the remembered one, the first
        const models = catalog.models.filter(m => m.role === 'model')
        const running = useStrixLlamaStatus.getState().status?.model_path
        const current = get().selected
        const pick = [current, sessionStorage.getItem(SELECTED_KEY), localStorage.getItem(LAST_MODEL_KEY)]
          .find(id => id && models.some(m => m.id === id))
          || models.find(m => m.path === running)?.id || models[0]?.id || ''
        if (pick !== current || !get().profile) await get().select(pick)
        return catalog
      } catch (e) {
        set({ error: String(e) })
        return undefined
      }
    },
    select: async (id) => {
      set({ selected: id, profile: undefined, saved: undefined, companions: undefined })
      if (!id) return
      sessionStorage.setItem(SELECTED_KEY, id)
      try {
        const r = await request<{ profile: Profile; companions: Companions }>('profile', { id })
        if (get().selected === id) set({ profile: r.profile, saved: { ...r.profile }, companions: r.companions })
      } catch (e) {
        if (get().selected === id) set({ error: String(e) })
      }
    },
    setField: (key, value) => set(s => (s.profile ? { profile: { ...s.profile, [key]: value } } : {})),
    discard: () => set(s => (s.saved ? { profile: { ...s.saved }, error: '' } : {})),
    save: () => act(async () => {
      const { selected, profile } = get()
      if (!profile) return
      const r = await request<{ profile: Profile }>('save', { id: selected, profile })
      set({ profile: r.profile, saved: { ...r.profile } })
    }),
    // loads with the saved settings: unsaved edits are saved first by reload(), never loaded silently. A model
    // that is running is unloaded first - the manager starts one only when none runs - so this also switches.
    load: (id) => act(async () => {
      // from the chat header before any model page has been open, nothing is selected yet: read the
      // catalog, which selects the remembered model, else the first one
      if (!id && !get().selected) await get().loadCatalog()
      const target = id || get().selected
      if (!target) return
      if (useStrixLlamaStatus.getState().status?.identity) await request('stop')
      await request('start', { id: target })
      localStorage.setItem(LAST_MODEL_KEY, target)
      await refreshStatus()
    }),
    unload: () => act(async () => {
      await request('stop')
      await refreshStatus()
    }),
    reload: () => act(async () => {
      const { selected, profile, saved } = get()
      if (!selected) return
      if (profile && !sameProfile(profile, saved)) {
        const r = await request<{ profile: Profile }>('save', { id: selected, profile })
        set({ profile: r.profile, saved: { ...r.profile } })
      }
      if (useStrixLlamaStatus.getState().status?.identity) await request('stop')
      await request('start', { id: selected })
      localStorage.setItem(LAST_MODEL_KEY, selected)
      await refreshStatus()
    }),
    setRoots: (roots) => act(async () => {
      const catalog = await request<Catalog>('roots', { roots })
      set({ catalog })
    }),
    clearError: () => set({ error: '' }),
  }
})

// What Load starts when no model is named: the one selected on the model pages, which is the one loaded
// last until you pick another. The sidebar panel and the chat banner name it and pass it on, so they say
// what they do - and a model picked under Configuration is the one they load.
export function useNextModel() {
  const catalog = useModelStore(s => s.catalog)
  const selected = useModelStore(s => s.selected)
  const models = catalog?.models.filter(m => m.role === 'model') ?? []
  return models.find(m => m.id === selected) ?? models.find(m => m.id === localStorage.getItem(LAST_MODEL_KEY))
}
