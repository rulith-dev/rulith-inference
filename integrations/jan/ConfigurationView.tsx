import { useState } from 'react'
import { ArrowLeftRight, ChevronRight, Play, RotateCw, Square } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import { autoloadEnabled, fileName, gb, modelLabel, serverState, setAutoloadEnabled, useStrixLlamaStatus, type Model, type Profile } from './status'
import { sameProfile, useModelStore } from './store'
import { Choice, NumberField, Notice, Row, Section, StatePill, useTr } from './parts'

// The four levels this model's chat template actually has: it accepts low, medium and xhigh, folds
// 'high' into xhigh, and injects nothing at all for medium. A fifth level would be a duplicate.
const THINKING_LEVELS = ['off', 'low', 'medium', 'high']
// the K/V cache types the manager takes (KV_TYPES in tools/manager.py)
const KV_TYPES = ['f16', 'q8_0']
// trunk_decode_q6k stays in the profile and in the manager: existing profiles keep working and the
// switch is still reachable from the config file. It has no control here - measured prefill-neutral
// and 4% on decode for 2.9 GB, and at the production context it makes long prompts fail to load.

export default function ConfigurationView() {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const { catalog, selected, profile, saved, companions, busy, select, setField, discard, save, reload } = useModelStore()
  const [autoload, setAutoload] = useState(autoloadEnabled)
  const [advanced, setAdvanced] = useState(false)

  const models = catalog?.models.filter(m => m.role === 'model') || []
  const model = catalog?.models.find(m => m.id === selected)
  const drafts = catalog?.models.filter(m => m.role === 'draft') || []
  const projectors = catalog?.models.filter(m => m.role === 'projection') || []
  const qsaModel = model?.architecture === 'qwen4exp'
  const running = !!status?.identity && status.model_path === model?.path
  const dirty = !!profile && !sameProfile(profile, saved)
  // saved but not in effect: the server runs this model with other settings
  const pending = running && !dirty && !!status?.profile && !sameProfile(saved, status.profile)

  if (!catalog) return <div className="p-10 text-center text-sm text-muted-foreground">{tr('models.loadingCatalog')}</div>
  if (!models.length) return <Notice tone="info">{tr('config.noModel')}</Notice>

  const onSave = async () => { if (await save()) toast.success(tr(running ? 'notice.savedRunning' : 'notice.saved')) }
  const onReload = async () => { if (await reload()) toast(tr('notice.loading', { model: model?.name })) }
  const set = <K extends keyof Profile>(key: K) => (v: Profile[K]) => setField(key, v)
  const p = profile

  return (
    <div className="flex flex-col gap-5">
      {pending && (
        <Notice tone="info" title={tr('config.pending')}
          action={<Button size="sm" className="h-7" disabled={busy} onClick={() => void onReload()}><RotateCw className="size-3.5" />{tr('actions.reload')}</Button>} />
      )}

      <Section label={tr('config.model.title')}>
        <Row title={model?.name ?? '—'}
          description={model && <span className="block truncate font-mono text-xs" title={model.path}>{model.path.replace(/[\\/][^\\/]*$/, '')}</span>}
          control={<Choice ariaLabel={tr('config.model.file')} className="max-w-72" value={selected} onChange={id => void select(id)}
            options={models.map(m => ({ value: m.id, label: m.filename, hint: [m.quant, gb(m.size)].filter(Boolean).join(' · ') }))} />} />
        {model && <ServerRow model={model} />}
        {model && (
          <Row title={tr('config.model.attention')} info={qsaModel ? tr('config.model.attentionInfo') : undefined}
            description={qsaModel ? tr('config.model.attentionOn') : tr('config.model.attentionDefault')}
            control={<span className="text-sm text-muted-foreground">{tr('config.model.contextMax', { context: model.context?.toLocaleString() ?? '—' })}</span>} />
        )}
      </Section>

      {p && <>
        <Section label={tr('config.generation.title')}>
          <Row title={tr('config.generation.thinking')} info={tr('config.generation.thinkingInfo')}
            description={tr(`config.generation.thinkingHelp.${p.thinking}`)}
            control={<Choice ariaLabel={tr('config.generation.thinking')} value={String(p.thinking)} onChange={set('thinking')}
              options={THINKING_LEVELS.map(l => ({ value: l, label: tr(`config.generation.thinkingLevel.${l}`) }))} />} />
        </Section>

        <Section label={tr('config.context.title')}>
          <Row title={tr('config.context.length')} description={tr('config.context.lengthHelp')}
            control={<NumberField ariaLabel={tr('config.context.length')} value={p.context} onChange={set('context')} min={512} max={model?.context || 262144} suffix={tr('units.tokens')} />} />
          <Row title={tr('config.context.kvType')} description={tr('config.context.kvTypeHelp')} info={tr('config.context.kvTypeInfo')}
            control={<Choice ariaLabel={tr('config.context.kvType')} value={p.kv || 'f16'} onChange={set('kv')}
              options={KV_TYPES.map(k => ({ value: k, label: tr(`config.context.kvTypeOption.${k}`) }))} />} />
          <Row title={tr('config.context.promptCacheDisk')} description={tr('config.context.promptCacheDiskHelp')} info={tr('config.context.promptCacheDiskInfo')}
            control={<Switch aria-label={tr('config.context.promptCacheDisk')} checked={!!p.prompt_cache_disk} onCheckedChange={set('prompt_cache_disk')} />} />
          {p.prompt_cache_disk && (
            <Row title={tr('config.context.promptCacheDiskMib')} description={tr('config.context.promptCacheDiskMibHelp')}
              control={<NumberField ariaLabel={tr('config.context.promptCacheDiskMib')} value={p.prompt_cache_disk_mib} onChange={set('prompt_cache_disk_mib')} min={1024} max={262144} step={1024} suffix="MB" />} />
          )}
        </Section>

        <Section label={tr('config.concurrency.title')}>
          <Row title={tr('config.concurrency.parallel')} description={tr('config.concurrency.parallelHelp')} info={tr('config.concurrency.parallelInfo')}
            control={<NumberField ariaLabel={tr('config.concurrency.parallel')} value={p.parallel} onChange={set('parallel')} min={1} max={8} />} />
          {/* the pool is shared by several conversations: with one, the manager ignores it and the pool is the context */}
          <Row title={tr('config.concurrency.kvPool')} info={tr('config.concurrency.kvPoolInfo')} disabled={(p.parallel ?? 1) < 2}
            description={(p.parallel ?? 1) < 2 ? tr('config.concurrency.kvPoolSingle') : tr('config.concurrency.kvPoolHelp')}
            control={<NumberField ariaLabel={tr('config.concurrency.kvPool')} disabled={(p.parallel ?? 1) < 2} value={p.kv_pool} onChange={set('kv_pool')} min={0} max={1048576} step={256} suffix={tr('units.tokens')} />} />
        </Section>

        <Section label={tr('config.speculative.title')}>
          <Row title={tr('config.speculative.mtp')} info={tr('config.speculative.mtpInfo')}
            description={qsaModel ? tr('config.speculative.mtpHelp') : tr('config.speculative.mtpUnsupported')}
            control={<Switch aria-label={tr('config.speculative.mtp')} checked={p.mtp} disabled={!qsaModel} onCheckedChange={set('mtp')} />}
            note={qsaModel && companions && !companions.draft && <Notice tone="warning">{tr('config.speculative.noDraft')}</Notice>} />
          <Row title={tr('config.speculative.draftModel')} disabled={!p.mtp}
            description={companions?.draft && !companions.draft_head ? tr('config.speculative.draftHelp') : tr('config.speculative.draftHelpHead')}
            control={<Choice ariaLabel={tr('config.speculative.draftModel')} className="max-w-72" disabled={!p.mtp} value={p.draft} onChange={set('draft')}
              options={[
                // a saved draft outside the scanned folders would otherwise show as whichever option comes first
                ...(p.draft && !drafts.some(d => d.path === p.draft) ? [{ value: p.draft, label: fileName(p.draft), hint: tr('config.speculative.draftMissing') }] : []),
                ...drafts.map(d => ({ value: d.path, label: d.filename, hint: [d.quant, gb(d.size)].filter(Boolean).join(' · ') })),
              ]} />} />
          <Row title={tr('config.speculative.draftMax')} description={tr('config.speculative.draftMaxHelp')} disabled={!p.mtp}
            control={<NumberField ariaLabel={tr('config.speculative.draftMax')} disabled={!p.mtp} value={p.draft_max} onChange={set('draft_max')} min={1} max={8} suffix={tr('units.tokens')} />} />
          <Row title={tr('config.speculative.draftMin')} description={tr('config.speculative.draftMinHelp')} disabled={!p.mtp}
            control={<NumberField ariaLabel={tr('config.speculative.draftMin')} disabled={!p.mtp} value={p.draft_min} onChange={set('draft_min')} min={0} max={1} step={0.05} />} />
        </Section>

        <Section label={tr('config.vision.title')}>
          <Row title={tr('config.vision.toggle')} description={tr('config.vision.help')}
            control={<Switch aria-label={tr('config.vision.toggle')} checked={!!p.vision} onCheckedChange={set('vision')} />}
            note={companions && !companions.mmproj && !p.mmproj && <Notice tone="warning">{tr('config.vision.noProjector')}</Notice>} />
          {projectors.length > 0 && (
            <Row title={tr('config.vision.projector')} disabled={!p.vision}
              control={<Choice ariaLabel={tr('config.vision.projector')} className="max-w-72" disabled={!p.vision} value={p.mmproj || ''} onChange={set('mmproj')}
                options={[{ value: '', label: tr('config.vision.projectorAuto') }, ...projectors.map(m => ({ value: m.path, label: m.filename }))]} />} />
          )}
        </Section>

        <Section label={tr('config.startup.title')}>
          <Row title={tr('config.startup.autoload')} description={tr('config.startup.autoloadHelp')}
            control={<Switch aria-label={tr('config.startup.autoload')} checked={autoload} onCheckedChange={v => { setAutoload(v); setAutoloadEnabled(v) }} />} />
        </Section>

        <Collapsible open={advanced} onOpenChange={setAdvanced}>
          <section className="rounded-lg border bg-card">
            <CollapsibleTrigger asChild>
              <button type="button" className="flex w-full items-center gap-2 px-4 py-3 text-left">
                <ChevronRight className={cn('size-3.5 shrink-0 text-muted-foreground transition-transform', advanced && 'rotate-90')} />
                <span className="text-[13px] font-medium text-foreground">{tr('config.advanced.title')}</span>
                <span className="truncate text-xs text-muted-foreground">{tr('config.advanced.note')}</span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="divide-y divide-border border-t">
              <Row title={tr('config.advanced.gpuLayers')} description={tr('config.advanced.gpuLayersHelp')}
                control={<NumberField ariaLabel={tr('config.advanced.gpuLayers')} value={p.gpu_layers} onChange={set('gpu_layers')} min={0} max={999} />} />
              <Row title={tr('config.advanced.batch')} description={tr('config.advanced.batchHelp')}
                control={<NumberField ariaLabel={tr('config.advanced.batch')} value={p.batch} onChange={set('batch')} min={32} max={32768} suffix={tr('units.tokens')} />} />
              <Row title={tr('config.advanced.ubatch')} description={tr('config.advanced.ubatchHelp')}
                control={<NumberField ariaLabel={tr('config.advanced.ubatch')} value={p.ubatch} onChange={set('ubatch')} min={32} max={32768} suffix={tr('units.tokens')} />} />
              <Row title={tr('config.advanced.threads')} description={tr('config.advanced.threadsHelp')}
                control={<NumberField ariaLabel={tr('config.advanced.threads')} value={p.threads} onChange={set('threads')} min={1} max={32} />} />
              <Row title={tr('config.advanced.ngram')} description={tr('config.advanced.ngramHelp')}
                control={<Switch aria-label={tr('config.advanced.ngram')} checked={!!p.ngram_spec} onCheckedChange={set('ngram_spec')} />} />
            </CollapsibleContent>
          </section>
        </Collapsible>
      </>}


      {dirty && (
        <div className="sticky bottom-3 z-10 flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-popover px-4 py-3 shadow-lg">
          <span className="text-[13px] text-foreground">{running ? tr('config.unsavedRunning') : tr('config.unsaved')}</span>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" className="h-8" disabled={busy} onClick={discard}>{tr('actions.discard')}</Button>
            <Button size="sm" variant={running ? 'outline' : 'default'} className="h-8" disabled={busy} onClick={() => void onSave()}>{tr('actions.save')}</Button>
            {running && <Button size="sm" className="h-8" disabled={busy} onClick={() => void onReload()}><RotateCw className="size-3.5" />{tr('actions.saveReload')}</Button>}
          </div>
        </div>
      )}
    </div>
  )
}

// Whether the model on this page is the one the server runs, and the action that makes it so: Load, Switch
// when another model runs (it is unloaded first), Unload when this one does. Unsaved edits are saved first.
function ServerRow({ model }: { model: Model }) {
  const tr = useTr()
  const status = useStrixLlamaStatus(s => s.status)
  const { catalog, busy, reload, unload } = useModelStore()
  const state = serverState(status)
  const mine = !!status?.identity && status.model_path === model.path
  const other = !!status?.identity && !mine
  const loaded = modelLabel(catalog?.models.find(m => m.path === status?.model_path)) || status?.model_name || ''
  const run = async () => { if (await reload()) toast(tr(other ? 'notice.switching' : 'notice.loading', { model: modelLabel(model) })) }
  return (
    <Row title={tr('config.model.server')}
      description={mine ? tr(state === 'ready' ? 'config.model.serverReady' : 'config.model.serverLoading')
        : other ? tr('config.model.serverOther', { model: loaded }) : tr('config.model.serverStopped')}
      control={mine ? (
        <div className="flex items-center gap-2">
          <StatePill state={state} />
          <Button size="sm" variant="outline" className="h-8" disabled={busy} onClick={() => void unload()}><Square className="size-3.5" />{tr('actions.unload')}</Button>
        </div>
      ) : (
        <Button size="sm" className="h-8" disabled={busy || !status?.runtime_available} onClick={() => void run()}>
          {other ? <ArrowLeftRight className="size-3.5" /> : <Play className="size-3.5" />}{other ? tr('actions.switch') : tr('actions.load')}
        </Button>
      )} />
  )
}
