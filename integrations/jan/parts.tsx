import type { ReactNode } from 'react'
import { Check, ChevronsUpDown, Info } from 'lucide-react'
import { openUrl } from '@tauri-apps/plugin-opener'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { useTitlebarLayout } from '@/stores/titlebar-layout-store'
import type { ServerState } from './status'

// The small parts every view shares, in Rulith's design language (rulith-theme.css has the palette):
// section labels in small capitals above hairline cards, rows of name / one line / control, pills for state.

export const useTr = () => {
  const { t } = useTranslation()
  return (key: string, vars?: Record<string, unknown>) => t(`strixllama:${key}`, vars)
}

// On Windows and Linux the window's own buttons are an overlay pinned to the top right of every page
// (Jan's WindowControls: right-4, 32 px a button), so whatever sits at the right end of a page's top strip
// has to stop short of them. 0 in the browser preview and where the desktop puts them on the left.
export const useTitlebarRightInset = () => {
  const buttons = useTitlebarLayout(s => s.layout.right.length)
  return (IS_WINDOWS || IS_LINUX) && buttons > 0 ? buttons * 32 + 24 : 0
}

export const openExternal = (url: string) => { void openUrl(url).catch(() => window.open(url, '_blank')) }

const DOT: Record<ServerState, string> = {
  ready: 'bg-[var(--rl-green)]',
  loading: 'bg-[var(--rl-amber)] animate-pulse',
  stopped: 'bg-muted-foreground/50',
  failed: 'bg-destructive',
}

export function StateDot({ state, className }: { state: ServerState; className?: string }) {
  return <span aria-hidden className={cn('inline-block size-2 shrink-0 rounded-full', DOT[state], className)} />
}

const PILL: Record<ServerState, string> = {
  ready: 'border-[var(--rl-green)]/50 bg-[var(--rl-green-soft)] text-[var(--rl-green)]',
  loading: 'border-[var(--rl-amber)]/50 bg-[var(--rl-amber-soft)] text-[var(--rl-amber)]',
  stopped: 'border-border text-muted-foreground',
  failed: 'border-destructive/50 bg-destructive/10 text-destructive',
}

// Rulith's status pill ("Agent running"): an outlined capsule in the state's colour
export function StatePill({ state, className }: { state: ServerState; className?: string }) {
  const tr = useTr()
  return (
    <span className={cn('inline-flex h-6 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium', PILL[state], className)}>
      <StateDot state={state} />
      {tr(`state.${state}`)}
    </span>
  )
}

// A label in small capitals, with an optional count and an action at its right
export function Label({ children, count, action, className }: { children: ReactNode; count?: number; action?: ReactNode; className?: string }) {
  return (
    <div className={cn('mb-2 flex items-center justify-between gap-4', className)}>
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">
        {children}{count !== undefined && <span className="ml-1.5">{count}</span>}
      </h2>
      {action}
    </div>
  )
}

// A section: its label above, the rows in one hairline card
export function Section({ label, count, action, children, className }: { label: ReactNode; count?: number; action?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <section className={className}>
      <Label count={count} action={action}>{label}</Label>
      <div className="divide-y divide-border rounded-lg border bg-card">{children}</div>
    </section>
  )
}

// One setting: its name, a line saying what it does, an optional longer explanation behind an (i),
// and the control on the right. `note` is a line under the row for state the control cannot show.
export function Row({ title, description, info, control, note, disabled }: { title: ReactNode; description?: ReactNode; info?: string; control?: ReactNode; note?: ReactNode; disabled?: boolean }) {
  return (
    <div className="px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-x-8 gap-y-2">
        <div className={cn('min-w-56 flex-1 space-y-0.5', disabled && 'opacity-55')}>
          <div className="flex items-center gap-1.5 text-[13px] font-medium text-foreground">
            {title}
            {info && <InfoTip text={info} />}
          </div>
          {description && <div className="text-xs leading-normal text-muted-foreground">{description}</div>}
        </div>
        {control && <div className="ml-auto shrink-0">{control}</div>}
      </div>
      {note && <div className="mt-2">{note}</div>}
    </div>
  )
}

export function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="text-muted-foreground/70 hover:text-foreground" aria-label={text}>
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-80 text-xs leading-relaxed">{text}</TooltipContent>
    </Tooltip>
  )
}

export type Option = { value: string; label: string; hint?: string }

// A select: an outline button showing the value, a menu with a check on the current one
export function Choice({ value, options, onChange, disabled, className, ariaLabel }: { value: string; options: Option[]; onChange: (v: string) => void; disabled?: boolean; className?: string; ariaLabel?: string }) {
  const current = options.find(o => o.value === value)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled}>
        <Button variant="outline" size="sm" aria-label={ariaLabel} className={cn('h-8 min-w-36 justify-between text-[13px] font-normal', className)}>
          <span className="truncate">{current?.label ?? value}</span>
          <ChevronsUpDown className="ml-2 size-3.5 shrink-0 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-56 max-w-[32rem]">
        {options.map(o => (
          <DropdownMenuItem key={o.value} className="my-0.5 cursor-pointer items-start" onClick={() => onChange(o.value)}>
            <Check className={cn('mt-0.5 size-4 shrink-0', o.value === value ? 'opacity-100' : 'opacity-0')} />
            <div className="min-w-0">
              <div className="truncate">{o.label}</div>
              {o.hint && <div className="truncate text-xs text-muted-foreground">{o.hint}</div>}
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// Rulith's segmented choice ("Last 30 days · Last 90 days · All time"): the current one filled
export function Segments({ value, options, onChange, ariaLabel }: { value: string; options: Option[]; onChange: (v: string) => void; ariaLabel?: string }) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="flex flex-wrap gap-1.5">
      {options.map(o => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} onClick={() => onChange(o.value)}
          className={cn('h-7 rounded-md border px-2.5 text-xs font-medium transition-colors',
            o.value === value ? 'border-primary bg-primary text-primary-foreground' : 'border-border text-foreground hover:bg-accent')}>
          {o.label}{o.hint && <span className={cn('ml-1.5', o.value === value ? 'opacity-70' : 'text-muted-foreground')}>{o.hint}</span>}
        </button>
      ))}
    </div>
  )
}

export function NumberField({ value, onChange, min, max, step = 1, suffix, disabled, ariaLabel }: { value: number | undefined; onChange: (v: number) => void; min: number; max: number; step?: number; suffix?: string; disabled?: boolean; ariaLabel?: string }) {
  return (
    <div className="flex items-center gap-2">
      <Input type="number" aria-label={ariaLabel} className="h-8 w-32 text-right text-[13px] tabular-nums" min={min} max={max} step={step} disabled={disabled}
        value={value === undefined ? '' : String(value)} onChange={e => onChange(Number(e.target.value))} />
      <span className="w-12 text-xs text-muted-foreground">{suffix}</span>
    </div>
  )
}

const TONE = {
  error: 'border-l-destructive bg-destructive/[0.07]',
  warning: 'border-l-[var(--rl-amber)] bg-[var(--rl-amber-soft)]',
  info: 'border-l-[var(--rl-blue)] bg-card',
  success: 'border-l-[var(--rl-green)] bg-[var(--rl-green-soft)]',
}

// Rulith's notice ("Earlier call awaiting recovery"): a card with a coloured left edge, an optional title
export function Notice({ tone, title, children, action, className }: { tone: keyof typeof TONE; title?: ReactNode; children?: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <div role={tone === 'error' ? 'alert' : 'status'}
      className={cn('flex items-start gap-4 rounded-md border border-l-2 px-4 py-3 text-xs', TONE[tone], className)}>
      <div className="min-w-0 flex-1 space-y-1 leading-relaxed">
        {title && <div className="text-[13px] font-semibold text-foreground">{title}</div>}
        {children && <div className="whitespace-pre-line text-muted-foreground">{children}</div>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}

// "Review →": a quiet text action at the end of a row
export function RowAction({ children, onClick, disabled }: { children: ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled}
      className="shrink-0 text-xs text-muted-foreground transition-colors hover:text-foreground disabled:pointer-events-none disabled:opacity-50">
      {children} <span aria-hidden>→</span>
    </button>
  )
}

// The line at the foot of every page, as Rulith's console has it
export function PageFooter() {
  const tr = useTr()
  const links: [string, string][] = [
    ['rulith.ai', 'https://rulith.ai'],
    [tr('footer.source'), 'https://github.com/rulith-dev/strixllama'],
    [tr('footer.guide'), 'https://github.com/rulith-dev/strixllama/blob/main/docs/getting-started.md'],
    [tr('footer.licenses'), 'https://github.com/rulith-dev/strixllama/blob/main/NOTICE.md'],
  ]
  return (
    <footer className="mt-8 border-t pt-4 text-xs text-muted-foreground">
      <span className="font-semibold uppercase tracking-wide">Strix Llama</span> · {tr('footer.madeBy')} ·{' '}
      {links.map(([label, url], i) => (
        <span key={url}>
          <button type="button" className="text-[var(--rl-blue)] hover:underline" onClick={() => openExternal(url)}>{label}</button>
          {i < links.length - 1 && ' · '}
        </span>
      ))}
    </footer>
  )
}
