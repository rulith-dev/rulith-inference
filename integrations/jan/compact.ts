import type { UIMessage } from '@ai-sdk/react'
import { toast } from 'sonner'
import i18n from '@/i18n'
import { useThreads } from '@/hooks/useThreads'
import { useStrixLlamaStatus } from './status'

// A conversation that outgrows the model's context. Without this the request that crosses it fails ("Model ran out
// of context size") and so does every later one. Jan has a trimmer and a summarizer of its own (lib/context-manager.ts),
// but it estimates 3.5 characters a token - a third of the truth for Chinese - and it re-trims or re-summarizes on
// every request: the prompt's head changes each turn, so the server re-reads the whole window each time (minutes at
// 250K tokens) instead of reusing what it has cached.
//
// Here a conversation is folded once, when a request would pass HIGH of the context: the model summarizes everything
// up to its last answer - asked right after that answer with the same prompt, so the server has it cached and only
// writes the summary - and from then on requests carry the summary in the system prompt and the messages after it,
// the same head every time. The thread keeps the summary and the id of the last message it covers (metadata
// strix_compact); editing or deleting that message drops it. When no summary can be made (the request fails, or the
// conversation is already past what fits), the oldest messages go instead, with a note saying so.

export type CompactState = {
  upto: string                // the id of the last message the text stands for
  text: string
  kind: 'summary' | 'dropped'
}

const KEY = 'strix_compact'
const HIGH = 0.8   // a request estimated past this share of the context is compacted first
const KEEP = 0.5   // without a summary, the oldest messages go until the rest is estimated under this share

export const INSTRUCTION =
  'Our conversation is about to outgrow your context window, so everything above will be replaced by a summary that ' +
  'you write now. Summarize the whole conversation so far for your own later use, so that you can go on as if nothing ' +
  'was lost: what the user wants and asked for, the key facts, figures, names and sources found (with their URLs), ' +
  'decisions and conclusions, code, commands or wording that matter, and what is still open. Give the latest exchange ' +
  'the most detail. Write in the language the user writes in. Do not call any tool. Output only the summary, at most ' +
  'about 800 words.'

// Qwen's tokenizer takes about one token per CJK character and three to four characters of other text: counting CJK
// characters one each and the rest by three errs on the large side, which is the side to err on here
const CJK = /[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]/g
export const estimateTokens = (text: string) => {
  if (!text) return 0
  const cjk = text.match(CJK)?.length ?? 0
  return cjk + Math.ceil((text.length - cjk) / 3)
}

const messageText = (m: UIMessage): string => {
  const out: string[] = []
  for (const p of m.parts ?? []) {
    if (p.type === 'text' || p.type === 'reasoning') out.push((p as { text?: string }).text ?? '')
    else if (p.type === 'dynamic-tool' || p.type.startsWith('tool-')) out.push(JSON.stringify(p))
  }
  const files = (m.metadata as { inline_file_contents?: Array<{ content?: string }> } | undefined)?.inline_file_contents
  if (Array.isArray(files)) for (const f of files) if (f?.content) out.push(f.content)
  return out.join('\n')
}
export const estimateMessage = (m: UIMessage) => estimateTokens(messageText(m)) + 8

// the server's count for the request that produced this answer, and the answer: the conversation up to here
const usageOf = (m: UIMessage): number | undefined => {
  const u = (m.metadata as { usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number } } | undefined)
    ?.usage
  const t = u?.totalTokens ?? (u?.inputTokens ?? 0) + (u?.outputTokens ?? 0)
  return t > 0 ? t : undefined
}

const lastIndex = <T,>(xs: T[], f: (x: T) => boolean) => {
  for (let i = xs.length - 1; i >= 0; i--) if (f(xs[i])) return i
  return -1
}

export const systemWith = (system: string | undefined, state: CompactState) => {
  const block = state.kind === 'summary'
    ? '# Earlier in this conversation\nThe first part of this conversation no longer fits in your context. This is ' +
      'the summary of it that you wrote:\n\n' + state.text
    : state.text
  return system ? system + '\n\n' + block : block
}

const save = (threadId: string, state: CompactState | undefined) => {
  const thread = useThreads.getState().threads[threadId]
  if (!thread) return
  const metadata = { ...(thread.metadata ?? {}) } as Record<string, unknown>
  if (state) metadata[KEY] = state
  else delete metadata[KEY]
  useThreads.getState().updateThread(threadId, { metadata })
}

// the window a request sends - the messages after the last compaction, the system prompt carrying it - compacted first
// when it would not fit
export async function compactConversation(a: {
  threadId: string
  messages: UIMessage[]
  system: string | undefined
  // the model's summary of `messages` (the request as it went last time, plus the instruction as a user turn)
  summarize: (messages: UIMessage[], system: string | undefined) => Promise<string>
}): Promise<{ messages: UIMessage[]; system: string | undefined }> {
  const context = useStrixLlamaStatus.getState().status?.profile?.context
  let state = useThreads.getState().threads[a.threadId]?.metadata?.[KEY] as CompactState | undefined
  let window = a.messages
  let system = a.system
  if (state) {
    const at = a.messages.findIndex((m) => m.id === state!.upto)
    if (at < 0) {
      // the message it ended at was edited away or deleted: the conversation is what it shows again
      state = undefined
      save(a.threadId, undefined)
    } else {
      window = a.messages.slice(at + 1)
      system = systemWith(a.system, state)
    }
  }
  if (!context || context <= 0) return { messages: window, system }

  // the server's own count up to the last answer it gave for this window, an estimate for what came after
  const answered = lastIndex(window, (m) => m.role === 'assistant' && usageOf(m) !== undefined)
  const known = answered >= 0 ? usageOf(window[answered])! : estimateTokens(system ?? '')
  const after = (answered >= 0 ? window.slice(answered + 1) : window).reduce((n, m) => n + estimateMessage(m), 0)
  const before = known + after
  if (before <= HIGH * context) return { messages: window, system }

  // what gets folded: the window up to its last answer; the new turn stays as it is
  const cut = lastIndex(window, (m) => m.role === 'assistant')
  if (cut < 0) return { messages: window, system }
  const head = window.slice(0, cut + 1)
  const tail = window.slice(cut + 1)
  const tid = toast.loading(i18n.t('strixllama:compact.running'))
  let next: CompactState
  try {
    // the same request as last time plus the instruction: it only fits while the conversation up to the last answer
    // leaves room for the summary
    if (known > context - 4096) throw new Error('no room left for a summary')
    const text = (await a.summarize(head, system)).trim()
    if (!text) throw new Error('empty summary')
    next = { upto: head[head.length - 1].id, text, kind: 'summary' }
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      toast.dismiss(tid)
      throw err
    }
    console.warn('[strixllama] compaction without a summary:', err)
    // the newest messages of the head that fit beside the new turn, the rest dropped
    let budget = KEEP * context - estimateTokens(a.system ?? '') - tail.reduce((n, m) => n + estimateMessage(m), 0)
    let keepFrom = head.length
    while (keepFrom > 0 && budget - estimateMessage(head[keepFrom - 1]) > 0) budget -= estimateMessage(head[--keepFrom])
    // a window starts at a user turn
    while (keepFrom < head.length && head[keepFrom].role !== 'user') keepFrom++
    if (keepFrom === 0) keepFrom = Math.min(1, head.length)
    const dropped = keepFrom + (a.messages.length - window.length)
    const earlier = state?.kind === 'summary' ? '\n\nThe summary of the part before those, which you wrote:\n\n' + state.text : ''
    next = {
      upto: head[keepFrom - 1].id,
      text: `# Earlier in this conversation\nThe first ${dropped} messages of this conversation were removed to fit ` +
        `your context window.${earlier}`,
      kind: 'dropped',
    }
    window = head.slice(keepFrom).concat(tail)
    save(a.threadId, next)
    system = systemWith(a.system, next)
    toast.warning(i18n.t('strixllama:compact.dropped', { count: dropped }), { id: tid })
    return { messages: window, system }
  }
  save(a.threadId, next)
  system = systemWith(a.system, next)
  const after2 = estimateTokens(system) + tail.reduce((n, m) => n + estimateMessage(m), 0)
  toast.success(i18n.t('strixllama:compact.done', {
    from: Math.round(before / 1000) + 'K', to: Math.max(1, Math.round(after2 / 1000)) + 'K' }), { id: tid })
  return { messages: tail, system }
}
