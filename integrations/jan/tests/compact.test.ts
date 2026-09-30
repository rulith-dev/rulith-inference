// Tests for components/strixllama/compact.ts. Not part of the overlay's UI_FILES: tools like
// tmp/rename/test_compact.sh copy it next to the module and run it with the web app's vitest.
import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { UIMessage } from '@ai-sdk/react'

const threads: Record<string, { id: string; metadata?: Record<string, unknown> }> = {}
vi.mock('@/hooks/useThreads', () => ({
  useThreads: {
    getState: () => ({
      threads,
      updateThread: (id: string, u: { metadata: Record<string, unknown> }) => {
        threads[id] = { ...threads[id], ...u }
      },
    }),
  },
}))
let context: number | undefined = 100000
vi.mock('./status', () => ({
  useStrixLlamaStatus: { getState: () => ({ status: { profile: { context } } }) },
}))
vi.mock('sonner', () => ({
  toast: { loading: vi.fn(() => 't'), success: vi.fn(), warning: vi.fn(), dismiss: vi.fn() },
}))
vi.mock('@/i18n', () => ({ default: { t: (k: string) => k } }))

import { compactConversation, estimateTokens, type CompactState } from './compact'

const text = (id: string, role: 'user' | 'assistant', t: string, usage?: number): UIMessage =>
  ({ id, role, parts: [{ type: 'text', text: t }], ...(usage ? { metadata: { usage: { totalTokens: usage } } } : {}) }) as UIMessage

const conv = (usage: number) => [
  text('u1', 'user', 'hello'),
  text('a1', 'assistant', 'hi', 10000),
  text('u2', 'user', 'more'),
  text('a2', 'assistant', 'answer', usage),
  text('u3', 'user', 'next question'),
]

describe('compactConversation', () => {
  beforeEach(() => {
    for (const k of Object.keys(threads)) delete threads[k]
    threads.t1 = { id: 't1', metadata: {} }
    context = 100000
  })

  it('estimates CJK one token a character and other text by three', () => {
    expect(estimateTokens('你好世界')).toBe(4)
    expect(estimateTokens('abcdef')).toBe(2)
  })

  it('leaves a conversation that fits alone', async () => {
    const summarize = vi.fn()
    const r = await compactConversation({ threadId: 't1', messages: conv(50000), system: 'S', summarize })
    expect(summarize).not.toHaveBeenCalled()
    expect(r.messages.map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
    expect(r.system).toBe('S')
  })

  it('summarizes up to the last answer and keeps the new turn', async () => {
    const summarize = vi.fn(async () => 'the summary')
    const r = await compactConversation({ threadId: 't1', messages: conv(85000), system: 'S', summarize })
    expect(summarize).toHaveBeenCalledTimes(1)
    const [head, sys] = summarize.mock.calls[0] as unknown as [UIMessage[], string]
    expect(head.map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2'])
    expect(sys).toBe('S')
    expect(r.messages.map((m) => m.id)).toEqual(['u3'])
    expect(r.system).toContain('S\n\n# Earlier in this conversation')
    expect(r.system).toContain('the summary')
    const st = threads.t1.metadata?.strix_compact as CompactState
    expect(st).toEqual({ upto: 'a2', text: 'the summary', kind: 'summary' })
  })

  it('sends the same head after a summary, and folds again past the threshold', async () => {
    threads.t1.metadata = { strix_compact: { upto: 'a2', text: 'old summary', kind: 'summary' } }
    const msgs = [...conv(99990), text('a3', 'assistant', 'x', 30000), text('u4', 'user', 'q')]
    const summarize = vi.fn(async () => 'new summary')
    const r = await compactConversation({ threadId: 't1', messages: msgs, system: 'S', summarize })
    expect(summarize).not.toHaveBeenCalled()
    expect(r.messages.map((m) => m.id)).toEqual(['u3', 'a3', 'u4'])
    expect(r.system).toContain('old summary')
    // the window grew past 80%: fold what it holds up to a4, with the old summary in the system prompt
    const more = [...msgs, text('a4', 'assistant', 'y', 82000), text('u5', 'user', 'q2')]
    const r2 = await compactConversation({ threadId: 't1', messages: more, system: 'S', summarize })
    expect(summarize).toHaveBeenCalledTimes(1)
    const [head, sys] = summarize.mock.calls[0] as unknown as [UIMessage[], string]
    expect(head.map((m) => m.id)).toEqual(['u3', 'a3', 'u4', 'a4'])
    expect(sys).toContain('old summary')
    expect(r2.messages.map((m) => m.id)).toEqual(['u5'])
    expect((threads.t1.metadata?.strix_compact as CompactState).upto).toBe('a4')
  })

  it('forgets a summary whose last message is gone', async () => {
    threads.t1.metadata = { strix_compact: { upto: 'gone', text: 'old', kind: 'summary' } }
    const r = await compactConversation({ threadId: 't1', messages: conv(50000), system: 'S', summarize: vi.fn() })
    expect(r.messages).toHaveLength(5)
    expect(r.system).toBe('S')
    expect(threads.t1.metadata?.strix_compact).toBeUndefined()
  })

  it('drops the oldest messages when the summary fails, keeping the old summary in the note', async () => {
    threads.t1.metadata = { strix_compact: { upto: 'a1', text: 'old summary', kind: 'summary' } }
    const long = 'z'.repeat(90000)   // ~30000 tokens
    const msgs = [
      text('u1', 'user', 'hello'), text('a1', 'assistant', 'hi', 9000),
      text('u2', 'user', long), text('a2', 'assistant', long, 70000),
      text('u3', 'user', 'short'), text('a3', 'assistant', 'ok', 85000),
      text('u4', 'user', 'now'),
    ]
    const summarize = vi.fn(async () => { throw new Error('server said no') })
    const r = await compactConversation({ threadId: 't1', messages: msgs, system: 'S', summarize })
    // budget 50000 - system - tail: a3, u3, a2 fit, u2 does not; a window starts at a user turn
    expect(r.messages.map((m) => m.id)).toEqual(['u3', 'a3', 'u4'])
    const st = threads.t1.metadata?.strix_compact as CompactState
    expect(st.kind).toBe('dropped')
    expect(st.upto).toBe('a2')
    expect(st.text).toContain('The first 4 messages')
    expect(st.text).toContain('old summary')
  })

  it('does not ask for a summary when the last request left no room for one', async () => {
    const summarize = vi.fn(async () => 's')
    await compactConversation({ threadId: 't1', messages: conv(97000), system: 'S', summarize })
    expect(summarize).not.toHaveBeenCalled()
    expect((threads.t1.metadata?.strix_compact as CompactState).kind).toBe('dropped')
  })

  it('passes an abort through without touching the thread', async () => {
    const summarize = vi.fn(async () => { const e = new Error('stop'); e.name = 'AbortError'; throw e })
    await expect(compactConversation({ threadId: 't1', messages: conv(85000), system: 'S', summarize })).rejects.toThrow('stop')
    expect(threads.t1.metadata?.strix_compact).toBeUndefined()
  })
})
