// Tests for components/strixllama/dates.ts. Not part of the overlay's UI_FILES: apply_test_overlay.py copies it next
// to the module, where the web app's vitest runs it.
import { describe, it, expect } from 'vitest'
import type { UIMessage } from '@ai-sdk/react'

import { dateNote, withDateNotes } from './dates'

const user = (id: string, t: string, at?: Date): UIMessage =>
  ({ id, role: 'user', parts: [{ type: 'text', text: t }], ...(at ? { metadata: { createdAt: at } } : {}) }) as UIMessage
const assistant = (id: string, t: string): UIMessage => ({ id, role: 'assistant', parts: [{ type: 'text', text: t }] }) as UIMessage
const textOf = (m: UIMessage) => (m.parts[0] as { type: 'text'; text: string }).text

describe('withDateNotes', () => {
  const d1 = new Date(2026, 9, 24, 10, 0)
  const d1b = new Date(2026, 9, 24, 23, 30)
  const d2 = new Date(2026, 9, 25, 8, 15)

  it('puts the date before the first user message only, on one day', () => {
    const out = withDateNotes([user('a', 'hi', d1), assistant('b', 'hello'), user('c', 'more', d1b)])
    expect(textOf(out[0])).toBe(`${dateNote(d1)}\n\nhi`)
    expect(textOf(out[1])).toBe('hello')
    expect(textOf(out[2])).toBe('more')
  })

  it('adds the date again where a later day starts', () => {
    const out = withDateNotes([user('a', 'hi', d1), assistant('b', 'hello'), user('c', 'next day', d2)])
    expect(textOf(out[2])).toBe(`${dateNote(d2)}\n\nnext day`)
    expect(dateNote(d2)).toBe('[Today is Sunday, October 25, 2026.]')
  })

  it('gives a message the same text on every request (the cached prefix stays)', () => {
    const msgs = [user('a', 'hi', d1), assistant('b', 'hello'), user('c', 'x')]
    const first = withDateNotes(msgs).map(textOf)
    const again = withDateNotes([...msgs, assistant('d', 'y'), user('e', 'z')]).map(textOf)
    expect(again.slice(0, 3)).toEqual(first)
  })

  it('leaves the stored messages alone', () => {
    const msgs = [user('a', 'hi', d1)]
    withDateNotes(msgs)
    expect(textOf(msgs[0])).toBe('hi')
  })

  it('adds a text part to a user message without one', () => {
    const img = { id: 'i', role: 'user', parts: [{ type: 'file', mediaType: 'image/png', url: 'data:' }], metadata: { createdAt: d1 } } as unknown as UIMessage
    const out = withDateNotes([img])
    expect(out[0].parts[0]).toEqual({ type: 'text', text: dateNote(d1) })
    expect(out[0].parts[1].type).toBe('file')
  })
})
