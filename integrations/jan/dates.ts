// strixllama: today's date for the model, in the conversation where it costs no cache.
//
// A model knows the date only when it is told; asked, it guesses one from its training data (users saw "October 2024"
// in October 2026). Jan's default system prompt ended with "Current date: {{current_date}}", but the system prompt is
// the first thing in every request: when the date changed at midnight, no conversation's cached prefix matched any
// more, and a long one was read again from the start (apply.py brand_text removes that line). Here a user message
// carries the date instead - the first message of a conversation and the first of each later day - taken from the
// message's own time, so a message reads the same each time the conversation is sent again and the history stays
// cached; only the newest message, which is read anyway, can bring a new date.
import { type UIMessage } from '@ai-sdk/react'

// when each message was first seen, for those that do not carry their creation time (a message just sent)
const firstSeen = new Map<string, number>()

function messageTime(m: UIMessage): Date {
  const meta = m.metadata as { createdAt?: Date | string | number } | undefined
  if (meta?.createdAt !== undefined) {
    const t = new Date(meta.createdAt)
    if (!isNaN(t.getTime())) {
      return t
    }
  }
  let seen = firstSeen.get(m.id)
  if (seen === undefined) {
    seen = Date.now()
    firstSeen.set(m.id, seen)
  }
  return new Date(seen)
}

export function dateNote(d: Date): string {
  const day = d.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
  return `[Today is ${day}.]`
}

// the messages with a date note before the text of each user message that starts a new local day
export function withDateNotes(messages: UIMessage[]): UIMessage[] {
  let last = ''
  return messages.map((m) => {
    if (m.role !== 'user') {
      return m
    }
    const when = messageTime(m)
    const day = when.toDateString()
    if (day === last) {
      return m
    }
    last = day
    const note = dateNote(when)
    const parts = [...m.parts]
    const i = parts.findIndex((p) => p.type === 'text')
    if (i >= 0) {
      const p = parts[i] as { type: 'text'; text: string }
      parts[i] = { ...p, text: `${note}\n\n${p.text}` }
    } else {
      parts.unshift({ type: 'text', text: note } as UIMessage['parts'][number])
    }
    return { ...m, parts }
  })
}
