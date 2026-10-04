import { invoke } from '@tauri-apps/api/core'
import { fetch as httpFetch } from '@tauri-apps/plugin-http'
import i18n from '@/i18n'
import { connection, useStrixLlamaStatus } from './status'

// Documents in the chat. Jan reads an attached document either into the message ("inline") or into a vector store
// through an embedding model its own llama.cpp engine serves; this build has no such engine (apply.py leaves it out),
// so every document goes into the message whole, and the model reads it with the conversation.

const noTauri = () =>
  import.meta.env.DEV && typeof (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ === 'undefined'

export const fileExtension = (name: string) => (name.includes('.') ? name.split('.').pop()!.toLowerCase() : '')

// The text of a file dropped on the chat box: its bytes go to the app, which reads them with the parser Jan's file
// picker uses (strixllama_parse_dropped in strixllama.rs). The browser preview, without the app, reads plain text.
export const parseDroppedDocument = async (file: File): Promise<string> => {
  if (noTauri()) return file.text()
  // a header value has to be Latin-1, and the app keeps letters and digits of it anyway
  return invoke<string>('strixllama_parse_dropped', new Uint8Array(await file.arrayBuffer()),
    { headers: { 'x-file-type': fileExtension(file.name).replace(/[^a-z0-9]/g, '') } })
}

// A document longer than the context the model was loaded with would be refused by the server, and, kept in the
// conversation, would make every later request of it too long as well: say so before it is sent. Counted with the
// server's own tokenizer; when the server cannot answer, the document goes and the server has the last word.
export const checkDocumentFits = async (name: string, text: string) => {
  const context = useStrixLlamaStatus.getState().status?.profile?.context
  if (!context || noTauri()) return
  let tokens: number | undefined
  try {
    // where the chat reaches the server, with its key when it asks for one (Configuration › Network)
    const { endpoint, apiKey } = connection()
    const res = await httpFetch(endpoint.replace(/\/v1$/, '') + '/tokenize', {
      method: 'POST', body: JSON.stringify({ content: text }),
      headers: { 'Content-Type': 'application/json', ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) },
    })
    if (res.ok) tokens = ((await res.json()) as { tokens?: unknown[] }).tokens?.length
  } catch { /* not running */ }
  // the rest of the conversation, the question and the answer need room too
  if (tokens !== undefined && tokens > context * 0.9)
    throw new Error(i18n.t('strixllama:attach.tooLong', {
      name, tokens: tokens.toLocaleString(), context: context.toLocaleString() }))
}

// The text of an attached document: the file picker's are read when the message is sent, dropped ones as they are
// dropped. Jan would fall back to its embedding engine when a file gives no text; here that is the end of it, with
// the reason.
export const readDocument = async (name: string, parse: () => Promise<string | undefined>) => {
  let text: string | undefined
  try {
    text = await parse()
  } catch (err) {
    const reason = err instanceof Error ? err.message : typeof err === 'string' ? err : JSON.stringify(err)
    throw new Error(i18n.t('strixllama:attach.unreadable', { name, reason }))
  }
  if (!text?.trim()) throw new Error(i18n.t('strixllama:attach.empty', { name }))
  await checkDocumentFits(name, text)
  return text
}
