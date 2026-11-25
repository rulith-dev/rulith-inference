// Run against the integration fixture with its verified offline UI installed and
// startup model loading disabled. Pass the local driver port and desktop binary.
import assert from 'node:assert/strict'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'

const [port, application] = process.argv.slice(2)
if (!/^\d+$/.test(port || '') || Number(port) > 65535 || !path.isAbsolute(application || '')) {
  throw Error('Usage: node brain-native-window.mjs <local driver port> <absolute desktop executable>')
}
const endpoint = 'http://127.0.0.1:' + port
async function command(method, route, body) {
  const res = await fetch(endpoint + route, { method, redirect: 'error',
    headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  const { value } = await res.json()
  if (!res.ok) throw Error('Desktop driver: ' + (value?.error || res.status))
  return value
}
const session = await command('POST', '/session', { capabilities: { alwaysMatch: {
  'tauri:options': { application },
} } })
const base = '/session/' + encodeURIComponent(session.sessionId)
const post = (route, body) => command('POST', base + route, body)
const element = 'element-6066-11e4-a52e-4f735466cecf'
async function find(selector) {
  const deadline = Date.now() + 30000
  while (Date.now() < deadline) {
    const found = await post('/elements', { using: 'css selector', value: selector })
    if (found.length) return found[0]
    await delay(100)
  }
  throw Error('Desktop fixture did not show ' + selector)
}
// Use the same harmless status read from the main document and the actual frame.
// Keep status contents and exception details inside the test window.
const probe = `
  const done = arguments[arguments.length - 1]
  const api = window.__TAURI_INTERNALS__
  if (!api || typeof api.invoke !== 'function') { done('unavailable'); return }
  api.invoke('brain_host_mode', {op:'mode_status',data:{}})
    .then(s => done(s && typeof s.state === 'string' ? 'allowed' : 'unexpected'), () => done('rejected'))
`
try {
  assert.equal(await post('/execute/async', { script: probe, args: [] }), 'allowed')
  const link = await find('a[href="/brain"]')
  await post('/element/' + link[element] + '/click', {})
  const frame = await find('iframe[src="https://brain-ui.rulith.invalid/index.html"]')
  await post('/frame', { id: frame })
  assert.equal(await post('/execute/sync', { script: 'return location.origin', args: [] }), 'https://brain-ui.rulith.invalid')
  const outcome = await post('/execute/async', { script: probe, args: [] })
  assert.ok(['unavailable','rejected'].includes(outcome), 'Brain frame reached the native status command')
  console.log(outcome === 'rejected' ? 'Native status call rejected in frame.' : 'Native invoke unavailable in frame.')
} finally {
  await command('DELETE', base)
}
