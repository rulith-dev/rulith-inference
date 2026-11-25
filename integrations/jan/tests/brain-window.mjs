// Run with node after applying the overlay. All fixture origins resolve to loopback.
import assert from 'node:assert/strict'
import { spawn, execFileSync } from 'node:child_process'
import { readFileSync, mkdtempSync, rmSync } from 'node:fs'
import http from 'node:http'
import https from 'node:https'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const web = fileURLToPath(new URL('../../../src/jan/web-app/', import.meta.url))
const require = createRequire(path.join(web, 'package.json'))
const entry = path.join(web, 'src/components/strixllama/brainBridge.ts')
const { outputFiles } = await require('esbuild').build({
  entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'iife', globalName: 'host',
  define: { 'process.env.NODE_ENV': '"production"' },
})
const profile = mkdtempSync(path.join(tmpdir(), 'brain-window-'))
const parent = 'http://tauri.localhost'
const child = 'https://brain-ui.rulith.invalid'
const wrong = 'https://other.rulith.invalid'
const page = '<!doctype html><meta charset=utf-8><body>'
function childDocument() {
  return page + `<script>
    const tell = (type, data={}) => parent.postMessage({type,...data}, '${parent}')
    const hello = () => parent.postMessage({v:1,type:'hello'}, '${parent}')
    addEventListener('message', e => {
      if (e.origin !== '${parent}' || e.source !== parent) return
      if (e.data.type === 'ping') return tell('pong')
      if (e.data.type !== 'connect') return
      tell('connected')
      const port = e.ports[0]
      port.onmessage = e => tell('reply', {kind:e.data.type})
      hello()
      port.postMessage({v:1,type:'request',id:'047a9bc7-3259-4b76-bb25-c9112e221e88',op:'status',body:{}})
    })
    hello()
    tell('sent')
  </script>`
}
function parentDocument() {
  return page + '<script>window.onerror=(message)=>navigator.sendBeacon("/stage",String(message))</script><script>'
    + outputFiles[0].text + '</script><script>' + `
    let resolveMessage
    const messages = []
    window.addEventListener('message', e => {
      if (!['sent','connected','reply','pong'].includes(e.data?.type)) return
      messages.push(e)
      resolveMessage?.()
    })
    async function receive(type, frame) {
      navigator.sendBeacon('/stage', 'waiting for ' + type + ' at ' + frame.src)
      while (true) {
        const i = messages.findIndex(e => e.data.type === type && e.source === frame.contentWindow)
        if (i !== -1) return messages.splice(i, 1)[0].data
        await new Promise(r => { resolveMessage = r })
      }
    }
    function frame(url) {
      const el = document.createElement('iframe')
      el.sandbox = 'allow-scripts allow-same-origin'
      el.src = url
      return el
    }
    async function ping(el, origin) {
      el.contentWindow.postMessage({type:'ping'}, origin)
      await receive('pong', el)
    }
    const check = (ok, why) => { if (!ok) throw Error(why) }
    async function run() {
      const expected = frame('${wrong}/index.html')
      const sibling = frame('${child}/index.html')
      const rejected = new host.BrainBridge(expected, host.requiredCaps, () => { throw Error('foreign request') }, () => {})
      window.addEventListener('message', rejected.helloFrom)
      expected.onload = rejected.documentLoaded
      document.body.append(expected, sibling)
      await receive('sent', expected)
      await receive('sent', sibling)
      await ping(expected, '${wrong}')
      await ping(sibling, '${child}')
      check(!messages.some(e => e.data.type === 'connected'), 'foreign connection')
      rejected.close('navigate')
      window.removeEventListener('message', rejected.helloFrom)
      expected.remove(); sibling.remove()
      const trusted = frame('${child}/index.html')
      let called, reply, revoked
      const admitted = new Promise(r => { called = r })
      const b = new host.BrainBridge(trusted, host.requiredCaps, async (req, callback) => { reply = callback; called(req) }, why => { revoked = why })
      window.addEventListener('message', b.helloFrom)
      trusted.onload = b.documentLoaded
      document.body.append(trusted)
      const req = await admitted
      await receive('connected', trusted)
      await ping(trusted, '${child}')
      check(!messages.some(e => e.data.type === 'connected'), 'replayed connection')
      b.close('navigate')
      reply({v:1,type:'result',id:req.id,body:{}})
      check((await receive('reply', trusted)).kind === 'close', 'late reply')
      await ping(trusted, '${child}')
      check(revoked === 'navigate' && !messages.some(e => e.data.kind === 'result'), 'generation leak')
      document.body.textContent = 'PASS: origin, source, replay, late reply'
    }
    run().catch(e => { document.body.textContent = 'FAIL: ' + e.message })
      .then(() => fetch('/result', {method:'POST',body:document.body.textContent}))
  </script>`
}
const serve = handler => (_req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' })
  res.end(handler())
}
const servers = []
async function bind(server) {
  servers.push(server)
  await new Promise((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve))
  return server.address().port
}
try {
  const cert = path.join(profile, 'test.crt')
  const key = path.join(profile, 'test.key')
  execFileSync('openssl', ['req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=brain-ui.rulith.invalid',
    '-keyout',key,'-out',cert], { stdio: 'ignore', windowsHide: true })
  let report
  const result = new Promise(resolve => { report = resolve })
  let stage = 'starting the browser'
  let requests = 0
  let documents = 0
  const httpPort = await bind(http.createServer((req, res) => {
    requests++
    if (req.url === '/stage') {
      req.setEncoding('utf8')
      req.on('data', s => { stage = s.slice(0, 160) })
      return res.end()
    }
    if (req.method !== 'POST' || req.url !== '/result') return serve(parentDocument)(req, res)
    req.setEncoding('utf8')
    let verdict = ''
    req.on('data', data => { if (verdict.length < 256) verdict += data })
    req.on('end', () => { res.end(); report(verdict) })
  }))
  const httpsPort = await bind(https.createServer({ cert: readFileSync(cert), key: readFileSync(key) }, (req, res) => {
    documents++
    serve(childDocument)(req, res)
  }))
  const browser = process.env.BRAIN_TEST_BROWSER || path.join(process.env['ProgramFiles(x86)'], 'Microsoft/Edge/Application/msedge.exe')
  const proc = spawn(browser, ['--headless','--disable-gpu','--in-process-gpu','--use-angle=swiftshader',
    '--disable-background-networking','--disable-component-update','--disable-extensions',
    '--no-first-run','--no-proxy-server','--ignore-certificate-errors',
    '--user-data-dir=' + profile,
    '--host-resolver-rules=MAP tauri.localhost 127.0.0.1:' + httpPort + ', MAP *.rulith.invalid 127.0.0.1:' + httpsPort + ', MAP * ~NOTFOUND',
    parent + '/brain'], { windowsHide: true, stdio: 'ignore' })
  const exited = new Promise(resolve => {
    proc.once('error', error => resolve(error.message))
    proc.once('exit', code => resolve('Browser exited: ' + code))
  })
  let timer
  try {
    const timeout = new Promise(resolve => { timer = setTimeout(() => resolve([stage, requests, documents].join(' / ')), 30000) })
    assert.equal(await Promise.race([result, timeout, exited]), 'PASS: origin, source, replay, late reply')
  } finally {
    clearTimeout(timer)
    proc.kill()
    await exited
  }
  console.log('PASS: origin, source, replay, late reply in a browser window')
} finally {
  for (const server of servers) server.close()
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
}
