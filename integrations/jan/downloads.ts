import { invoke } from '@tauri-apps/api/core'
import { revealItemInDir } from '@tauri-apps/plugin-opener'
import { toast } from 'sonner'
import i18n from '@/i18n'

// Downloads the page starts (issue #5): the download button of a code block in an answer, a table's CSV, the Logs
// page's log. Each makes a blob URL, clicks a link with `download` on it and revokes the URL right away - which in a
// browser saves the file and in this app's webview saved it nowhere anyone could see. Here the click hands the bytes
// to the app instead (strixllama_save_download in strixllama.rs), which writes them into the Downloads folder, and a
// toast says where, with a button to show the file. The blobs are kept by URL as they are made (fetch here is the
// HTTP plugin's, which cannot read a blob URL), and a URL being saved is revoked only once its bytes are read.

const inApp = () => typeof (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== 'undefined'

const blobs = new Map<string, Blob>()
const reading = new Set<string>()
const create = URL.createObjectURL.bind(URL)
const revoke = URL.revokeObjectURL.bind(URL)

// what XHR still reads when a blob was made before this module ran
const bytesOf = (href: string) =>
  new Promise<ArrayBuffer>((resolve, reject) => {
    const x = new XMLHttpRequest()
    x.open('GET', href)
    x.responseType = 'arraybuffer'
    x.onload = () => resolve(x.response as ArrayBuffer)
    x.onerror = () => reject(new Error('the file could not be read'))
    x.send()
  })

const save = async (href: string, name: string) => {
  try {
    const blob = blobs.get(href)
    const bytes = new Uint8Array(blob ? await blob.arrayBuffer() : await bytesOf(href))
    const path = await invoke<string>('strixllama_save_download', bytes, {
      headers: { 'x-file-name': encodeURIComponent(name || 'download') },
    })
    toast.success(i18n.t('strixllama:download.saved', { path }), {
      action: { label: i18n.t('strixllama:download.show'), onClick: () => void revealItemInDir(path) },
    })
  } catch (err) {
    toast.error(i18n.t('strixllama:download.failed', { reason: err instanceof Error ? err.message : String(err) }))
  } finally {
    reading.delete(href)
    blobs.delete(href)
    revoke(href)
  }
}

if (inApp() && !(window as unknown as { __strixDownloads?: boolean }).__strixDownloads) {
  ;(window as unknown as { __strixDownloads?: boolean }).__strixDownloads = true
  URL.createObjectURL = (obj: Blob | MediaSource) => {
    const url = create(obj)
    if (obj instanceof Blob) blobs.set(url, obj)
    return url
  }
  URL.revokeObjectURL = (url: string) => {
    if (reading.has(url)) return
    blobs.delete(url)
    revoke(url)
  }
  const click = HTMLAnchorElement.prototype.click
  HTMLAnchorElement.prototype.click = function (this: HTMLAnchorElement) {
    if (this.hasAttribute('download') && this.href.startsWith('blob:')) {
      reading.add(this.href)
      void save(this.href, this.getAttribute('download') || '')
      return
    }
    return click.call(this)
  }
}
