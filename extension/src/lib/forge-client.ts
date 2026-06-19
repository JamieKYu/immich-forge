// HTTP client for the Forge server. Runs inside the service worker so that all
// cross-origin calls go through the extension's host_permissions and job state
// survives the side panel being closed/reopened.

import type {
  AcceptResponse,
  ForgeOperations,
  ImmichAsset,
  JobInfo,
  Settings,
} from './types'

export class ForgeClient {
  constructor(private settings: Settings) {}

  private get base() {
    return this.settings.forgeUrl.replace(/\/$/, '')
  }

  private headers(extra: Record<string, string> = {}) {
    const h: Record<string, string> = { ...extra }
    if (this.settings.forgeToken) {
      h['Authorization'] = `Bearer ${this.settings.forgeToken}`
    }
    return h
  }

  async health(): Promise<{ ok: boolean; immich: boolean; gpu: unknown }> {
    const r = await fetch(`${this.base}/health`)
    if (!r.ok) throw new Error(`health ${r.status}`)
    return r.json()
  }

  async asset(assetId: string): Promise<ImmichAsset> {
    let r: Response
    try {
      r = await fetch(`${this.base}/immich/asset/${assetId}`, {
        headers: this.headers(),
      })
    } catch (e) {
      // Network-level failure: bad Forge URL, server down, mixed-content, etc.
      throw new Error(`can't reach Forge at ${this.base} (${(e as Error).message})`)
    }
    if (!r.ok) {
      let detail = r.statusText
      try {
        detail = (await r.json())?.detail ?? detail
      } catch {
        /* non-json body */
      }
      throw new Error(`HTTP ${r.status}: ${detail}`)
    }
    return r.json()
  }

  async thumbnailDataUrl(assetId: string): Promise<string> {
    const r = await fetch(`${this.base}/immich/thumbnail/${assetId}`, {
      headers: this.headers(),
    })
    if (!r.ok) throw new Error(`thumbnail ${r.status}`)
    return blobToDataUrl(await r.blob())
  }

  async forge(assetId: string, operations: ForgeOperations): Promise<JobInfo> {
    const r = await fetch(`${this.base}/forge`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ asset_id: assetId, operations }),
    })
    if (!r.ok) throw new Error(`forge ${r.status}: ${await r.text()}`)
    return r.json()
  }

  async job(jobId: string): Promise<JobInfo> {
    const r = await fetch(`${this.base}/forge/${jobId}`, { headers: this.headers() })
    if (!r.ok) throw new Error(`job ${r.status}`)
    return r.json()
  }

  async resultDataUrl(jobId: string): Promise<string> {
    const r = await fetch(`${this.base}/forge/${jobId}/result`, {
      headers: this.headers(),
    })
    if (!r.ok) throw new Error(`result ${r.status}`)
    return blobToDataUrl(await r.blob())
  }

  async accept(jobId: string): Promise<AcceptResponse> {
    const r = await fetch(`${this.base}/forge/${jobId}/accept`, {
      method: 'POST',
      headers: this.headers(),
    })
    if (!r.ok) throw new Error(`accept ${r.status}: ${await r.text()}`)
    return r.json()
  }
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}
