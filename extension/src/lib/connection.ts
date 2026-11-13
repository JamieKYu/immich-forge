// End-to-end connection check used by the options page ("Save & test") and the
// side panel's status: can we reach the server, is the token accepted, and can
// the server reach Immich? Distinguishes the failures so the UI can say which
// one to fix.

import { ForgeClient } from './forge-client'
import type { ModelsInfo } from './types'

export type ConnectionResult =
  | { ok: true; immich: boolean; models: ModelsInfo }
  | {
      ok: false
      kind: 'invalid-url' | 'unreachable' | 'unauthorized' | 'server-token-unset' | 'error'
      message: string
    }

export function isValidForgeUrl(url: string): boolean {
  try {
    const u = new URL(url)
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch {
    return false
  }
}

export async function testConnection(
  forgeUrl: string,
  forgeToken: string,
): Promise<ConnectionResult> {
  if (!isValidForgeUrl(forgeUrl)) {
    return {
      ok: false,
      kind: 'invalid-url',
      message: 'Enter a full address, e.g. http://gpu-host:8000',
    }
  }
  const client = new ForgeClient({ forgeUrl, forgeToken })

  // /health needs no token: it tells us the server is there at all.
  let immich: boolean
  try {
    immich = (await client.health()).immich
  } catch {
    return {
      ok: false,
      kind: 'unreachable',
      message: "Can't reach the Forge server at that address. Is it running?",
    }
  }

  // /models needs the token: a 401 means the token is wrong.
  try {
    const models = await client.models()
    return { ok: true, immich, models }
  } catch (e) {
    const status = (e as { status?: number }).status
    if (status === 401) {
      return { ok: false, kind: 'unauthorized', message: 'The server rejected this API token.' }
    }
    if (status === 503) {
      return {
        ok: false,
        kind: 'server-token-unset',
        message: 'The server has no FORGE_API_TOKEN set, so its API is disabled.',
      }
    }
    return { ok: false, kind: 'error', message: (e as Error).message }
  }
}
