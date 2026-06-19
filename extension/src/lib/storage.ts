import type { Settings } from './types'

const KEY = 'forgeSettings'
const DEFAULTS: Settings = { forgeUrl: '', forgeToken: '' }

// Use local (not sync) storage: it holds the Forge bearer token.
export async function loadSettings(): Promise<Settings> {
  const out = await chrome.storage.local.get(KEY)
  return { ...DEFAULTS, ...(out[KEY] as Partial<Settings> | undefined) }
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.local.set({ [KEY]: settings })
}
