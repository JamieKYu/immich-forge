import { DEFAULT_OPERATIONS, type Settings } from './types'

const KEY = 'forgeSettings'
const DEFAULTS: Settings = {
  forgeUrl: '',
  forgeToken: '',
  operations: DEFAULT_OPERATIONS,
}

// Use local (not sync) storage: it holds the Forge bearer token.
export async function loadSettings(): Promise<Settings> {
  const out = await chrome.storage.local.get(KEY)
  const stored = out[KEY] as Partial<Settings> | undefined
  return {
    ...DEFAULTS,
    ...stored,
    // Deep-merge operations so older saved settings (or partial ones) still
    // get any newly added operation fields.
    operations: { ...DEFAULT_OPERATIONS, ...stored?.operations },
  }
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.local.set({ [KEY]: settings })
}
