import { DEFAULT_OPERATIONS, type Settings } from './types'

export const SETTINGS_KEY = 'forgeSettings'
const KEY = SETTINGS_KEY
// v2: "Brighten dark photo" became its own switch. Before, low_light only took
// effect alongside denoise (its checkbox was hidden when denoise was off), so
// a stale low_light=true under denoise=false must not start brightening.
export const SETTINGS_VERSION = 2
const DEFAULTS: Settings = {
  forgeUrl: '',
  forgeToken: '',
  operations: DEFAULT_OPERATIONS,
  version: SETTINGS_VERSION,
}

// Use local (not sync) storage: it holds the Forge bearer token.
export async function loadSettings(): Promise<Settings> {
  const out = await chrome.storage.local.get(KEY)
  const stored = out[KEY] as Partial<Settings> | undefined
  // Deep-merge operations so older saved settings (or partial ones) still
  // get any newly added operation fields.
  const operations = { ...DEFAULT_OPERATIONS, ...stored?.operations }
  if (stored && (stored.version ?? 1) < 2 && !operations.denoise) {
    operations.low_light = false
  }
  return { ...DEFAULTS, ...stored, operations, version: SETTINGS_VERSION }
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.local.set({ [KEY]: { ...settings, version: SETTINGS_VERSION } })
}

// Run `cb` whenever settings are saved from any extension page (e.g. the
// options page while the side panel is open). Returns an unsubscribe function.
export function onSettingsChanged(cb: (s: Settings) => void): () => void {
  const listener = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
    if (area === 'local' && KEY in changes) void loadSettings().then(cb)
  }
  chrome.storage.onChanged.addListener(listener)
  return () => chrome.storage.onChanged.removeListener(listener)
}
