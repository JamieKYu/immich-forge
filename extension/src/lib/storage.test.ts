import { describe, expect, it, vi } from 'vitest'
import { loadSettings, saveSettings } from './storage'
import { DEFAULT_OPERATIONS } from './types'

describe('loadSettings', () => {
  it('returns defaults when nothing is stored', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({})
    const s = await loadSettings()
    expect(s.forgeUrl).toBe('')
    expect(s.forgeToken).toBe('')
    expect(s.operations).toEqual(DEFAULT_OPERATIONS)
  })

  it('overlays stored values onto defaults', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { forgeUrl: 'http://h:8000', forgeToken: 'tok' },
    })
    const s = await loadSettings()
    expect(s.forgeUrl).toBe('http://h:8000')
    expect(s.forgeToken).toBe('tok')
    // operations absent from storage -> filled from defaults
    expect(s.operations).toEqual(DEFAULT_OPERATIONS)
  })

  it('deep-merges partial operations with operation defaults', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { operations: { colorize: true } },
    })
    const s = await loadSettings()
    expect(s.operations.colorize).toBe(true)
    // newly-added/omitted fields still come from defaults
    expect(s.operations.upscale).toBe(DEFAULT_OPERATIONS.upscale)
    expect(s.operations.upscale_factor).toBe(DEFAULT_OPERATIONS.upscale_factor)
  })

  it('starts new installs in Auto mode', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({})
    expect((await loadSettings()).mode).toBe('auto')
  })

  it('keeps existing users (settings saved before Auto mode) in Custom', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { forgeUrl: 'http://h:8000', forgeToken: 'tok', version: 2 },
    })
    expect((await loadSettings()).mode).toBe('custom')
  })

  it('keeps a mode the user chose', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { forgeUrl: 'http://h:8000', mode: 'auto' },
    })
    expect((await loadSettings()).mode).toBe('auto')
  })

  it('drops a stale low_light from pre-v2 settings when denoise was off', async () => {
    // v1 hid the Brighten checkbox while denoise was off, so it never ran.
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { operations: { denoise: false, low_light: true } },
    })
    const s = await loadSettings()
    expect(s.operations.low_light).toBe(false)
    expect(s.version).toBe(2)
  })

  it('keeps low_light from pre-v2 settings when denoise was on', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { operations: { denoise: true, low_light: true } },
    })
    expect((await loadSettings()).operations.low_light).toBe(true)
  })

  it('keeps brighten-only defaults saved by v2', async () => {
    vi.mocked(chrome.storage.local.get).mockResolvedValue({
      forgeSettings: { version: 2, operations: { denoise: false, low_light: true } },
    })
    expect((await loadSettings()).operations.low_light).toBe(true)
  })
})

describe('saveSettings', () => {
  it('writes under the forgeSettings key in local storage', async () => {
    const settings = {
      forgeUrl: 'http://h:8000',
      forgeToken: 'tok',
      operations: DEFAULT_OPERATIONS,
      mode: 'auto' as const,
    }
    await saveSettings(settings)
    expect(chrome.storage.local.set).toHaveBeenCalledWith({
      forgeSettings: { ...settings, version: 2 },
    })
  })
})
