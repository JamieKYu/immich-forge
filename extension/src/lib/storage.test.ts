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
})

describe('saveSettings', () => {
  it('writes under the forgeSettings key in local storage', async () => {
    const settings = {
      forgeUrl: 'http://h:8000',
      forgeToken: 'tok',
      operations: DEFAULT_OPERATIONS,
    }
    await saveSettings(settings)
    expect(chrome.storage.local.set).toHaveBeenCalledWith({ forgeSettings: settings })
  })
})
