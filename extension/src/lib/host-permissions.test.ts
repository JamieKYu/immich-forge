import { describe, expect, it, vi } from 'vitest'
import {
  ensureForgeAccess,
  forgeOriginPattern,
  hasForgeAccess,
} from './host-permissions'

describe('forgeOriginPattern', () => {
  it('builds an origin match pattern from a URL', () => {
    expect(forgeOriginPattern('http://gpu-host:8000')).toBe('http://gpu-host:8000/*')
  })

  it('drops path/query, keeping just the origin', () => {
    expect(forgeOriginPattern('https://h:8443/forge?x=1')).toBe('https://h:8443/*')
  })

  it('returns null for an empty or invalid URL', () => {
    expect(forgeOriginPattern('')).toBeNull()
    expect(forgeOriginPattern('not a url')).toBeNull()
  })
})

describe('ensureForgeAccess', () => {
  it('requests the scoped origin and resolves the grant result', async () => {
    vi.mocked(chrome.permissions.request).mockResolvedValue(true)
    const ok = await ensureForgeAccess('http://gpu-host:8000')
    expect(ok).toBe(true)
    expect(chrome.permissions.request).toHaveBeenCalledWith({
      origins: ['http://gpu-host:8000/*'],
    })
  })

  it('returns false without requesting for an invalid URL', async () => {
    expect(await ensureForgeAccess('')).toBe(false)
    expect(chrome.permissions.request).not.toHaveBeenCalled()
  })

  it('returns false if the request throws', async () => {
    vi.mocked(chrome.permissions.request).mockRejectedValue(new Error('no gesture'))
    expect(await ensureForgeAccess('http://gpu-host:8000')).toBe(false)
  })
})

describe('hasForgeAccess', () => {
  it('reports whether the origin is already granted', async () => {
    vi.mocked(chrome.permissions.contains).mockResolvedValue(true)
    expect(await hasForgeAccess('http://gpu-host:8000')).toBe(true)
    expect(chrome.permissions.contains).toHaveBeenCalledWith({
      origins: ['http://gpu-host:8000/*'],
    })
  })

  it('returns false for an invalid URL', async () => {
    expect(await hasForgeAccess('')).toBe(false)
  })
})
