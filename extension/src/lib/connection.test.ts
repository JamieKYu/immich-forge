import { beforeEach, describe, expect, it, vi } from 'vitest'
import { isValidForgeUrl, testConnection } from './connection'

function response(status: number, body: unknown = {}): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: '',
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
})

describe('isValidForgeUrl', () => {
  it('accepts http(s) URLs only', () => {
    expect(isValidForgeUrl('http://gpu-host:8000')).toBe(true)
    expect(isValidForgeUrl('https://forge.example')).toBe(true)
    expect(isValidForgeUrl('gpu-host:8000')).toBe(false)
    expect(isValidForgeUrl('ftp://x')).toBe(false)
    expect(isValidForgeUrl('')).toBe(false)
  })
})

describe('testConnection', () => {
  it('rejects an invalid URL without fetching', async () => {
    const r = await testConnection('nope', 't')
    expect(r).toMatchObject({ ok: false, kind: 'invalid-url' })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('reports an unreachable server', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    expect(await testConnection('http://h:8000', 't')).toMatchObject({
      ok: false,
      kind: 'unreachable',
    })
  })

  it('reports a rejected token', async () => {
    fetchMock
      .mockResolvedValueOnce(response(200, { ok: true, immich: true }))
      .mockResolvedValueOnce(response(401))
    expect(await testConnection('http://h:8000', 'bad')).toMatchObject({
      ok: false,
      kind: 'unauthorized',
    })
  })

  it('reports a server with no token configured', async () => {
    fetchMock
      .mockResolvedValueOnce(response(200, { ok: true, immich: true }))
      .mockResolvedValueOnce(response(503))
    expect(await testConnection('http://h:8000', 't')).toMatchObject({
      ok: false,
      kind: 'server-token-unset',
    })
  })

  it('returns Immich reachability and capabilities on success', async () => {
    fetchMock
      .mockResolvedValueOnce(response(200, { ok: true, immich: false }))
      .mockResolvedValueOnce(response(200, { dejpeg: 'fbcnn' }))
    const r = await testConnection('http://h:8000', 'tok')
    expect(r).toEqual({ ok: true, immich: false, models: { dejpeg: 'fbcnn' } })
    expect(fetchMock).toHaveBeenLastCalledWith('http://h:8000/models', {
      headers: { Authorization: 'Bearer tok' },
    })
  })
})
