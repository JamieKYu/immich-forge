import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ForgeClient } from './forge-client'
import { DEFAULT_OPERATIONS, type Settings } from './types'

function settings(over: Partial<Settings> = {}): Settings {
  return {
    forgeUrl: 'http://gpu-host:8000',
    forgeToken: 'secret',
    operations: DEFAULT_OPERATIONS,
    ...over,
  }
}

function jsonResponse(body: unknown, init: Partial<Response> = {}): Response {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
})

describe('base URL handling', () => {
  it('strips a trailing slash from the configured Forge URL', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: 'a1' }))
    const client = new ForgeClient(settings({ forgeUrl: 'http://gpu-host:8000/' }))
    await client.asset('a1')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://gpu-host:8000/immich/asset/a1',
      expect.anything(),
    )
  })
})

describe('auth header', () => {
  it('attaches a bearer token when one is configured', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: 'a1' }))
    await new ForgeClient(settings({ forgeToken: 'secret' })).asset('a1')
    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers.Authorization).toBe('Bearer secret')
  })

  it('omits the Authorization header when no token is set', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: 'a1' }))
    await new ForgeClient(settings({ forgeToken: '' })).asset('a1')
    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers.Authorization).toBeUndefined()
  })
})

describe('asset() error handling', () => {
  it('surfaces the JSON `detail` on an HTTP error', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ detail: 'asset not found' }, { ok: false, status: 404 }),
    )
    await expect(new ForgeClient(settings()).asset('missing')).rejects.toThrow(
      /HTTP 404: asset not found/,
    )
  })

  it('wraps a network-level failure with a reachability hint', async () => {
    fetchMock.mockRejectedValue(new Error('Failed to fetch'))
    await expect(new ForgeClient(settings()).asset('a1')).rejects.toThrow(
      /can't reach Forge at http:\/\/gpu-host:8000/,
    )
  })
})

describe('forge()', () => {
  it('POSTs asset id + operations as JSON', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ job_id: 'j1' }))
    const ops = { ...DEFAULT_OPERATIONS, colorize: true }
    await new ForgeClient(settings()).forge('a1', ops)

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://gpu-host:8000/forge')
    expect(init.method).toBe('POST')
    expect(init.headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(init.body)).toEqual({ asset_id: 'a1', operations: ops })
  })
})

describe('health()', () => {
  it('returns parsed status on success', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ok: true, immich: true, gpu: {} }))
    const out = await new ForgeClient(settings()).health()
    expect(out.ok).toBe(true)
  })

  it('throws on a non-ok response', async () => {
    fetchMock.mockResolvedValue(jsonResponse({}, { ok: false, status: 500 }))
    await expect(new ForgeClient(settings()).health()).rejects.toThrow(/health 500/)
  })
})
