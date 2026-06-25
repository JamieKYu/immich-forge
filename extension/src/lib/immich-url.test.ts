import { describe, expect, it } from 'vitest'
import { parseAssetId } from './immich-url'

const UUID = '0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9'

describe('parseAssetId', () => {
  it('parses the main timeline viewer route', () => {
    expect(parseAssetId(`http://immich.local:2283/photos/${UUID}`)).toBe(UUID)
  })

  it('parses the album viewer route (/photo/<uuid>)', () => {
    expect(
      parseAssetId(`http://immich.local/albums/abc/photo/${UUID}`),
    ).toBe(UUID)
  })

  it('parses the shared-link viewer route', () => {
    expect(parseAssetId(`https://share.example/share/KEY/photo/${UUID}`)).toBe(UUID)
  })

  it('lowercases the returned id', () => {
    const upper = UUID.toUpperCase()
    expect(parseAssetId(`http://h/photos/${upper}`)).toBe(UUID)
  })

  it('ignores host differences (keys off the path UUID)', () => {
    expect(parseAssetId(`http://immich-server:2283/photos/${UUID}`)).toBe(UUID)
  })

  it('returns null for a non-asset page', () => {
    expect(parseAssetId('http://immich.local/albums/abc')).toBeNull()
  })

  it('returns null for a malformed UUID', () => {
    expect(parseAssetId('http://immich.local/photos/not-a-uuid')).toBeNull()
  })

  it('returns null for junk / undefined input', () => {
    expect(parseAssetId('not a url')).toBeNull()
    expect(parseAssetId(undefined)).toBeNull()
    expect(parseAssetId('')).toBeNull()
  })
})
