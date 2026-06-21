import { useEffect, useRef, useState } from 'react'
import type { ForgeClient } from '../lib/forge-client'

// Loads the full-resolution original for a 1:1-quality preview, and falls back
// to the thumbnail if the original can't be fetched or the browser can't render
// it (e.g. a HEIC/RAW original). Returns { src, onError } to wire to an <img>.
export function useAssetImage(client: ForgeClient, assetId: string) {
  const [src, setSrc] = useState<string | null>(null)
  const usedThumb = useRef(false)

  useEffect(() => {
    let alive = true
    usedThumb.current = false
    const thumb = () => {
      usedThumb.current = true
      client.thumbnailDataUrl(assetId).then((s) => alive && setSrc(s)).catch(() => {})
    }
    client
      .originalDataUrl(assetId)
      .then((s) => alive && setSrc(s))
      .catch(thumb) // original fetch failed outright → thumbnail
    return () => {
      alive = false
    }
  }, [assetId])

  // <img> couldn't decode the original (non-web format) → swap to thumbnail.
  const onError = () => {
    if (usedThumb.current) return
    usedThumb.current = true
    client.thumbnailDataUrl(assetId).then(setSrc).catch(() => {})
  }

  return { src, onError }
}
