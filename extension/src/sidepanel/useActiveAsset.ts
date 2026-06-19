import { useEffect, useState } from 'react'
import { parseAssetId } from '../lib/immich-url'

// The asset id of the photo in the active tab.
//   undefined = still resolving
//   null      = active tab is not an Immich photo page
//   string    = asset id to forge
export function useActiveAsset(): string | null | undefined {
  const [assetId, setAssetId] = useState<string | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false

    // Resolve immediately from the active tab so the panel is correct on open,
    // independent of service-worker timing.
    chrome.tabs
      .query({ active: true, currentWindow: true })
      .then(([tab]) => {
        if (!cancelled) setAssetId(parseAssetId(tab?.url))
      })
      .catch(() => !cancelled && setAssetId(null))

    // The service worker mirrors the active tab's asset id (incl. SPA nav) into
    // session storage; follow it for live updates while the panel is open.
    const onChange = (
      changes: Record<string, chrome.storage.StorageChange>,
      area: string,
    ) => {
      if (area === 'session' && 'activeAssetId' in changes) {
        setAssetId(changes.activeAssetId.newValue ?? null)
      }
    }
    chrome.storage.onChanged.addListener(onChange)
    return () => {
      cancelled = true
      chrome.storage.onChanged.removeListener(onChange)
    }
  }, [])

  return assetId
}
