// Light up the toolbar icon when the active tab is viewing an Immich photo, and
// publish that asset id for the side panel to pick up.
//
// - Per-tab action state: enabled + green badge on a photo page, disabled (grey,
//   inert) otherwise. A disabled action won't open the side panel on click, so
//   the affordance matches "you can forge this".
// - Active asset is mirrored into chrome.storage.session so the panel can read
//   it on open and react to changes (including SPA navigation between photos).

import { parseAssetId } from '../lib/immich-url'

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((err) => console.error('sidePanel behavior', err))
})

// Per-state icon paths (relative to the extension root; copied from public/).
const iconSet = (state: 'active' | 'inactive') => ({
  16: `icons/forge-${state}-16.png`,
  32: `icons/forge-${state}-32.png`,
  48: `icons/forge-${state}-48.png`,
})

async function applyIcon(tabId: number, assetId: string | null) {
  // The lit (colored) icon means "you can forge this photo"; the unlit (grey)
  // icon means there's nothing to forge here. The action stays enabled in both
  // states so a click always opens the panel (which guides you when unlit).
  try {
    await chrome.action.setIcon({
      tabId,
      path: iconSet(assetId ? 'active' : 'inactive'),
    })
    await chrome.action.setTitle({
      tabId,
      title: assetId ? 'Forge this photo' : 'Forge for Immich',
    })
  } catch {
    // tab may have closed between event and update; ignore.
  }
}

async function publishActive(assetId: string | null) {
  await chrome.storage.session.set({ activeAssetId: assetId })
}

async function handleTab(tabId: number, url: string | undefined, isActive: boolean) {
  const assetId = parseAssetId(url)
  await applyIcon(tabId, assetId)
  if (isActive) await publishActive(assetId)
}

// Full loads + same-document URL changes.
chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === 'complete' || info.url) {
    void handleTab(tabId, tab.url, tab.active)
  }
})

// Switching tabs.
chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId)
    await handleTab(tabId, tab.url, true)
  } catch {
    /* tab gone */
  }
})

// Immich is a client-routed SPA — navigating between photos uses history
// pushState and does NOT fire onUpdated reliably. Catch it here.
chrome.webNavigation.onHistoryStateUpdated.addListener((d) => {
  if (d.frameId !== 0) return
  void chrome.tabs.get(d.tabId).then((tab) => handleTab(d.tabId, d.url, tab.active))
})

// Switching windows.
chrome.windows.onFocusChanged.addListener(async (windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) return
  const [tab] = await chrome.tabs.query({ active: true, windowId })
  if (tab?.id != null) await handleTab(tab.id, tab.url, true)
})
