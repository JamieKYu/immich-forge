// Reload the active tab in the panel's window (the Immich photo page) so a
// freshly accepted forge appears in the asset's stack.
export async function reloadActiveTab(): Promise<void> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    if (tab?.id != null) await chrome.tabs.reload(tab.id)
  } catch {
    // Tab may have closed between accept and refresh; nothing to reload.
  }
}
