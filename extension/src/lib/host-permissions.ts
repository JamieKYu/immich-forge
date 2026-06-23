// The Forge server origin is user-defined, so we don't hold host access up
// front. We request it at runtime (scoped to just that origin) when the user
// saves their settings — see manifest's optional_host_permissions.

// The origin match pattern Chrome needs (e.g. "http://gpu-host:8000/*"), or null
// if the URL is empty/invalid.
export function forgeOriginPattern(forgeUrl: string): string | null {
  try {
    return new URL(forgeUrl).origin + '/*'
  } catch {
    return null
  }
}

// Request access to the user's Forge origin. MUST be called synchronously from a
// user gesture (e.g. a click handler) — chrome.permissions.request requires one.
// Resolves true if access is (already) granted. No prompt is shown when the
// permission is already held.
export async function ensureForgeAccess(forgeUrl: string): Promise<boolean> {
  const pattern = forgeOriginPattern(forgeUrl)
  if (!pattern) return false
  try {
    return await chrome.permissions.request({ origins: [pattern] })
  } catch {
    return false
  }
}

// Non-gesture check: does the extension currently have access to this origin?
export async function hasForgeAccess(forgeUrl: string): Promise<boolean> {
  const pattern = forgeOriginPattern(forgeUrl)
  if (!pattern) return false
  try {
    return await chrome.permissions.contains({ origins: [pattern] })
  } catch {
    return false
  }
}
