// Recognize an Immich asset being viewed from a tab URL.
//
// We key off the asset UUID in the path rather than the host, because the host
// the user browses (e.g. http://192.168.50.150:2283) often differs from how the
// Forge server reaches Immich internally (e.g. http://immich-server:2283). The
// UUID is all the server needs.
//
// Covers the common viewer routes:
//   /photos/<uuid>                       (main timeline viewer)
//   /albums/<albumId>/photo/<uuid>       (album viewer)
//   /share/<key>/photo/<uuid>            (shared link)

const UUID = '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
const ASSET_RE = new RegExp(`/photos?/(${UUID})`)

export function parseAssetId(url: string | undefined): string | null {
  if (!url) return null
  try {
    const { pathname } = new URL(url)
    const m = pathname.match(ASSET_RE)
    return m ? m[1].toLowerCase() : null
  } catch {
    return null
  }
}
