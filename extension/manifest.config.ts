import { defineManifest } from '@crxjs/vite-plugin'

// Host permissions let the service worker call the Forge server cross-origin
// without CORS friction. The user's actual Forge URL is added at runtime via
// the optional_host_permissions flow (chrome.permissions.request).
export default defineManifest({
  manifest_version: 3,
  name: 'Forge for Immich',
  version: '0.1.0',
  description: 'Forge low-quality Immich photos: upscale, face-restore, colorize.',
  // tabs/webNavigation: detect the asset id in the active tab's URL (incl. SPA
  // navigation). activeAssetId is mirrored into storage.session for the panel.
  permissions: ['storage', 'sidePanel', 'tabs', 'activeTab', 'webNavigation'],
  host_permissions: ['http://*/*', 'https://*/*'],
  background: {
    service_worker: 'src/background/service-worker.ts',
    type: 'module',
  },
  side_panel: {
    default_path: 'src/sidepanel/index.html',
  },
  // Brand icon (extensions page / menus): the active, colored mark.
  icons: {
    16: 'icons/forge-active-16.png',
    32: 'icons/forge-active-32.png',
    48: 'icons/forge-active-48.png',
    128: 'icons/forge-active-128.png',
  },
  action: {
    default_title: 'Open Forge for Immich',
    // Resting toolbar state is the unlit (grey) icon; the service worker swaps
    // to the active icon on photo pages.
    default_icon: {
      16: 'icons/forge-inactive-16.png',
      32: 'icons/forge-inactive-32.png',
      48: 'icons/forge-inactive-48.png',
    },
  },
})
