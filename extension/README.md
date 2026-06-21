# Immich Forge — Chrome Extension

Manifest V3 side-panel extension. It detects when you're **viewing a single
photo** in the Immich web app and lights up the toolbar icon. Click it to open
the side panel for that photo: preview, enhancement options, **Forge**, review
before/after, and accept to stack the forged image as the new primary asset.

It talks **only** to the Forge server (server-as-broker), which holds the Immich
API key — no Immich credentials are stored in the browser.

## How activation works

- The service worker watches the active tab's URL for an Immich asset id
  (`/photos/<uuid>`, also album/share `/photo/<uuid>`). It keys off the **asset
  UUID, not the host**, so it works regardless of how you reach Immich.
- On a photo page the toolbar shows the **lit (colored) icon**; otherwise the
  **unlit (grey) icon**. Clicking always opens the panel — which guides you to a
  photo when unlit.
- Immich is a client-routed SPA, so `webNavigation.onHistoryStateUpdated` is used
  to catch navigation _between_ photos, not just full page loads. The active
  asset id is mirrored into `storage.session`, so an open panel updates live as
  you move between photos.

## Develop

```bash
npm install
npm run build        # outputs ./dist
```

Then load it in Chrome:

1. Go to `chrome://extensions`, enable **Developer mode**.
2. **Load unpacked** → select `extension/dist`.
3. Open the side panel (the icon is grey until you're on a photo) and, in
   **settings**, enter your Forge server URL (e.g. `http://gpu-host:8000`) and
   token, then **Test** → **Save**.
4. Open a photo in Immich (`…/photos/<id>`) — the icon lights up (colored).
   Click it to forge that photo.

`npm run dev` rebuilds on change (reload the unpacked extension to pick up
service-worker/manifest changes).

## Structure

```
manifest.config.ts          MV3 manifest (via @crxjs/vite-plugin)
icons/generate_icons.py     regenerates the toolbar icons into public/icons/
public/icons/              lit (active) + unlit (inactive) PNGs, 16/32/48/128
src/background/             service worker: per-tab icon swap + active asset id
src/lib/                    types, settings storage, ForgeClient, immich-url parser
src/sidepanel/             React UI
  App.tsx                  settings vs. current-photo flow
  useActiveAsset.ts        active tab's asset id (live, SPA-aware)
  views/SettingsView       Forge URL + token + sticky default enhancements
  views/ConfigureView      Forge button + preview
  views/ReviewView         progress, before/after, accept & stack
```

## Notes

- Cross-origin calls to Forge work via `host_permissions`; tighten
  `allow_origins` on the server and the host pattern in the manifest for
  production.
- Job polling currently lives in the side panel. To survive the panel closing,
  move polling into the service worker and persist job ids in
  `chrome.storage.session` (noted in `service-worker.ts`).
- Toolbar icons are the "F" logo in two states (active/gradient, inactive/grey),
  extracted from the design sheet `icons/forge-icons.png` (a flattened preview on
  a transparency checkerboard). Regenerate with
  `server/.venv/bin/python icons/generate_icons.py` (needs Pillow + numpy +
  OpenCV); output lands in `public/icons/` and is copied verbatim into
  `dist/icons/`.
