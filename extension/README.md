# Forge for Immich — Chrome Extension

Manifest V3 side-panel extension. It detects when you're **viewing a single
photo** in the Immich web app and lights up the toolbar icon. Click it to open
the side panel for that photo: preview, enhancement options, **Forge**, compare
before/after, and save to stack the forged image as the new primary asset.

> Unofficial, independent project — not affiliated with or endorsed by Immich or
> FUTO.

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
3. Open the side panel and click **Set up Forge server** (or the extension's
   **Options**). Enter your Forge server address (e.g. `http://gpu-host:8000`)
   and API token, then **Save & test**. Chrome prompts you to grant access to
   that Forge origin (scoped host access), and the connection is checked before
   it's saved. After that the address and token stay out of the way; the panel's
   status and gear icon link back to the options page.
4. Open a photo in Immich (`…/photos/<id>`) — the icon lights up (colored).
   Click it, choose enhancements for that photo, and **Forge**.

The photo screen has an **Auto | Custom** switch. **Auto** (the default for new
installs) starts each photo from the server's suggestions, with a short reason
under every option, e.g. "800×600 source" or "already in color". **Custom**
starts from the default enhancements set on the options page. Changing an
option in Auto turns that photo into Custom, starting from the suggestions;
clicking the switch itself is remembered for the next photo. Changes on the
photo screen apply to that photo only. Servers without Auto suggestions (before
the release that added them) get no switch, just Custom. Options the connected server lacks or has turned off are
greyed out with the reason.

`npm run dev` rebuilds on change (reload the unpacked extension to pick up
service-worker/manifest changes).

## Structure

```
manifest.config.ts          MV3 manifest (via @crxjs/vite-plugin)
public/icons/              lit (active) + unlit (inactive) PNGs, 16/32/48/128
src/background/             service worker: per-tab icon swap + active asset id
src/lib/                    types, settings storage, ForgeClient, immich-url parser,
                            connection test, operation helpers (labels, capability gating)
src/ui/                     components shared by the panel and options page
  EnhancementControls      grouped enhancement options ("Repair" / "Enhance")
src/options/                Chrome options page: server setup + default enhancements
src/sidepanel/             React UI
  App.tsx                  connection status + current-photo flow
  useActiveAsset.ts        active tab's asset id (live, SPA-aware)
  views/PhotoView          preview, per-photo enhancements, sticky Forge button
  views/ReviewView         progress, before/after, Save to Immich / Discard
  views/Compare            before/after divider
```

## Notes

- Cross-origin calls to Forge work via `optional_host_permissions`, requested at
  runtime scoped to the user's Forge origin (see `lib/host-permissions.ts`). The
  server keeps CORS closed by default — the extension doesn't rely on it.
- Job polling currently lives in the side panel. To survive the panel closing,
  move polling into the service worker and persist job ids in
  `chrome.storage.session` (noted in `service-worker.ts`).
- Toolbar icons are the "F" logo in two states (active/gradient, inactive/grey),
  committed under `public/icons/`. The design sheet and the generator that
  produces them are maintained separately; the committed PNGs are copied verbatim
  from `public/icons/` into `dist/icons/` at build.
