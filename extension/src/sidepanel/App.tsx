import { useEffect, useState } from 'react'
import { type ConnectionResult, testConnection } from '../lib/connection'
import { ForgeClient } from '../lib/forge-client'
import { ensureForgeAccess, forgeOriginPattern, hasForgeAccess } from '../lib/host-permissions'
import { loadSettings, onSettingsChanged } from '../lib/storage'
import type { ForgeOperations, ImmichAsset, JobInfo, ModelsInfo, Settings } from '../lib/types'
import { GearIcon } from '../ui/icons'
import { useActiveAsset } from './useActiveAsset'
import { useAssetImage } from './useAssetImage'
import { PhotoView } from './views/PhotoView'
import { ReviewView } from './views/ReviewView'

// Server setup lives on Chrome's options page; the panel only links there.
// openOptionsPage() rejects ("Could not create an options page") when the
// running manifest has no options_ui, e.g. an unpacked build updated on disk
// without reloading the extension. Fall back to opening the page in a tab.
const OPTIONS_PATH = 'src/options/index.html'
function openOptions() {
  chrome.runtime.openOptionsPage().catch(() => {
    void chrome.tabs.create({ url: chrome.runtime.getURL(OPTIONS_PATH) })
  })
}

export function App() {
  const [settings, setSettings] = useState<Settings | null>(null)
  // Whether we hold host access to the configured Forge origin (null = checking).
  const [access, setAccess] = useState<boolean | null>(null)
  const [connection, setConnection] = useState<ConnectionResult | null>(null)
  const assetId = useActiveAsset()

  useEffect(() => {
    loadSettings().then(setSettings)
    // Pick up saves from the options page while the panel is open.
    return onSettingsChanged(setSettings)
  }, [])

  const forgeUrl = settings?.forgeUrl ?? ''
  const forgeToken = settings?.forgeToken ?? ''

  useEffect(() => {
    if (!forgeUrl) return
    setAccess(null)
    hasForgeAccess(forgeUrl).then(setAccess)
  }, [forgeUrl])

  useEffect(() => {
    if (!forgeUrl || !access) return
    let live = true
    setConnection(null)
    testConnection(forgeUrl, forgeToken).then((r) => live && setConnection(r))
    return () => {
      live = false
    }
  }, [forgeUrl, forgeToken, access])

  if (!settings) return null

  const client = forgeUrl ? new ForgeClient(settings) : null
  const models = connection?.ok ? connection.models : null

  let body
  if (!client) {
    body = <Welcome />
  } else if (access === false) {
    body = <GrantAccess forgeUrl={forgeUrl} onGranted={() => setAccess(true)} />
  } else if (assetId === undefined || access === null) {
    body = <Message>Detecting current photo…</Message>
  } else if (assetId === null) {
    body = <Message>Open a photo in Immich to forge it.</Message>
  } else {
    // key by assetId so navigating to another photo resets the flow.
    body = (
      <PhotoForge
        key={assetId}
        client={client}
        assetId={assetId}
        defaults={settings.operations}
        models={models}
      />
    )
  }

  return (
    <div className="app">
      <header>
        <h1>Forge</h1>
        <span className="status">
          {client && access && <ConnectionStatus connection={connection} />}
          <button className="icon" aria-label="Settings" title="Settings" onClick={openOptions}>
            <GearIcon />
          </button>
        </span>
      </header>
      {client && access && connection && !connection.ok && (
        <div className="banner">
          <span>{connection.message}</span>
          <button onClick={openOptions}>Fix in settings</button>
        </div>
      )}
      <main>{body}</main>
    </div>
  )
}

function ConnectionStatus({ connection }: { connection: ConnectionResult | null }) {
  if (!connection) return <span className="muted">…</span>
  const [cls, text] = !connection.ok
    ? connection.kind === 'unreachable'
      ? ['bad', 'server offline']
      : ['bad', 'not connected']
    : connection.immich
      ? ['ok', 'connected']
      : ['warn', 'Immich unreachable']
  return (
    <a className="status-link" onClick={openOptions} title="Connection settings">
      <span className={`dot ${cls}`}>●</span> {text}
    </a>
  )
}

function Message({ children }: { children: React.ReactNode }) {
  return (
    <div className="body">
      <p className="muted">{children}</p>
    </div>
  )
}

// First run: no server configured yet.
function Welcome() {
  return (
    <div className="body">
      <p>Welcome to Forge for Immich.</p>
      <p className="muted">
        Connect your Forge server once, then open any photo in Immich to enhance it.
      </p>
      <button className="primary" onClick={openOptions}>
        Set up Forge server
      </button>
    </div>
  )
}

// Shown when a Forge server is configured but the user hasn't granted host
// access to its origin yet (fresh install carried a saved URL, or access was
// revoked). One click re-requests it, scoped to just that origin.
function GrantAccess({ forgeUrl, onGranted }: { forgeUrl: string; onGranted: () => void }) {
  const origin = forgeOriginPattern(forgeUrl)?.replace(/\/\*$/, '') ?? forgeUrl
  async function grant() {
    if (await ensureForgeAccess(forgeUrl)) onGranted()
  }
  return (
    <div className="body">
      <p>Allow Forge to connect to your server?</p>
      <p className="muted">
        Forge needs permission to reach <code>{origin}</code>.
      </p>
      <button className="primary" onClick={grant}>
        Grant access
      </button>
    </div>
  )
}

// Photo screen → review flow for one asset. Remounts (resets) when assetId
// changes. Holds this photo's one-off enhancement choices, so "Discard" from
// review comes back with them intact.
function PhotoForge({
  client,
  assetId,
  defaults,
  models,
}: {
  client: ForgeClient
  assetId: string
  defaults: ForgeOperations
  models: ModelsInfo | null
}) {
  const [asset, setAsset] = useState<ImmichAsset | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<{ info: JobInfo; operations: ForgeOperations } | null>(null)
  const [operations, setOperations] = useState<ForgeOperations>(defaults)
  // Loaded once here and shared by the photo screen and the before/after view.
  const preview = useAssetImage(client, assetId)

  useEffect(() => {
    setError(null)
    client
      .asset(assetId)
      .then(setAsset)
      .catch((e: Error) => setError(e.message))
  }, [assetId])

  // Fall back to a minimal asset shell if metadata hasn't loaded yet; the id is
  // all the server needs to forge.
  const subject: ImmichAsset = asset ?? { id: assetId, originalFileName: 'photo', type: 'IMAGE' }

  if (error) {
    return (
      <div className="body">
        <p className="error">Couldn't load this asset from the Forge server.</p>
        <p className="muted" style={{ wordBreak: 'break-word' }}>
          asset {assetId}
          <br />
          {error}
        </p>
        <p className="muted">
          Check the Forge server's <code>IMMICH_BASE_URL</code> / <code>IMMICH_API_KEY</code>,
          and that the asset belongs to that API key's user.
        </p>
      </div>
    )
  }

  if (job) {
    return (
      <ReviewView
        client={client}
        initialJob={job.info}
        operations={job.operations}
        before={preview}
        onBack={() => setJob(null)}
      />
    )
  }

  return (
    <PhotoView
      client={client}
      asset={subject}
      preview={preview}
      operations={operations}
      onOperationsChange={setOperations}
      models={models}
      onSubmitted={(info, ops) => setJob({ info, operations: ops })}
    />
  )
}
