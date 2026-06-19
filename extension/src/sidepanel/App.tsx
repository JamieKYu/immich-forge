import { useEffect, useState } from 'react'
import { ForgeClient } from '../lib/forge-client'
import { loadSettings, saveSettings } from '../lib/storage'
import {
  DEFAULT_OPERATIONS,
  type ForgeOperations,
  type ImmichAsset,
  type JobInfo,
  type Settings,
} from '../lib/types'
import { useActiveAsset } from './useActiveAsset'
import { SettingsView } from './views/SettingsView'
import { ConfigureView } from './views/ConfigureView'
import { ReviewView } from './views/ReviewView'

export function App() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [health, setHealth] = useState<{ forge: boolean; immich: boolean } | null>(
    null,
  )
  const assetId = useActiveAsset()

  const client = settings && settings.forgeUrl ? new ForgeClient(settings) : null

  useEffect(() => {
    loadSettings().then((s) => {
      setSettings(s)
      if (!s.forgeUrl) setShowSettings(true)
    })
  }, [])

  useEffect(() => {
    if (!client) return
    client
      .health()
      .then((h) => setHealth({ forge: h.ok, immich: h.immich }))
      .catch(() => setHealth({ forge: false, immich: false }))
  }, [settings])

  async function persist(next: Settings) {
    await saveSettings(next)
    setSettings(next)
    setShowSettings(false)
  }

  return (
    <div className="app">
      <header>
        <h1>Immich Forge</h1>
        <span className="status">
          {health === null
            ? '…'
            : !health.forge
              ? '○ forge offline'
              : health.immich
                ? '● connected'
                : '◐ immich unreachable'}
          {' · '}
          <a onClick={() => setShowSettings((v) => !v)} style={{ cursor: 'pointer' }}>
            {showSettings ? 'close' : 'settings'}
          </a>
        </span>
      </header>
      <main>
        {showSettings || !client ? (
          <SettingsView settings={settings} onSave={persist} client={client} />
        ) : assetId === undefined ? (
          <p className="muted">Detecting current photo…</p>
        ) : assetId === null ? (
          <NoPhoto />
        ) : (
          // key by assetId so navigating to another photo resets the flow.
          <PhotoForge key={assetId} client={client} assetId={assetId} />
        )}
      </main>
    </div>
  )
}

function NoPhoto() {
  return (
    <div>
      <p>Open a photo in Immich to forge it.</p>
      <p className="muted">
        Navigate to a single photo (its URL looks like
        <code> /photos/&lt;id&gt;</code>) and the toolbar icon lights up.
      </p>
    </div>
  )
}

// Configure → review flow for one asset. Remounts (resets) when assetId changes.
function PhotoForge({ client, assetId }: { client: ForgeClient; assetId: string }) {
  const [asset, setAsset] = useState<ImmichAsset | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [operations, setOperations] = useState<ForgeOperations>(DEFAULT_OPERATIONS)
  const [job, setJob] = useState<JobInfo | null>(null)

  useEffect(() => {
    setError(null)
    client
      .asset(assetId)
      .then(setAsset)
      .catch((e: Error) => setError(e.message))
  }, [assetId])

  // Fall back to a minimal asset shell if metadata hasn't loaded yet; the id is
  // all the server needs to forge.
  const subject: ImmichAsset =
    asset ?? { id: assetId, originalFileName: 'photo', type: 'IMAGE' }

  if (error) {
    return (
      <div>
        <p className="error">Couldn't load this asset from the Forge server.</p>
        <p className="muted" style={{ wordBreak: 'break-word' }}>
          asset {assetId}
          <br />
          {error}
        </p>
        <p className="muted">
          Check the Forge server's <code>IMMICH_BASE_URL</code> /{' '}
          <code>IMMICH_API_KEY</code>, and that the asset belongs to that API
          key's user.
        </p>
      </div>
    )
  }

  if (job) {
    return (
      <ReviewView
        client={client}
        asset={subject}
        initialJob={job}
        operations={operations}
        onReforge={() => setJob(null)}
        onDone={() => setJob(null)}
      />
    )
  }

  return (
    <ConfigureView
      client={client}
      asset={subject}
      operations={operations}
      setOperations={setOperations}
      onSubmitted={setJob}
    />
  )
}
