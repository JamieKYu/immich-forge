import { useEffect, useRef, useState } from 'react'
import { ForgeClient } from '../lib/forge-client'
import { loadSettings, saveSettings } from '../lib/storage'
import { ensureForgeAccess, forgeOriginPattern, hasForgeAccess } from '../lib/host-permissions'
import {
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
  // True once a forge job is submitted (the review screen is showing).
  const [reviewing, setReviewing] = useState(false)
  // Whether we hold host access to the configured Forge origin (null = checking).
  const [access, setAccess] = useState<boolean | null>(null)
  const assetId = useActiveAsset()
  // Latest values being edited in the settings panel, reported up by SettingsView
  // so the header "close" link can save them.
  const pending = useRef<Settings | null>(null)

  const client = settings && settings.forgeUrl ? new ForgeClient(settings) : null

  useEffect(() => {
    loadSettings().then((s) => {
      setSettings(s)
      if (!s.forgeUrl) setShowSettings(true)
    })
  }, [])

  useEffect(() => {
    if (!settings?.forgeUrl) {
      setAccess(true) // nothing to reach yet; settings view will handle setup
      return
    }
    hasForgeAccess(settings.forgeUrl).then(setAccess)
  }, [settings])

  useEffect(() => {
    if (!client || access === false) return
    client
      .health()
      .then((h) => setHealth({ forge: h.ok, immich: h.immich }))
      .catch(() => setHealth({ forge: false, immich: false }))
  }, [settings, access])

  async function persist(next: Settings) {
    // Request host access to the user's Forge origin first, so this call stays
    // within the click gesture (chrome.permissions.request requires one).
    if (next.forgeUrl) await ensureForgeAccess(next.forgeUrl)
    await saveSettings(next)
    setSettings(next)
    setShowSettings(false)
  }

  // Header link: opening just shows the panel; closing saves the edited values.
  function toggleSettings() {
    if (!showSettings) {
      setShowSettings(true)
    } else if (pending.current) {
      persist(pending.current)
    } else {
      setShowSettings(false)
    }
  }

  const title =
    showSettings || !client
      ? 'Configure'
      : reviewing
        ? 'Review'
        : 'Original'

  return (
    <div className="app">
      <header>
        <h1>{title}</h1>
        <span className="status">
          {health === null
            ? '…'
            : !health.forge
              ? '○ forge offline'
              : health.immich
                ? (
                  <>
                    <span style={{ color: 'var(--accent-2)' }}>●</span> connected
                  </>
                )
                : '◐ immich unreachable'}
          {' · '}
          <a onClick={toggleSettings} style={{ cursor: 'pointer' }}>
            {showSettings ? 'close' : 'settings'}
          </a>
        </span>
      </header>
      <main>
        {showSettings || !client ? (
          <SettingsView
            settings={settings}
            onChange={(s) => {
              pending.current = s
            }}
          />
        ) : access === false ? (
          <GrantAccess
            forgeUrl={settings!.forgeUrl}
            onGranted={() => setAccess(true)}
          />
        ) : assetId === undefined ? (
          <p className="muted">Detecting current photo…</p>
        ) : assetId === null ? (
          <NoPhoto />
        ) : (
          // key by assetId so navigating to another photo resets the flow.
          <PhotoForge
            key={assetId}
            client={client}
            assetId={assetId}
            operations={settings!.operations}
            onReviewingChange={setReviewing}
          />
        )}
      </main>
    </div>
  )
}

// Shown when a Forge server is configured but the user hasn't granted host
// access to its origin yet (fresh install carried a saved URL, or access was
// revoked). One click re-requests it, scoped to just that origin.
function GrantAccess({
  forgeUrl,
  onGranted,
}: {
  forgeUrl: string
  onGranted: () => void
}) {
  const origin = forgeOriginPattern(forgeUrl)?.replace(/\/\*$/, '') ?? forgeUrl
  async function grant() {
    if (await ensureForgeAccess(forgeUrl)) onGranted()
  }
  return (
    <div>
      <p>Allow Forge to connect to your server?</p>
      <p className="muted">
        Forge needs permission to reach <code>{origin}</code>.
      </p>
      <button className="primary" style={{ width: '100%' }} onClick={grant}>
        Grant access
      </button>
    </div>
  )
}

function NoPhoto() {
  return (
    <div>
      <p>Open a photo in Immich to forge it.</p>
    </div>
  )
}

// Configure → review flow for one asset. Remounts (resets) when assetId changes.
function PhotoForge({
  client,
  assetId,
  operations,
  onReviewingChange,
}: {
  client: ForgeClient
  assetId: string
  operations: ForgeOperations
  onReviewingChange: (reviewing: boolean) => void
}) {
  const [asset, setAsset] = useState<ImmichAsset | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<JobInfo | null>(null)

  useEffect(() => {
    setError(null)
    client
      .asset(assetId)
      .then(setAsset)
      .catch((e: Error) => setError(e.message))
  }, [assetId])

  // Drive the header title: review screen while a job exists, reset on unmount
  // (settings opened, photo changed, or navigated away).
  useEffect(() => {
    onReviewingChange(!!job)
  }, [job])
  useEffect(() => () => onReviewingChange(false), [])

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
      onSubmitted={setJob}
    />
  )
}
