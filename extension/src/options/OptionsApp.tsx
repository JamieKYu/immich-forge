import { useEffect, useState } from 'react'
import { type ConnectionResult, isValidForgeUrl, testConnection } from '../lib/connection'
import { ensureForgeAccess, hasForgeAccess } from '../lib/host-permissions'
import { describeOperations } from '../lib/operations'
import { loadSettings, onSettingsChanged, saveSettings } from '../lib/storage'
import {
  DEFAULT_OPERATIONS,
  type ForgeOperations,
  type ModelsInfo,
  type Settings,
} from '../lib/types'
import { EnhancementControls } from '../ui/EnhancementControls'
import { EyeIcon, EyeOffIcon } from '../ui/icons'

// Chrome's extension options page: one-time server setup plus the default
// enhancements every photo starts from. The side panel never shows the URL or
// token; it links here when the connection needs attention.
export function OptionsApp() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  // Bumped to re-run the connection check (e.g. after granting access).
  const [recheck, setRecheck] = useState(0)

  useEffect(() => {
    loadSettings().then(setSettings)
    return onSettingsChanged(setSettings)
  }, [])

  // Check the saved connection on open and whenever it changes.
  const forgeUrl = settings?.forgeUrl
  const forgeToken = settings?.forgeToken
  useEffect(() => {
    if (!forgeUrl) return setStatus(null)
    let live = true
    setStatus({ state: 'checking' })
    hasForgeAccess(forgeUrl).then(async (granted) => {
      if (!live) return
      if (!granted) return setStatus({ state: 'no-access' })
      const result = await testConnection(forgeUrl, forgeToken ?? '')
      if (live) setStatus({ state: 'result', result })
    })
    return () => {
      live = false
    }
  }, [forgeUrl, forgeToken, recheck])

  if (!settings) return null

  const models = status?.state === 'result' && status.result.ok ? status.result.models : null

  return (
    <div className="options">
      <h1>Forge for Immich</h1>
      <ServerSection
        settings={settings}
        status={status}
        onRecheck={() => setRecheck((n) => n + 1)}
      />
      <DefaultsSection settings={settings} models={models} />
    </div>
  )
}

type Status =
  | { state: 'checking' }
  | { state: 'no-access' }
  | { state: 'result'; result: ConnectionResult }

function ServerSection({
  settings,
  status,
  onRecheck,
}: {
  settings: Settings
  status: Status | null
  onRecheck: () => void
}) {
  const configured = !!settings.forgeUrl
  const [editing, setEditing] = useState(!configured)

  return (
    <section>
      <h2>Forge server</h2>
      {editing ? (
        <ServerForm
          settings={settings}
          onSaved={() => setEditing(false)}
          onCancel={configured ? () => setEditing(false) : undefined}
        />
      ) : (
        <>
          <div className="server-row">
            <span>
              <StatusDot status={status} /> {host(settings.forgeUrl)}
            </span>
            <button onClick={() => setEditing(true)}>Change</button>
          </div>
          <StatusDetail status={status} forgeUrl={settings.forgeUrl} onGranted={onRecheck} />
        </>
      )}
    </section>
  )
}

function ServerForm({
  settings,
  onSaved,
  onCancel,
}: {
  settings: Settings
  onSaved: () => void
  onCancel?: () => void
}) {
  const [url, setUrl] = useState(settings.forgeUrl)
  // Prefilled (masked) with the saved token; the eye button reveals it.
  const [token, setToken] = useState(settings.forgeToken)
  const [showToken, setShowToken] = useState(false)
  const [busy, setBusy] = useState(false)
  const [failure, setFailure] = useState<string | null>(null)


  async function save(skipTest: boolean) {
    const forgeUrl = url.trim().replace(/\/+$/, '')
    const forgeToken = token.trim()
    setFailure(null)
    if (!isValidForgeUrl(forgeUrl)) {
      setFailure('Enter a full address, e.g. http://gpu-host:8000')
      return
    }
    // Must be the first await: chrome.permissions.request needs the click gesture.
    if (!(await ensureForgeAccess(forgeUrl))) {
      setFailure('Chrome needs permission to connect to this address. Try again and choose Allow.')
      return
    }
    setBusy(true)
    try {
      if (!skipTest) {
        const result = await testConnection(forgeUrl, forgeToken)
        if (!result.ok) {
          setFailure(result.message)
          return
        }
      }
      await saveSettings({ ...settings, forgeUrl, forgeToken })
      onSaved()
    } finally {
      setBusy(false)
    }
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        void save(false)
      }}
    >
      <label htmlFor="forge-url">Server address</label>
      <input
        id="forge-url"
        type="text"
        placeholder="http://gpu-host:8000"
        value={url}
        autoFocus
        onChange={(e) => setUrl(e.target.value)}
      />
      <label htmlFor="forge-token">API token</label>
      <div className="input-wrap">
        <input
          id="forge-token"
          type={showToken ? 'text' : 'password'}
          placeholder="FORGE_API_TOKEN from the server"
          value={token}
          autoComplete="off"
          onChange={(e) => setToken(e.target.value)}
        />
        <button
          type="button"
          className="reveal"
          aria-label={showToken ? 'Hide token' : 'Show token'}
          aria-pressed={showToken}
          onClick={() => setShowToken((v) => !v)}
        >
          {showToken ? <EyeOffIcon /> : <EyeIcon />}
        </button>
      </div>

      {failure && <p className="error">{failure}</p>}

      <div className="row" style={{ marginTop: 14 }}>
        <button type="submit" className="primary" disabled={busy || !url.trim()}>
          {busy ? 'Testing…' : 'Save & test'}
        </button>
        {failure && !busy && (
          <button type="button" onClick={() => void save(true)}>
            Save anyway
          </button>
        )}
        {onCancel && (
          <button type="button" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
        )}
      </div>
    </form>
  )
}

function StatusDot({ status }: { status: Status | null }) {
  if (!status || status.state === 'checking') return <span className="dot">…</span>
  if (status.state === 'result' && status.result.ok) {
    return <span className={`dot ${status.result.immich ? 'ok' : 'warn'}`}>●</span>
  }
  return <span className="dot bad">●</span>
}

function StatusDetail({
  status,
  forgeUrl,
  onGranted,
}: {
  status: Status | null
  forgeUrl: string
  onGranted: () => void
}) {
  if (!status || status.state === 'checking') return <p className="muted">Checking connection…</p>
  if (status.state === 'no-access') {
    return (
      <p className="muted">
        Chrome hasn't been allowed to connect to this server.{' '}
        <a
          href="#"
          onClick={async (e) => {
            e.preventDefault()
            if (await ensureForgeAccess(forgeUrl)) onGranted()
          }}
        >
          Allow access
        </a>
      </p>
    )
  }
  const r = status.result
  if (!r.ok) return <p className="error">{r.message}</p>
  if (!r.immich) {
    return (
      <p className="note">
        Connected, but the server can't reach Immich. Check its IMMICH_BASE_URL and IMMICH_API_KEY.
      </p>
    )
  }
  return <p className="muted">Connected.</p>
}

function DefaultsSection({
  settings,
  models,
}: {
  settings: Settings
  // Server capabilities, so unavailable options are greyed out here too.
  models: ModelsInfo | null
}) {
  const update = (operations: ForgeOperations) => void saveSettings({ ...settings, operations })
  const summary = describeOperations(settings.operations)

  return (
    <section>
      <h2>Default enhancements</h2>
      <p className="muted">
        In Custom mode, every photo starts with these. In Auto mode, Forge suggests settings for
        each photo instead. Either way you can change them per photo in the side panel. Changes
        save automatically.
      </p>
      <EnhancementControls operations={settings.operations} onChange={update} models={models} />
      <div className="row between" style={{ marginTop: 8 }}>
        <span className="muted">{summary.length ? summary.join(' · ') : 'Nothing selected'}</span>
        <button onClick={() => update(DEFAULT_OPERATIONS)}>Restore built-in defaults</button>
      </div>
    </section>
  )
}

function host(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}
