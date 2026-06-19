import { useState } from 'react'
import { ForgeClient } from '../../lib/forge-client'
import type { Settings } from '../../lib/types'

export function SettingsView({
  settings,
  onSave,
  client,
}: {
  settings: Settings | null
  onSave: (s: Settings) => void
  client: ForgeClient | null
}) {
  const [forgeUrl, setForgeUrl] = useState(settings?.forgeUrl ?? '')
  const [forgeToken, setForgeToken] = useState(settings?.forgeToken ?? '')
  const [test, setTest] = useState<string | null>(null)

  async function testConnection() {
    setTest('testing…')
    try {
      const c = client ?? new ForgeClient({ forgeUrl, forgeToken })
      const h = await c.health()
      setTest(h.ok ? `ok — immich:${h.immich ? '✓' : '✗'}` : 'unhealthy')
    } catch (e) {
      setTest(`failed: ${(e as Error).message}`)
    }
  }

  return (
    <div>
      <label>Forge server URL</label>
      <input
        type="text"
        placeholder="http://gpu-host:8000"
        value={forgeUrl}
        onChange={(e) => setForgeUrl(e.target.value)}
      />
      <label>Forge API token</label>
      <input
        type="password"
        placeholder="Bearer token (leave blank if auth disabled)"
        value={forgeToken}
        onChange={(e) => setForgeToken(e.target.value)}
      />
      <p className="muted" style={{ marginTop: 10 }}>
        The Forge server holds your Immich API key — it is never stored in the
        browser.
      </p>
      <div className="row" style={{ marginTop: 12 }}>
        <button onClick={testConnection}>Test</button>
        <button
          className="primary"
          disabled={!forgeUrl}
          onClick={() => onSave({ forgeUrl, forgeToken })}
        >
          Save
        </button>
        {test && <span className="muted">{test}</span>}
      </div>
    </div>
  )
}
