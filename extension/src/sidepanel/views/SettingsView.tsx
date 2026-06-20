import { useState } from 'react'
import { ForgeClient } from '../../lib/forge-client'
import { DEFAULT_OPERATIONS, type ForgeOperations, type Settings } from '../../lib/types'

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
  const [operations, setOperations] = useState<ForgeOperations>(
    settings?.operations ?? DEFAULT_OPERATIONS,
  )
  const [test, setTest] = useState<string | null>(null)

  const set = (patch: Partial<ForgeOperations>) =>
    setOperations((o) => ({ ...o, ...patch }))

  async function testConnection() {
    setTest('testing…')
    try {
      const c = client ?? new ForgeClient({ forgeUrl, forgeToken, operations })
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

      <label style={{ marginTop: 16 }}>Default enhancements</label>
      <p className="muted" style={{ marginTop: 0 }}>
        Applied to every photo you forge. Saved on this device.
      </p>

      <div className="toggle">
        <input
          type="checkbox"
          checked={operations.colorize}
          onChange={(e) => set({ colorize: e.target.checked })}
        />
        <span>Colorize (black &amp; white → color)</span>
      </div>

      <div className="toggle">
        <input
          type="checkbox"
          checked={operations.upscale}
          onChange={(e) => set({ upscale: e.target.checked })}
        />
        <span>Upscale</span>
        {operations.upscale && (
          <select
            value={operations.upscale_factor}
            onChange={(e) =>
              set({ upscale_factor: Number(e.target.value) as 2 | 4 })
            }
          >
            <option value={2}>×2</option>
            <option value={4}>×4</option>
          </select>
        )}
      </div>

      <div className="toggle">
        <input
          type="checkbox"
          checked={operations.face_restore}
          onChange={(e) => set({ face_restore: e.target.checked })}
        />
        <span>Face restore</span>
      </div>
      {operations.face_restore && (
        <div>
          <label>
            Fidelity ({operations.face_fidelity.toFixed(2)}) — higher = closer to
            original
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={operations.face_fidelity}
            onChange={(e) => set({ face_fidelity: Number(e.target.value) })}
            style={{ width: '100%' }}
          />
        </div>
      )}

      <div className="row" style={{ marginTop: 16 }}>
        <button onClick={testConnection}>Test</button>
        <button
          className="primary"
          disabled={!forgeUrl}
          onClick={() => onSave({ forgeUrl, forgeToken, operations })}
        >
          Save
        </button>
        {test && <span className="muted">{test}</span>}
      </div>
    </div>
  )
}
