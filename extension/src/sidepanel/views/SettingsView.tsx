import { useEffect, useRef, useState } from 'react'
import { DEFAULT_OPERATIONS, type ForgeOperations, type Settings } from '../../lib/types'

export function SettingsView({
  settings,
  onChange,
}: {
  settings: Settings | null
  onChange: (s: Settings) => void
}) {
  const [forgeUrl, setForgeUrl] = useState(settings?.forgeUrl ?? '')
  const [forgeToken, setForgeToken] = useState(settings?.forgeToken ?? '')
  const [operations, setOperations] = useState<ForgeOperations>(
    settings?.operations ?? DEFAULT_OPERATIONS,
  )

  const set = (patch: Partial<ForgeOperations>) =>
    setOperations((o) => ({ ...o, ...patch }))

  // Report the current values up so the header "close" link can save them.
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  useEffect(() => {
    onChangeRef.current({ forgeUrl, forgeToken, operations })
  }, [forgeUrl, forgeToken, operations])

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
      <label style={{ marginTop: 16 }}>Default enhancements</label>

      <div className="toggle">
        <input
          type="checkbox"
          checked={operations.colorize}
          onChange={(e) => set({ colorize: e.target.checked })}
        />
        <span>Colorize</span>
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
            Fidelity ({operations.face_fidelity.toFixed(2)}) — higher is closer to
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
    </div>
  )
}
