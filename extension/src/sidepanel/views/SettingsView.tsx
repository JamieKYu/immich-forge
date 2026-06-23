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
  const [showToken, setShowToken] = useState(false)

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
      <div className="input-wrap">
        <input
          type={showToken ? 'text' : 'password'}
          placeholder="Bearer token (leave blank if auth disabled)"
          value={forgeToken}
          onChange={(e) => setForgeToken(e.target.value)}
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

      <label style={{ marginTop: 16 }}>Default enhancements</label>

      <div className="setting">
        <span>Colorize</span>
        <Switch checked={operations.colorize} onChange={(v) => set({ colorize: v })} />
      </div>

      <div className="setting">
        <span>Upscale</span>
        <Switch checked={operations.upscale} onChange={(v) => set({ upscale: v })} />
      </div>
      {operations.upscale && (
        <div className="suboption seg">
          {([2, 4] as const).map((f) => (
            <label key={f} className={operations.upscale_factor === f ? 'active' : ''}>
              <input
                type="radio"
                name="upscale_factor"
                checked={operations.upscale_factor === f}
                onChange={() => set({ upscale_factor: f })}
              />
              ×{f}
            </label>
          ))}
        </div>
      )}

      <div className="setting">
        <span>Face restore</span>
        <Switch
          checked={operations.face_restore}
          onChange={(v) => set({ face_restore: v })}
        />
      </div>
      {operations.face_restore && (
        <div className="suboption">
          <label style={{ margin: '0 0 4px' }}>
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

function Switch({
  checked,
  onChange,
}: {
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <label className="switch">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="slider" />
    </label>
  )
}

function EyeIcon() {
  return (
    <svg
      width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    >
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  )
}

function EyeOffIcon() {
  return (
    <svg
      width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    >
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  )
}
