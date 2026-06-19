import { useEffect, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import type { ForgeOperations, ImmichAsset, JobInfo } from '../../lib/types'

export function ConfigureView({
  client,
  asset,
  operations,
  setOperations,
  onSubmitted,
  onBack,
}: {
  client: ForgeClient
  asset: ImmichAsset
  operations: ForgeOperations
  setOperations: (o: ForgeOperations) => void
  onSubmitted: (j: JobInfo) => void
  onBack?: () => void
}) {
  const [thumb, setThumb] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    client.thumbnailDataUrl(asset.id).then(setThumb).catch(() => {})
  }, [asset.id])

  const set = (patch: Partial<ForgeOperations>) =>
    setOperations({ ...operations, ...patch })

  const nothingSelected =
    !operations.upscale && !operations.face_restore && !operations.colorize
  const notImage = !!asset.type && asset.type !== 'IMAGE'

  async function submit() {
    setSubmitting(true)
    setError(null)
    try {
      onSubmitted(await client.forge(asset.id, operations))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <div className="row between">
        {onBack ? <button onClick={onBack}>← Back</button> : <span />}
        <span className="muted">{asset.originalFileName}</span>
      </div>
      {thumb && <img src={thumb} style={{ width: '100%', borderRadius: 8, margin: '10px 0' }} />}
      {asset.type && asset.type !== 'IMAGE' && (
        <p className="error">This asset is a {asset.type.toLowerCase()}; only images can be forged.</p>
      )}

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

      {error && <p className="error">{error}</p>}
      <button
        className="primary"
        style={{ width: '100%', marginTop: 14 }}
        disabled={submitting || nothingSelected || notImage}
        onClick={submit}
      >
        {submitting ? 'Submitting…' : 'Forge'}
      </button>
    </div>
  )
}
