import { useEffect, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import type { ForgeOperations, ImmichAsset, JobInfo } from '../../lib/types'

export function ConfigureView({
  client,
  asset,
  operations,
  onSubmitted,
}: {
  client: ForgeClient
  asset: ImmichAsset
  operations: ForgeOperations
  onSubmitted: (j: JobInfo) => void
}) {
  const [thumb, setThumb] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    client.thumbnailDataUrl(asset.id).then(setThumb).catch(() => {})
  }, [asset.id])

  const summary = [
    operations.colorize && 'colorize',
    operations.upscale && `upscale ×${operations.upscale_factor}`,
    operations.face_restore && 'face restore',
  ]
    .filter(Boolean)
    .join(' · ')
  const nothingSelected = !summary
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
      <button
        className="primary"
        style={{ width: '100%' }}
        disabled={submitting || nothingSelected || notImage}
        onClick={submit}
      >
        {submitting ? 'Submitting…' : 'Forge'}
      </button>

      {nothingSelected && (
        <p className="muted" style={{ marginTop: 8 }}>
          No enhancements enabled — turn some on in settings.
        </p>
      )}
      {notImage && (
        <p className="error">
          This asset is a {asset.type.toLowerCase()}; only images can be forged.
        </p>
      )}
      {error && <p className="error">{error}</p>}

      <div className="row between" style={{ marginTop: 12 }}>
        <span className="muted">{asset.originalFileName}</span>
        {summary && <span className="muted">{summary}</span>}
      </div>
      {thumb && (
        <img src={thumb} style={{ width: '100%', borderRadius: 8, marginTop: 8 }} />
      )}
    </div>
  )
}
