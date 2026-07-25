import { useEffect, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import type { ForgeOperations, ImmichAsset, JobInfo } from '../../lib/types'
import { useAssetImage } from '../useAssetImage'

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
  const { src: preview, onError: onPreviewError } = useAssetImage(client, asset.id)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lowQuality, setLowQuality] = useState(false)

  const summary = [
    operations.denoise && (operations.low_light ? 'denoise + low-light' : 'denoise'),
    operations.colorize && 'colorize',
    operations.upscale && `upscale ×${operations.upscale_factor}`,
    operations.face_restore && 'face restore',
  ]
    .filter(Boolean)
    .join(' · ')
  const nothingSelected = !summary
  const notImage = !!asset.type && asset.type !== 'IMAGE'

  // Best-effort source-quality precheck: warn (softly, never blocking) when the
  // original is too degraded for the enhancement to deliver much. Any failure
  // (server down, older server without /analyze) just leaves the warning off.
  useEffect(() => {
    if (notImage) return
    let live = true
    client
      .analyze(asset.id, operations)
      .then((a) => live && setLowQuality(a.low_quality))
      .catch(() => live && setLowQuality(false))
    return () => {
      live = false
    }
  }, [asset.id, notImage])

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
      {lowQuality && !notImage && (
        <p className="note">
          This original has low detail, enhancement results may be limited.
        </p>
      )}

      <div className="row between" style={{ marginTop: 12 }}>
        <span className="muted">{asset.originalFileName}</span>
        {summary && <span className="muted">{summary}</span>}
      </div>
      {preview && (
        <img
          src={preview}
          onError={onPreviewError}
          style={{ width: '100%', borderRadius: 8, marginTop: 8 }}
        />
      )}
    </div>
  )
}
