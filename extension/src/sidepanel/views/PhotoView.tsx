import { useEffect, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import {
  describeOperations,
  toRequest,
  withoutUnavailable,
} from '../../lib/operations'
import type { ForgeOperations, ImmichAsset, JobInfo, ModelsInfo } from '../../lib/types'
import { EnhancementControls } from '../../ui/EnhancementControls'
import type { AssetImage } from '../useAssetImage'

// The photo screen: preview, this photo's enhancement choices, and a sticky
// Forge button. Choices start from the defaults set on the options page and
// apply to this photo only.
export function PhotoView({
  client,
  asset,
  preview,
  operations,
  onOperationsChange,
  models,
  onSubmitted,
}: {
  client: ForgeClient
  asset: ImmichAsset
  preview: AssetImage
  operations: ForgeOperations
  onOperationsChange: (ops: ForgeOperations) => void
  models: ModelsInfo | null
  onSubmitted: (job: JobInfo, ops: ForgeOperations) => void
}) {
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lowQuality, setLowQuality] = useState(false)

  // What will actually run: options the server can't do are dropped.
  const effective = withoutUnavailable(operations, models)
  const summary = describeOperations(effective)
  const nothingSelected = summary.length === 0
  const notImage = !!asset.type && asset.type !== 'IMAGE'

  // Best-effort source-quality precheck: warn (softly, never blocking) when the
  // original is too degraded for the enhancement to deliver much. Any failure
  // (server down, older server without /analyze) just leaves the warning off.
  useEffect(() => {
    if (notImage) return
    let live = true
    client
      .analyze(asset.id, toRequest(effective))
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
      onSubmitted(await client.forge(asset.id, toRequest(effective)), effective)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const w = asset.exifInfo?.exifImageWidth
  const h = asset.exifInfo?.exifImageHeight

  return (
    <>
      <div className="body">
        {preview.src ? (
          <img className="preview" src={preview.src} onError={preview.onError} alt="" />
        ) : (
          <div className="preview placeholder" />
        )}
        <p className="caption muted">
          {asset.originalFileName}
          {w && h ? ` · ${w}×${h}` : ''}
        </p>

        {notImage && (
          <p className="error">
            This asset is a {asset.type.toLowerCase()}; only images can be forged.
          </p>
        )}
        {lowQuality && !notImage && (
          <p className="note">This original has low detail, so results may be limited.</p>
        )}

        <EnhancementControls
          operations={operations}
          onChange={onOperationsChange}
          models={models}
        />

      </div>

      <div className="actionbar">
        {error && <p className="error actionbar-error">{error}</p>}
        <p className="actionbar-summary muted">
          {nothingSelected ? 'Turn on at least one enhancement' : summary.join(' · ')}
        </p>
        <button
          className="primary"
          disabled={submitting || nothingSelected || notImage}
          onClick={submit}
        >
          {submitting ? 'Submitting…' : 'Forge'}
        </button>
      </div>
    </>
  )
}
