import { useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import { describeOperations, toRequest, withoutUnavailable } from '../../lib/operations'
import type {
  ForgeOperations,
  ImmichAsset,
  JobInfo,
  Mode,
  ModelsInfo,
  OperationKey,
} from '../../lib/types'
import { EnhancementControls } from '../../ui/EnhancementControls'
import type { AssetImage } from '../useAssetImage'

// Result of /analyze for this photo. `suggested` is null when the server
// doesn't do Auto mode (or the call failed).
export type PhotoAnalysis =
  | { state: 'loading' }
  | {
      state: 'done'
      lowQuality: boolean
      suggested: ForgeOperations | null
      reasons: Partial<Record<OperationKey, string>>
    }

// The photo screen: preview, the Auto | Custom switch, this photo's
// enhancements, and a sticky Forge button. Auto shows the server's
// suggestions with a reason per option; Custom starts from the defaults on the
// options page. Either way, changes apply to this photo only.
export function PhotoView({
  client,
  asset,
  preview,
  models,
  analysis,
  mode,
  autoAvailable,
  adjustedFromAuto,
  operations,
  onOperationsChange,
  onModeChange,
  onSubmitted,
}: {
  client: ForgeClient
  asset: ImmichAsset
  preview: AssetImage
  models: ModelsInfo | null
  analysis: PhotoAnalysis
  // The mode in effect for this photo.
  mode: Mode
  // False when the server can't suggest (older server): no switch is shown.
  autoAvailable: boolean
  // Custom, but started from the suggestions (the user changed an option).
  adjustedFromAuto: boolean
  // What will run; null while Auto is still analyzing.
  operations: ForgeOperations | null
  onOperationsChange: (ops: ForgeOperations) => void
  onModeChange: (mode: Mode) => void
  onSubmitted: (job: JobInfo, ops: ForgeOperations) => void
}) {
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // What will actually run: options the server can't do are dropped.
  const effective = operations && withoutUnavailable(operations, models)
  const summary = effective ? describeOperations(effective) : []
  const nothingSelected = !!effective && summary.length === 0
  const notImage = !!asset.type && asset.type !== 'IMAGE'
  const done = analysis.state === 'done' ? analysis : null
  // Auto's reasons replace the help text while the suggestions are in play.
  // Highlight the ones Auto suggested, not whatever the switches show now.
  const notes =
    done?.suggested && (mode === 'auto' || adjustedFromAuto)
      ? Object.fromEntries(
          Object.entries(done.reasons).map(([op, text]) => [
            op,
            { text: text!, suggested: !!done.suggested![op as OperationKey] },
          ]),
        )
      : undefined

  async function submit() {
    if (!effective) return
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

  // One line beside the switch; Auto's verdicts are green like the suggestions.
  const caption: { text: string; suggested?: boolean } | null =
    mode === 'auto'
      ? !done
        ? { text: 'Analyzing this photo…' }
        : nothingSelected
          ? { text: 'This photo looks good as it is', suggested: true }
          : { text: 'Suggested for this photo', suggested: true }
      : adjustedFromAuto
        ? { text: 'Adjusted from the suggestions' }
        : null

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
        {done?.lowQuality && !notImage && (
          <p className="note">This original has low detail, so results may be limited.</p>
        )}

        {autoAvailable && (
          <div className="mode-row">
            <div className="seg" role="radiogroup" aria-label="Mode">
              {(['auto', 'custom'] as const).map((m) => (
                <label key={m} className={mode === m ? 'active' : ''}>
                  <input
                    type="radio"
                    name="mode"
                    checked={mode === m}
                    onChange={() => onModeChange(m)}
                  />
                  {m === 'auto' ? 'Auto' : 'Custom'}
                </label>
              ))}
            </div>
            {caption && (
              <span className={`mode-caption${caption.suggested ? ' suggested' : ' muted'}`}>
                {caption.text}
              </span>
            )}
          </div>
        )}

        {operations ? (
          <EnhancementControls
            operations={operations}
            onChange={onOperationsChange}
            models={models}
            notes={notes}
          />
        ) : (
          <div className="analyzing" aria-busy="true" />
        )}
      </div>

      <div className="actionbar">
        {error && <p className="error actionbar-error">{error}</p>}
        <p className="actionbar-summary muted">
          {!effective
            ? 'Analyzing…'
            : nothingSelected
              ? mode === 'auto'
                ? 'Nothing suggested. Switch to Custom to choose'
                : 'Turn on at least one enhancement'
              : summary.join(' · ')}
        </p>
        <button
          className="primary"
          disabled={submitting || !effective || nothingSelected || notImage}
          onClick={submit}
        >
          {submitting ? 'Submitting…' : 'Forge'}
        </button>
      </div>
    </>
  )
}
