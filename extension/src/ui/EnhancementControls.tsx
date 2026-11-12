import type { ReactNode } from 'react'
import { type GatedOperation, unavailableReason } from '../lib/operations'
import type { ForgeOperations, ModelsInfo } from '../lib/types'
import { Switch } from './Switch'

// The enhancement options, grouped by intent rather than pipeline order:
// "Repair" what's wrong with the photo, then "Enhance" it. Each slider sits
// behind an "Adjust strength" disclosure; right is always the stronger effect,
// and the endpoints name what you'll see rather than a setting.
// Shared by the side panel (per-photo choices) and the options page (defaults).
export function EnhancementControls({
  operations,
  onChange,
  models,
}: {
  operations: ForgeOperations
  onChange: (next: ForgeOperations) => void
  // Server capabilities; null = unknown (everything shown as available).
  models: ModelsInfo | null
}) {
  const set = (patch: Partial<ForgeOperations>) => onChange({ ...operations, ...patch })
  const reason = (op: GatedOperation) => unavailableReason(models, op)

  return (
    <div className="controls">
      <h2 className="group">Repair</h2>
      <div className="group-card">

        <Option
          title="Clean up JPEG blocks"
          help="Removes blocky compression artifacts. Can soften photos that are already clean."
          unavailable={reason('dejpeg')}
          control={
            <Switch
              label="Clean up JPEG blocks"
              checked={operations.dejpeg && !reason('dejpeg')}
              disabled={!!reason('dejpeg')}
              onChange={(v) => set({ dejpeg: v })}
            />
          }
        />

        <Option
          title="Reduce noise"
          help="Smooths grain from high-ISO shots and old scans."
          unavailable={reason('denoise')}
          control={
            <Switch
              label="Reduce noise"
              checked={operations.denoise && !reason('denoise')}
              disabled={!!reason('denoise')}
              onChange={(v) => set({ denoise: v })}
            />
          }
        >
          {operations.denoise && !reason('denoise') && (
            <AdjustStrength>
              <Slider
                label="Noise reduction strength"
                left="Keep detail"
                right="Smoother"
                min={0.1}
                value={operations.denoise_strength}
                onChange={(v) => set({ denoise_strength: v })}
              />
            </AdjustStrength>
          )}
        </Option>

        <Option
          title="Brighten dark photo"
          help="Lifts underexposed shots. Pair with Reduce noise to clean up what brightening reveals."
          control={
            <Switch
              label="Brighten dark photo"
              checked={operations.low_light}
              onChange={(v) => set({ low_light: v })}
            />
          }
        />
      </div>

      <h2 className="group">Enhance</h2>
      <div className="group-card">

        <Option
          title="Enlarge"
          help="More pixels for printing and zooming in."
          control={
            <div className="seg" role="radiogroup" aria-label="Enlarge">
              {(['off', 2, 4] as const).map((f) => {
                const active = f === 'off' ? !operations.upscale : operations.upscale && operations.upscale_factor === f
                return (
                  <label key={f} className={active ? 'active' : ''}>
                    <input
                      type="radio"
                      name="enlarge"
                      checked={active}
                      onChange={() =>
                        set(f === 'off' ? { upscale: false } : { upscale: true, upscale_factor: f })
                      }
                    />
                    {f === 'off' ? 'Off' : `×${f}`}
                  </label>
                )
              })}
            </div>
          }
        />

        <Option
          title="Restore faces"
          help="Sharpens soft or blurry faces."
          unavailable={reason('face_restore')}
          control={
            <Switch
              label="Restore faces"
              checked={operations.face_restore && !reason('face_restore')}
              disabled={!!reason('face_restore')}
              onChange={(v) => set({ face_restore: v })}
            />
          }
        >
          {operations.face_restore && !reason('face_restore') && (
            <AdjustStrength>
              {/* Shown as strength = 1 - face_fidelity (fidelity 1 = closest to
                  the original face), so right is stronger like the other slider. */}
              <Slider
                label="Face restoration strength"
                left="Closer to original"
                right="Sharper"
                min={0}
                value={1 - operations.face_fidelity}
                onChange={(v) => set({ face_fidelity: Math.round((1 - v) * 100) / 100 })}
                note="Higher can change how the person looks."
              />
            </AdjustStrength>
          )}
        </Option>

        <Option
          title="Colorize"
          help="Adds color to black-and-white photos."
          unavailable={reason('colorize')}
          control={
            <Switch
              label="Colorize"
              checked={operations.colorize && !reason('colorize')}
              disabled={!!reason('colorize')}
              onChange={(v) => set({ colorize: v })}
            />
          }
        />
      </div>
    </div>
  )
}

function Option({
  title,
  help,
  unavailable,
  control,
  children,
}: {
  title: string
  help: string
  unavailable?: string | null
  control: ReactNode
  children?: ReactNode
}) {
  return (
    <div className={`option${unavailable ? ' unavailable' : ''}`}>
      <div className="option-head">
        <span className="option-title">{title}</span>
        {control}
      </div>
      <p className="option-help">{unavailable ?? help}</p>
      {children}
    </div>
  )
}

function AdjustStrength({ children }: { children: ReactNode }) {
  return (
    <details className="finetune">
      <summary>Adjust strength</summary>
      {children}
    </details>
  )
}

function Slider({
  label,
  left,
  right,
  min,
  value,
  onChange,
  note,
}: {
  label: string
  left: string
  right: string
  min: number
  value: number
  onChange: (v: number) => void
  // Optional caveat shown under the slider.
  note?: string
}) {
  return (
    <>
      <div className="slider-row">
        <span className="muted">{left}</span>
        <input
          type="range"
          aria-label={label}
          min={min}
          max={1}
          step={0.05}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <span className="muted">{right}</span>
      </div>
      {note && <p className="slider-note">{note}</p>}
    </>
  )
}
