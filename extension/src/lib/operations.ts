// Pure helpers around ForgeOperations: what the UI shows vs. what the server
// is sent, plain-language labels, and server capability checks.

import {
  DEFAULT_OPERATIONS,
  type ForgeAnalysis,
  type ForgeOperations,
  type ModelsInfo,
} from './types'

// The UI treats "Reduce noise" and "Brighten dark photo" as independent
// switches, but the server only brightens inside its denoise stage (low_light
// runs first, then the denoised result is blended back toward the brightened
// input by denoise_strength). So "brighten only" is sent as denoise at
// strength 0 — brightened, not denoised. Works on every server version.
export function toRequest(ops: ForgeOperations): ForgeOperations {
  if (ops.low_light && !ops.denoise) {
    return { ...ops, denoise: true, denoise_strength: 0 }
  }
  return ops
}

// The server's suggested operations as a complete set, or null when the server
// doesn't do Auto mode (older servers omit `suggested`).
export function suggestedOperations(analysis: ForgeAnalysis | null): ForgeOperations | null {
  if (!analysis?.suggested) return null
  return { ...DEFAULT_OPERATIONS, ...analysis.suggested }
}

// Plain-language summary of what will run, in the same order as the controls.
export function describeOperations(ops: ForgeOperations): string[] {
  return [
    ops.dejpeg && 'JPEG cleanup',
    ops.denoise && 'reduce noise',
    ops.low_light && 'brighten',
    ops.upscale && `enlarge ×${ops.upscale_factor}`,
    ops.face_restore && 'restore faces',
    ops.colorize && 'colorize',
  ].filter((x): x is string => !!x)
}

// Toggleable operations that a server can lack or have turned off, mapped to
// their /models key.
const GATED = {
  dejpeg: 'dejpeg',
  denoise: 'denoise',
  face_restore: 'face',
  colorize: 'colorize',
} as const
export type GatedOperation = keyof typeof GATED

// Oldest server release that ships each stage (for the "update" hint).
const MIN_SERVER: Partial<Record<GatedOperation, string>> = { dejpeg: 'v0.0.9' }

// Why `op` can't be used on this server, or null if it can. With no /models
// info (not yet loaded, or the call failed) everything is assumed available.
export function unavailableReason(
  models: ModelsInfo | null,
  op: GatedOperation,
): string | null {
  if (!models) return null
  const backend = models[GATED[op]]
  if (backend === undefined) {
    const min = MIN_SERVER[op]
    return min
      ? `Needs Forge server ${min} or newer`
      : 'Not supported by your Forge server'
  }
  if (backend === 'none') return 'Turned off on your Forge server'
  return null
}

// Switch off anything the server can't do, so the summary and the request
// match what will actually run.
export function withoutUnavailable(
  ops: ForgeOperations,
  models: ModelsInfo | null,
): ForgeOperations {
  const out = { ...ops }
  for (const op of Object.keys(GATED) as GatedOperation[]) {
    if (unavailableReason(models, op)) out[op] = false
  }
  return out
}

// Human-readable progress label for a job's `stage` (or status).
export function stageLabel(stage: string | null, ops: ForgeOperations): string {
  switch (stage) {
    case null:
    case 'queued':
      return 'Waiting in queue'
    case 'starting':
      return 'Starting'
    case 'dejpeg':
      return 'Cleaning up JPEG'
    case 'denoise':
      if (ops.low_light && !ops.denoise) return 'Brightening'
      return ops.low_light ? 'Brightening & reducing noise' : 'Reducing noise'
    case 'colorize':
      return 'Colorizing'
    case 'face_restore':
      return 'Restoring faces'
    case 'upscale':
      return ops.face_restore ? 'Enlarging & restoring faces' : 'Enlarging'
    case 'encoding':
      return 'Finishing up'
    default:
      return stage.charAt(0).toUpperCase() + stage.slice(1).replace(/_/g, ' ')
  }
}
