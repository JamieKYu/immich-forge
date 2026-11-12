// Mirrors the Forge server's Pydantic schemas.

export interface ForgeOperations {
  dejpeg: boolean // FBCNN JPEG artifact removal; runs first
  denoise: boolean
  denoise_strength: number // 0..1 (blend denoised result toward original)
  low_light: boolean // HVI-CIDNet brighten (CLAHE fallback), before denoise
  colorize: boolean
  upscale: boolean
  upscale_factor: 2 | 4
  face_restore: boolean
  face_fidelity: number // 0..1
}

export type JobStatus = 'queued' | 'running' | 'done' | 'error'

export interface JobInfo {
  job_id: string
  asset_id: string
  status: JobStatus
  progress: number
  stage: string | null
  error: string | null
  notes: string[]
  new_asset_id: string | null
  stack_id: string | null
}

export interface AcceptResponse {
  new_asset_id: string
  stack_id: string
}

// Pre-submit source-quality precheck. `low_quality` drives a soft, non-blocking
// warning; `metrics` is raw debug data the UI ignores.
export interface ForgeAnalysis {
  low_quality: boolean
  metrics: Record<string, number>
}

export interface ImmichAsset {
  id: string
  originalFileName: string
  type: string
  exifInfo?: { exifImageWidth?: number; exifImageHeight?: number }
}

export interface Settings {
  forgeUrl: string // e.g. http://gpu-host:8000
  forgeToken: string
  // Default enhancements every photo starts from. Edited on the options page.
  operations: ForgeOperations
  // Stored-settings schema version; see storage.ts for migrations.
  version?: number
}

// GET /models: active backend per stage, e.g. { dejpeg: 'fbcnn', face: 'none' }.
// A stage key missing entirely means the server predates that stage.
export type ModelsInfo = Record<string, string>

export const DEFAULT_OPERATIONS: ForgeOperations = {
  dejpeg: false,
  denoise: false,
  denoise_strength: 1,
  low_light: false,
  colorize: false,
  upscale: true,
  upscale_factor: 4,
  face_restore: false,
  face_fidelity: 0.95, // lean strongly toward fidelity: lower values let CodeFormer reinvent eyes (mismatched iris colour)
}
