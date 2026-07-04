// Mirrors the Forge server's Pydantic schemas.

export interface ForgeOperations {
  denoise: boolean
  denoise_strength: number // 0..1 (blend denoised result toward original)
  low_light: boolean // classical CLAHE + gamma brighten, after denoise
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

export interface ImmichAsset {
  id: string
  originalFileName: string
  type: string
  exifInfo?: { exifImageWidth?: number; exifImageHeight?: number }
}

export interface Settings {
  forgeUrl: string // e.g. http://gpu-host:8000
  forgeToken: string
  // Sticky default enhancements, edited on the settings page.
  operations: ForgeOperations
}

export const DEFAULT_OPERATIONS: ForgeOperations = {
  denoise: false,
  denoise_strength: 1,
  low_light: false,
  colorize: false,
  upscale: true,
  upscale_factor: 4,
  face_restore: false,
  face_fidelity: 0.85, // lean strongly toward fidelity: lower values hallucinate eyes/detail


}
