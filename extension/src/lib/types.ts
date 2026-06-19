// Mirrors the Forge server's Pydantic schemas.

export interface ForgeOperations {
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
  colorize: false,
  upscale: true,
  upscale_factor: 4,
  face_restore: false,
  face_fidelity: 0.5,
}
