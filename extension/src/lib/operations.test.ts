import { describe, expect, it } from 'vitest'
import {
  describeOperations,
  stageLabel,
  suggestedOperations,
  toRequest,
  unavailableReason,
  withoutUnavailable,
} from './operations'
import { DEFAULT_OPERATIONS, type ForgeOperations } from './types'

const ops = (over: Partial<ForgeOperations> = {}): ForgeOperations => ({
  ...DEFAULT_OPERATIONS,
  ...over,
})

describe('toRequest', () => {
  it('sends brighten-only as denoise at strength 0', () => {
    const r = toRequest(ops({ low_light: true, denoise: false, denoise_strength: 0.7 }))
    expect(r).toMatchObject({ low_light: true, denoise: true, denoise_strength: 0 })
  })

  it('leaves brighten + denoise untouched', () => {
    const o = ops({ low_light: true, denoise: true, denoise_strength: 0.7 })
    expect(toRequest(o)).toEqual(o)
  })

  it('leaves denoise-only untouched', () => {
    const o = ops({ denoise: true, denoise_strength: 0.4 })
    expect(toRequest(o)).toEqual(o)
  })
})

describe('suggestedOperations', () => {
  it('is null when the server returned no suggestions (older server)', () => {
    expect(suggestedOperations(null)).toBeNull()
    expect(suggestedOperations({ low_quality: false, metrics: {} })).toBeNull()
    expect(suggestedOperations({ low_quality: false, metrics: {}, suggested: null })).toBeNull()
  })

  it('fills any fields the server left out from the defaults', () => {
    const partial = { colorize: true } as unknown as ForgeOperations
    const ops = suggestedOperations({ low_quality: false, metrics: {}, suggested: partial })
    expect(ops).toEqual({ ...DEFAULT_OPERATIONS, colorize: true })
  })
})

describe('describeOperations', () => {
  it('lists enabled operations in control order', () => {
    expect(
      describeOperations(
        ops({ dejpeg: true, denoise: true, low_light: true, face_restore: true, colorize: true }),
      ),
    ).toEqual(['JPEG cleanup', 'reduce noise', 'brighten', 'enlarge ×4', 'restore faces', 'colorize'])
  })

  it('is empty when nothing is on', () => {
    expect(describeOperations(ops({ upscale: false }))).toEqual([])
  })
})

describe('unavailableReason', () => {
  const full = { dejpeg: 'fbcnn', denoise: 'scunet', face: 'codeformer', colorize: 'ddcolor' }

  it('assumes everything is available when capabilities are unknown', () => {
    expect(unavailableReason(null, 'dejpeg')).toBeNull()
  })

  it('names the minimum server version for a missing stage', () => {
    const { dejpeg: _, ...old } = full
    expect(unavailableReason(old, 'dejpeg')).toBe('Needs Forge server v0.0.9 or newer')
  })

  it('reports a stage turned off on the server', () => {
    expect(unavailableReason({ ...full, face: 'none' }, 'face_restore')).toBe(
      'Turned off on your Forge server',
    )
  })

  it('returns null for an active backend', () => {
    expect(unavailableReason(full, 'colorize')).toBeNull()
  })
})

describe('withoutUnavailable', () => {
  it('switches off operations the server cannot run', () => {
    const r = withoutUnavailable(ops({ dejpeg: true, colorize: true }), {
      denoise: 'scunet',
      face: 'codeformer',
      colorize: 'none',
    })
    expect(r.dejpeg).toBe(false)
    expect(r.colorize).toBe(false)
    expect(r.upscale).toBe(true)
  })
})

describe('stageLabel', () => {
  it('maps server stages to plain words', () => {
    expect(stageLabel('dejpeg', ops())).toBe('Cleaning up JPEG')
    expect(stageLabel('encoding', ops())).toBe('Finishing up')
    expect(stageLabel(null, ops())).toBe('Waiting in queue')
  })

  it('describes the denoise stage by what it is doing', () => {
    expect(stageLabel('denoise', ops({ denoise: true }))).toBe('Reducing noise')
    expect(stageLabel('denoise', ops({ low_light: true }))).toBe('Brightening')
    expect(stageLabel('denoise', ops({ denoise: true, low_light: true }))).toBe(
      'Brightening & reducing noise',
    )
  })

  it('mentions faces when upscale runs the combined stage', () => {
    expect(stageLabel('upscale', ops({ face_restore: true }))).toBe('Enlarging & restoring faces')
  })

  it('falls back to a readable version of unknown stages', () => {
    expect(stageLabel('new_stage', ops())).toBe('New stage')
  })
})
