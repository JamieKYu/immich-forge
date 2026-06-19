import { useEffect, useRef, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import type { ForgeOperations, ImmichAsset, JobInfo } from '../../lib/types'

export function ReviewView({
  client,
  asset,
  initialJob,
  operations,
  onReforge,
  onDone,
}: {
  client: ForgeClient
  asset: ImmichAsset
  initialJob: JobInfo
  operations: ForgeOperations
  onReforge: () => void
  onDone: () => void
}) {
  const [job, setJob] = useState<JobInfo>(initialJob)
  const [before, setBefore] = useState<string | null>(null)
  const [after, setAfter] = useState<string | null>(null)
  const [accepting, setAccepting] = useState(false)
  const [accepted, setAccepted] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number>()

  // Poll job status until done/error.
  useEffect(() => {
    client.thumbnailDataUrl(asset.id).then(setBefore).catch(() => {})

    async function poll() {
      try {
        const j = await client.job(initialJob.job_id)
        setJob(j)
        if (j.status === 'done') {
          setAfter(await client.resultDataUrl(j.job_id))
          return
        }
        if (j.status === 'error') {
          setError(j.error ?? 'forge failed')
          return
        }
        timer.current = self.setTimeout(poll, 1500)
      } catch (e) {
        setError((e as Error).message)
      }
    }
    poll()
    return () => clearTimeout(timer.current)
  }, [initialJob.job_id])

  async function accept() {
    setAccepting(true)
    setError(null)
    try {
      const r = await client.accept(job.job_id)
      setAccepted(r.new_asset_id)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setAccepting(false)
    }
  }

  const ops = [
    operations.colorize && 'colorize',
    operations.upscale && `upscale ×${operations.upscale_factor}`,
    operations.face_restore && 'face restore',
  ].filter(Boolean).join(' · ')

  if (accepted) {
    return (
      <div>
        <p>✓ Forged asset stacked as the new primary.</p>
        <p className="muted">New asset id: {accepted}</p>
        <button className="primary" style={{ width: '100%' }} onClick={onDone}>
          Done
        </button>
      </div>
    )
  }

  return (
    <div>
      <p className="muted">{ops}</p>
      {error && <p className="error">{error}</p>}

      {job.status !== 'done' && !error && (
        <div>
          <p className="muted">
            {job.stage ?? job.status}… {Math.round(job.progress * 100)}%
          </p>
          <div className="bar">
            <div style={{ width: `${Math.max(5, job.progress * 100)}%` }} />
          </div>
        </div>
      )}

      <div className="compare" style={{ marginTop: 12 }}>
        <figure>
          <img src={before ?? ''} />
          <figcaption>Before</figcaption>
        </figure>
        <figure>
          {after ? <img src={after} /> : <div className="muted">processing…</div>}
          <figcaption>After</figcaption>
        </figure>
      </div>

      <div className="row" style={{ marginTop: 14 }}>
        <button onClick={onReforge}>Re-forge</button>
        <button
          className="accept"
          style={{ flex: 1 }}
          disabled={job.status !== 'done' || accepting}
          onClick={accept}
        >
          {accepting ? 'Stacking…' : 'Accept & stack as primary'}
        </button>
      </div>
    </div>
  )
}
