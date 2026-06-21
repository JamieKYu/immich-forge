import { useEffect, useRef, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import type { ForgeOperations, ImmichAsset, JobInfo } from '../../lib/types'
import { useAssetImage } from '../useAssetImage'

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
  const { src: before, onError: onBeforeError } = useAssetImage(client, asset.id)
  const [after, setAfter] = useState<string | null>(null)
  const [accepting, setAccepting] = useState(false)
  const [accepted, setAccepted] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number>()

  // Poll job status until done/error.
  useEffect(() => {
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
      {error && <p className="error">{error}</p>}

      <div className="row">
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

      <p className="muted" style={{ marginTop: 8 }}>{ops}</p>

      {job.notes?.map((note, i) => (
        <p key={i} className="note">{note}</p>
      ))}

      {/* After on top, Before on the bottom. */}
      <div className="compare" style={{ marginTop: 8 }}>
        <figure>
          {after ? (
            <img src={after} />
          ) : (
            // While forging, the After slot shows live progress in place of the image.
            <div className="processing">
              <span className="muted">{job.stage ?? job.status}…</span>
              <span className="pct">{Math.round(job.progress * 100)}%</span>
              <div className="bar">
                <div style={{ width: `${Math.max(5, job.progress * 100)}%` }} />
              </div>
            </div>
          )}
          <figcaption>After</figcaption>
        </figure>
        <figure>
          {before ? (
            <img src={before} onError={onBeforeError} />
          ) : (
            <div className="processing">loading…</div>
          )}
          <figcaption>Before</figcaption>
        </figure>
      </div>
    </div>
  )
}
