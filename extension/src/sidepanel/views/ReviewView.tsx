import { useEffect, useRef, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import type { ForgeOperations, JobInfo } from '../../lib/types'
import { reloadActiveTab } from '../../lib/tabs'

export function ReviewView({
  client,
  initialJob,
  operations,
  onReforge,
  onDone,
}: {
  client: ForgeClient
  initialJob: JobInfo
  operations: ForgeOperations
  onReforge: () => void
  onDone: () => void
}) {
  const [job, setJob] = useState<JobInfo>(initialJob)
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
      // Refresh the Immich page so the newly stacked primary shows up.
      void reloadActiveTab()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setAccepting(false)
    }
  }

  const ops = [
    operations.denoise && (operations.low_light ? 'denoise + low-light' : 'denoise'),
    operations.colorize && 'colorize',
    operations.upscale && `upscale ×${operations.upscale_factor}`,
    operations.face_restore && 'face restore',
  ].filter(Boolean).join(' · ')

  if (accepted) {
    return (
      <div>
        <p>Forged image now stacked as the new primary.</p>
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
        <button
          className="accept"
          style={{ flex: 1 }}
          disabled={job.status !== 'done' || accepting}
          onClick={accept}
        >
          {accepting ? 'Stacking…' : 'Save to Immich'}
        </button>
        <button onClick={onReforge}>Reset</button>
      </div>

      <p className="muted" style={{ marginTop: 8 }}>{ops}</p>

      {job.notes?.map((note, i) => (
        <p key={i} className="note">{note}</p>
      ))}

      <div className="compare" style={{ marginTop: 8 }}>
        <figure>
          {after ? (
            <>
              <img src={after} />
              <figcaption>Forged</figcaption>
            </>
          ) : (
            // While forging, show live progress in place of the image.
            <div className="processing">
              <span className="muted">{job.stage ?? job.status}…</span>
              <span className="pct">{Math.round(job.progress * 100)}%</span>
              <div className="bar">
                <div style={{ width: `${Math.max(5, job.progress * 100)}%` }} />
              </div>
            </div>
          )}
        </figure>
      </div>
    </div>
  )
}
