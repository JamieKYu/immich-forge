import { useEffect, useRef, useState } from 'react'
import type { ForgeClient } from '../../lib/forge-client'
import { describeOperations, stageLabel } from '../../lib/operations'
import type { ForgeOperations, JobInfo } from '../../lib/types'
import { reloadActiveTab } from '../../lib/tabs'
import type { AssetImage } from '../useAssetImage'
import { Compare } from './Compare'

export function ReviewView({
  client,
  initialJob,
  operations,
  before,
  onBack,
}: {
  client: ForgeClient
  initialJob: JobInfo
  // What this job was submitted with (for labels and the summary).
  operations: ForgeOperations
  before: AssetImage
  // Back to the photo screen with the same choices. An unsaved result is
  // simply left behind; nothing reaches Immich without "Save to Immich".
  onBack: () => void
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

  if (accepted) {
    return (
      <>
        <div className="body">
          <p>Saved. The forged image is now the primary photo in its Immich stack.</p>
        </div>
        <div className="actionbar">
          <button className="primary" onClick={onBack}>
            Done
          </button>
        </div>
      </>
    )
  }

  const failed = job.status === 'error' || !!error

  return (
    <>
      <div className="body">
        {error && <p className="error" style={{ marginTop: 0 }}>{error}</p>}
        {job.notes?.map((note, i) => (
          <p key={i} className="note">
            {note}
          </p>
        ))}

        {after ? (
          <Compare before={before.src} after={after} onBeforeError={before.onError} />
        ) : !failed ? (
          // While forging, show live progress in place of the image.
          <div className="processing">
            <span className="muted">
              {job.status === 'done'
                ? 'Loading result'
                : stageLabel(job.status === 'queued' ? 'queued' : job.stage, operations)}
              …
            </span>
            <span className="pct">{Math.round(job.progress * 100)}%</span>
            <div className="bar">
              <div style={{ width: `${Math.max(5, job.progress * 100)}%` }} />
            </div>
          </div>
        ) : null}

        <p className="caption muted">{describeOperations(operations).join(' · ')}</p>
      </div>

      {/* Primary action rightmost (macOS / Chrome convention). */}
      <div className="actionbar">
        <button onClick={onBack} disabled={accepting} title="Throw this result away and go back to the settings">
          Discard
        </button>
        <button
          className="accept"
          disabled={job.status !== 'done' || !after || accepting}
          onClick={accept}
        >
          {accepting ? 'Saving…' : 'Save to Immich'}
        </button>
      </div>
    </>
  )
}
