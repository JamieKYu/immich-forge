import { useRef, useState } from 'react'

// Before/after comparison: the forged image with the original laid over it,
// clipped at a divider. Drag on the image (or use the slider) to move the
// divider. Both images are fitted to the panel width.
export function Compare({
  before,
  after,
  onBeforeError,
}: {
  before: string | null
  after: string
  onBeforeError?: () => void
}) {
  const [pos, setPos] = useState(50) // divider, % from the left
  const inner = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  // Divider follows the pointer, measured against the image box.
  function moveTo(clientX: number) {
    const rect = inner.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }

  return (
    <div>
      <div className="compare-frame">
        <div
          ref={inner}
          className="compare-inner"
          onPointerDown={(e) => {
            if (e.button !== 0) return
            dragging.current = true
            e.currentTarget.setPointerCapture(e.pointerId)
            moveTo(e.clientX)
          }}
          onPointerMove={(e) => dragging.current && moveTo(e.clientX)}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
        >
          <img src={after} alt="Forged" draggable={false} />
          {before && (
            <img
              className="before"
              src={before}
              alt="Original"
              draggable={false}
              onError={onBeforeError}
              style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
            />
          )}
          {before && <div className="divider" style={{ left: `${pos}%` }} />}
        </div>
        {before && <span className="chip left">Before</span>}
        <span className="chip right">After</span>
      </div>

      <div className="compare-controls">
        <input
          type="range"
          aria-label="Before / after divider"
          min={0}
          max={100}
          value={pos}
          disabled={!before}
          onChange={(e) => setPos(Number(e.target.value))}
        />
      </div>
    </div>
  )
}
