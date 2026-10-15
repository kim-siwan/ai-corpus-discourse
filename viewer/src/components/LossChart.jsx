import { useRef, useState } from 'react'

// A small line chart for training curves: one value per epoch per series,
// one y-scale, a dashed line at the best epoch, and a crosshair that reads
// every series at the epoch under the pointer. Plain SVG, no chart library.
//
// series: [{ key, label, points: [[epoch, value], …] }] — colour comes from
// the CSS class lchart-line--<key>, so light and dark themes each get their own.
const W = 560
const M = { l: 42, r: 52, t: 10, b: 26 }

// A round step (1, 2, 2.5 or 5 × 10^k) giving about `n` intervals over the range.
function niceStep(range, n) {
  const raw = range / n
  const mag = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw)
}

function ticks(lo, hi, step) {
  const out = []
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

export default function LossChart({ title, series, best, xMax, height = 190, digits = 2 }) {
  const [hover, setHover] = useState(null)
  const ref = useRef(null)
  const all = series.flatMap((s) => s.points)
  const H = height

  const x1 = Math.max(xMax ?? 0, ...all.map((p) => p[0]), 1)
  let y0 = Math.min(...all.map((p) => p[1]))
  let y1 = Math.max(...all.map((p) => p[1]))
  if (!all.length) [y0, y1] = [0, 1]
  if (y1 - y0 < 1e-6) [y0, y1] = [y0 - 0.5, y1 + 0.5]
  const yStep = niceStep(y1 - y0, 4)
  ;[y0, y1] = [Math.max(0, Math.floor(y0 / yStep) * yStep), Math.ceil(y1 / yStep) * yStep]
  const xStep = Math.max(1, niceStep(x1, 6))

  const sx = (x) => M.l + (x / x1) * (W - M.l - M.r)
  const sy = (y) => M.t + (1 - (y - y0) / (y1 - y0)) * (H - M.t - M.b)
  const path = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('')

  function onMove(e) {
    const box = ref.current?.getBoundingClientRect()
    if (!box || !all.length) return
    const x = ((e.clientX - box.left) / box.width) * W
    const epoch = Math.round(((x - M.l) / (W - M.l - M.r)) * x1)
    const lastEpoch = Math.max(...all.map((p) => p[0]))
    setHover(Math.min(Math.max(epoch, 0), lastEpoch))
  }

  const read = hover === null ? [] : series.map((s) => ({ ...s, value: s.points.find((p) => p[0] === hover)?.[1] }))

  return (
    <figure className="lchart">
      <figcaption className="lchart-head">
        <span className="lchart-title">{title}</span>
        {series.length > 1 && (
          <span className="lchart-legend">
            {series.map((s) => (
              <span key={s.key} className="lchart-key">
                <i className={`lchart-swatch lchart-swatch--${s.key}`} />
                {s.label}
              </span>
            ))}
          </span>
        )}
      </figcaption>
      <div className="lchart-plot">
        <svg
          ref={ref}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`${title}: ${series.map((s) => s.label).join(' and ')} by epoch`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {ticks(y0, y1, yStep).map((y) => (
            <g key={y}>
              <line className="lchart-grid" x1={M.l} x2={W - M.r} y1={sy(y)} y2={sy(y)} />
              <text className="lchart-tick" x={M.l - 6} y={sy(y) + 3} textAnchor="end">
                {y.toFixed(digits)}
              </text>
            </g>
          ))}
          {ticks(0, x1, xStep).map((x) => (
            <text key={x} className="lchart-tick" x={sx(x)} y={H - 8} textAnchor="middle">
              {x}
            </text>
          ))}
          <text className="lchart-tick" x={W - M.r} y={H - 8} dx={14} textAnchor="start">
            epoch
          </text>
          {best != null && (
            <g>
              <line className="lchart-best" x1={sx(best)} x2={sx(best)} y1={M.t} y2={H - M.b} />
              <text className="lchart-tick" x={sx(best) + 4} y={M.t + 9}>
                best {best}
              </text>
            </g>
          )}
          {series.map((s) =>
            s.points.length ? (
              <g key={s.key}>
                <path className={`lchart-line lchart-line--${s.key}`} d={path(s.points)} />
                <text
                  className="lchart-end"
                  x={sx(s.points.at(-1)[0]) + 5}
                  y={sy(s.points.at(-1)[1]) + 3}
                >
                  {s.label}
                </text>
              </g>
            ) : null,
          )}
          {hover !== null && (
            <g>
              <line className="lchart-cross" x1={sx(hover)} x2={sx(hover)} y1={M.t} y2={H - M.b} />
              {read.map((s) =>
                s.value === undefined ? null : (
                  <circle
                    key={s.key}
                    className={`lchart-dot lchart-line--${s.key}`}
                    cx={sx(hover)}
                    cy={sy(s.value)}
                    r={4}
                  />
                ),
              )}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="lchart-tip" style={{ left: `${(sx(hover) / W) * 100}%` }}>
            <b>epoch {hover}</b>
            {read.map((s) => (
              <span key={s.key}>
                <i className={`lchart-swatch lchart-swatch--${s.key}`} />
                {s.label} {s.value === undefined ? '—' : s.value.toFixed(3)}
              </span>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}
