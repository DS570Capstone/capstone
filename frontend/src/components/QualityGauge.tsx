import type { WaveFeatures } from '../api/client'

const GRADE_COLOR: Record<string, string> = {
  A: '#22c55e', B: '#84cc16', C: '#eab308', D: '#f97316', F: '#ef4444',
}

export default function QualityGauge({ quality }: { quality: WaveFeatures['quality'] }) {
  const color = GRADE_COLOR[quality.grade] ?? '#6366f1'
  const pct = Math.round(quality.overall * 100)

  const metrics = [
    { label: 'Smoothness',   val: quality.smoothness },
    { label: 'Control',      val: quality.control },
    { label: 'Efficiency',   val: quality.efficiency },
    { label: 'Consistency',  val: quality.consistency },
    { label: 'Symmetry',     val: quality.symmetry },
  ]

  return (
    <div className="bg-[#171717] rounded-2xl p-6 flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <span className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Quality</span>
        <span className="text-xs text-zinc-600">{pct}%</span>
      </div>

      {/* Big grade circle */}
      <div className="flex items-center gap-5">
        <div
          className="w-16 h-16 rounded-full flex items-center justify-center text-3xl font-bold shrink-0"
          style={{ background: `${color}22`, border: `2px solid ${color}`, color }}
        >
          {quality.grade}
        </div>
        {/* Overall bar */}
        <div className="flex-1">
          <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${pct}%`, background: color }}
            />
          </div>
          <span className="text-xs text-zinc-600 mt-1 block">Overall score</span>
        </div>
      </div>

      {/* Sub-metrics */}
      <div className="grid grid-cols-1 gap-2">
        {metrics.map(({ label, val }) => (
          <div key={label} className="flex items-center gap-3">
            <span className="text-xs text-zinc-500 w-24 shrink-0">{label}</span>
            <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.round(val * 100)}%`, background: color }}
              />
            </div>
            <span className="text-xs text-zinc-400 w-8 text-right">{Math.round(val * 100)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
