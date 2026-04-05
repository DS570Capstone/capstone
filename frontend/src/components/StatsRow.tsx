import type { AnalysisResult } from '../api/client'

export default function StatsRow({ result }: { result: AnalysisResult }) {
  const wf = result.wave_features
  const items = [
    { label: 'FPS',         value: result.fps.toFixed(0) },
    { label: 'Frames',      value: result.n_frames.toString() },
    { label: 'Duration',    value: `${result.duration_sec.toFixed(2)}s` },
    { label: 'Osc. Count',  value: wf.harmonic.oscillation_count.toString() },
    { label: 'Dom. Freq',   value: `${wf.frequency.dominant_hz.toFixed(2)} Hz` },
    { label: 'Efficiency',  value: `${wf.energy.efficiency_pct.toFixed(1)}%` },
    { label: 'Anomaly',     value: result.unsupervised.anomaly_score.toFixed(3) },
    { label: 'Depth',       value: result.depth_features.depth_enabled ? 'On' : 'Off' },
  ]

  return (
    <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
      {items.map(({ label, value }) => (
        <div key={label} className="bg-[#171717] rounded-xl p-3 flex flex-col items-center gap-1">
          <span className="text-indigo-400 font-bold text-lg leading-none">{value}</span>
          <span className="text-zinc-600 text-[10px] uppercase tracking-widest text-center">{label}</span>
        </div>
      ))}
    </div>
  )
}
