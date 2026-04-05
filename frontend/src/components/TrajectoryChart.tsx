import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import type { AnalysisResult } from '../api/client'

const TRAJ_CONFIG = [
  { key: 'bar_path_trajectory', label: 'Bar Path',  color: '#6366f1' },
  { key: 'arm_trajectory',      label: 'Arms',      color: '#22c55e' },
  { key: 'legs_trajectory',     label: 'Legs',      color: '#f59e0b' },
  { key: 'core_trajectory',     label: 'Core',      color: '#a78bfa' },
]

const PHASE_COLORS: Record<string, string> = {
  setup: '#3b82f6', concentric: '#22c55e',
  lockout: '#eab308', eccentric: '#f97316', rest: '#6b7280',
}

type Props = {
  trajectories: AnalysisResult['trajectories']
  phases: AnalysisResult['phase_segments']
}

export default function TrajectoryChart({ trajectories, phases }: Props) {
  const len = trajectories.bar_path_trajectory.length || 128

  const data = Array.from({ length: len }, (_, i) => ({
    frame: i,
    bar: trajectories.bar_path_trajectory[i] ?? 0,
    arm: trajectories.arm_trajectory[i] ?? 0,
    legs: trajectories.legs_trajectory[i] ?? 0,
    core: trajectories.core_trajectory[i] ?? 0,
  }))

  return (
    <div className="bg-[#171717] rounded-2xl p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <span className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Trajectories</span>
        <div className="flex gap-4 flex-wrap">
          {TRAJ_CONFIG.map(({ key, label, color }) => (
            <div key={key} className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 rounded" style={{ background: color }} />
              <span className="text-xs text-zinc-500">{label}</span>
            </div>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <XAxis dataKey="frame" tick={{ fontSize: 10, fill: '#52525b' }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fontSize: 10, fill: '#52525b' }} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#a1a1aa' }}
            itemStyle={{ color: '#e4e4e7' }}
          />
          {phases.map((ph, i) => (
            <ReferenceLine
              key={i}
              x={ph.start_frame}
              stroke={PHASE_COLORS[ph.type] ?? '#52525b'}
              strokeDasharray="3 3"
              strokeOpacity={0.5}
              label={{ value: ph.type.slice(0, 3), position: 'top', fontSize: 9, fill: PHASE_COLORS[ph.type] ?? '#52525b' }}
            />
          ))}
          <Line type="monotone" dataKey="bar"  stroke="#6366f1" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="arm"  stroke="#22c55e" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="legs" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="core" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>

      {/* Phase legend */}
      <div className="flex gap-3 flex-wrap">
        {phases.map((ph, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: PHASE_COLORS[ph.type] ?? '#52525b' }} />
            <span className="text-[11px] text-zinc-500">{ph.type} {ph.duration_sec.toFixed(2)}s</span>
          </div>
        ))}
      </div>
    </div>
  )
}
