import { AlertTriangle, CheckCircle } from 'lucide-react'

const FAULT_LABELS: Record<string, string> = {
  left_right_lockout_asymmetry: 'Lockout Asymmetry',
  bar_tilt_instability: 'Bar Tilt',
  lateral_bar_drift: 'Lateral Bar Drift',
  uneven_press_timing: 'Uneven Press Timing',
  compensatory_lateral_shift: 'Lateral Trunk Shift',
  trunk_shift_under_load: 'Trunk Shift Under Load',
  hip_shift_compensation: 'Hip Shift',
  unstable_lockout: 'Unstable Lockout',
  forward_bar_drift_depth_proxy: 'Forward Bar Drift (depth)',
}

export default function FaultFlags({ flags }: { flags: Record<string, boolean> }) {
  const active = Object.entries(flags).filter(([, v]) => v)
  const clean = Object.entries(flags).filter(([, v]) => !v)

  return (
    <div className="bg-[#171717] rounded-2xl p-6 flex flex-col gap-4">
      <span className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Fault Flags</span>

      {active.length === 0 ? (
        <div className="flex items-center gap-2 text-emerald-400">
          <CheckCircle size={16} />
          <span className="text-sm">No faults detected</span>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {active.map(([key]) => (
            <div key={key} className="flex items-center gap-2.5 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              <AlertTriangle size={13} className="text-red-400 shrink-0" />
              <span className="text-sm text-red-300">{FAULT_LABELS[key] ?? key}</span>
            </div>
          ))}
        </div>
      )}

      {clean.length > 0 && active.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {clean.map(([key]) => (
            <span key={key} className="text-[11px] text-zinc-600 bg-zinc-800/60 rounded px-2 py-0.5">
              ✓ {FAULT_LABELS[key] ?? key}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
