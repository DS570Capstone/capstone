import { useParams, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { Triangle, ArrowLeft, Loader, AlertCircle, AlertTriangle, ExternalLink, Play, History } from 'lucide-react'
import { useAnalysis } from '../hooks/useAnalysis'
import QualityGauge from '../components/QualityGauge'
import FaultFlags from '../components/FaultFlags'
import TrajectoryChart from '../components/TrajectoryChart'
import CoachFeedback from '../components/CoachFeedback'
import StatsRow from '../components/StatsRow'

const STAGE_STEPS = [
  'Queued',
  'Loading video',
  'Estimating pose',
  'Estimating depth',
  'Building trajectories',
  'Computing features',
  'Clustering',
  'Generating feedback',
  'Logging to WandB',
  'Done',
]

export default function Dashboard() {
  const { videoId } = useParams<{ videoId: string }>()
  const navigate = useNavigate()
  const { state, start } = useAnalysis(videoId ?? null)
  const displayName = state.phase === 'done' && state.filename
    ? state.filename.replace(/\.[^.]+$/, '')   // strip extension
    : videoId?.slice(0, 8) + '…'

  useEffect(() => {
    if (videoId && state.phase === 'idle') start()
  }, [videoId])

  return (
    <div className="min-h-screen bg-[#09090b] flex flex-col">
      {/* Nav */}
      <nav className="px-6 py-4 flex items-center gap-3 border-b border-zinc-800/60 sticky top-0 z-10 bg-[#09090b]/90 backdrop-blur">
        <button onClick={() => navigate('/')} className="text-zinc-500 hover:text-white transition-colors">
          <ArrowLeft size={16} />
        </button>
        <div className="w-6 h-6 rounded-md bg-indigo-500 flex items-center justify-center">
          <Triangle size={11} fill="white" stroke="none" />
        </div>
        <span className="text-white font-bold">LiftLens</span>
        <span className="text-zinc-700 text-sm ml-1">/ {displayName}</span>

        <button
          onClick={() => navigate('/history')}
          className="ml-auto flex items-center gap-1.5 text-xs text-zinc-400 hover:text-white transition-colors"
        >
          <History size={14} />
          History
        </button>

        {state.phase === 'done' && state.wandbUrl && (
          <a href={state.wandbUrl} target="_blank" rel="noopener noreferrer"
            className="ml-auto flex items-center gap-1.5 text-xs text-yellow-400 hover:text-yellow-300 transition-colors bg-yellow-400/10 px-3 py-1.5 rounded-lg border border-yellow-400/20">
            <span className="font-bold">W</span>
            WandB Run
            <ExternalLink size={10} />
          </a>
        )}
      </nav>

      <div className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">

        {/* ── Queuing / Polling ── */}
        {(state.phase === 'queuing' || state.phase === 'polling') && (
          <div className="flex flex-col items-center justify-center py-24 gap-6">
            <div className="relative">
              <div className="w-16 h-16 rounded-full border-2 border-indigo-500/30 flex items-center justify-center">
                <Loader size={24} className="text-indigo-400 animate-spin" />
              </div>
            </div>
            <div className="text-center">
              <p className="text-white font-semibold text-lg mb-1">
                {state.phase === 'queuing' ? 'Starting analysis…' : state.status.stage}
              </p>
              <p className="text-zinc-500 text-sm">This takes 20–60s depending on video length</p>
            </div>

            {/* Progress bar */}
            {state.phase === 'polling' && (
              <div className="w-72 flex flex-col gap-2">
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                    style={{ width: `${state.status.progress}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-zinc-600">
                  <span>{state.status.progress}%</span>
                  <span>{state.status.stage}</span>
                </div>
              </div>
            )}

            {/* Stage steps */}
            <div className="flex gap-1.5 flex-wrap justify-center max-w-sm">
              {STAGE_STEPS.map((s, i) => {
                const cur = state.phase === 'polling' ? state.status.stage : ''
                const curIdx = STAGE_STEPS.indexOf(cur)
                const done = i < curIdx
                const active = i === curIdx
                return (
                  <span key={s} className={`text-[10px] px-2 py-0.5 rounded-full transition-colors ${
                    active ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40' :
                    done ? 'bg-zinc-800 text-zinc-500' : 'text-zinc-700'
                  }`}>
                    {s}
                  </span>
                )
              })}
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {state.phase === 'error' && (() => {
          // Validation messages contain specific cues; system errors are generic.
          const isValidation = state.message?.includes('detected') ||
            state.message?.includes('camera') ||
            state.message?.includes('movement') ||
            state.message?.includes('suitable')
          return (
            <div className="flex flex-col items-center justify-center py-24 gap-4 max-w-lg mx-auto text-center">
              {isValidation
                ? <AlertTriangle size={36} className="text-yellow-400" />
                : <AlertCircle size={36} className="text-red-400" />}
              <div>
                <p className={`font-semibold text-lg mb-2 ${isValidation ? 'text-yellow-300' : 'text-red-300'}`}>
                  {isValidation ? 'Video not suitable for analysis' : 'Analysis failed'}
                </p>
                <p className="text-zinc-400 text-sm leading-relaxed">{state.message}</p>
              </div>
              <button onClick={() => navigate('/')} className="mt-2 text-sm text-zinc-400 hover:text-white underline">
                Upload a different video
              </button>
            </div>
          )
        })()}

        {/* ── Results ── */}
        {state.phase === 'done' && (() => {
          const r = state.result
          const wf = r.wave_features
          return (
            <div className="flex flex-col gap-6">

              {/* Header */}
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <h1 className="text-white font-bold text-2xl">{displayName}</h1>
                  <p className="text-zinc-500 text-sm mt-0.5">
                    {r.exercise.toUpperCase()} · {r.camera_position} · {r.duration_sec.toFixed(2)}s
                  </p>
                </div>
                <div className="flex gap-2">
                  <span className="text-xs text-zinc-400 bg-zinc-800 px-3 py-1.5 rounded-lg border border-zinc-700">
                    {r.unsupervised.consensus_cluster_name}
                  </span>
                  {state.wandbUrl && (
                    <a href={state.wandbUrl} target="_blank" rel="noopener noreferrer"
                      className="flex items-center gap-1.5 text-xs text-yellow-400 bg-yellow-400/10 border border-yellow-400/20 px-3 py-1.5 rounded-lg hover:bg-yellow-400/20 transition-colors">
                      <span className="font-bold">W</span> WandB
                    </a>
                  )}
                </div>
              </div>

              {/* Stats row */}
              <StatsRow result={r} />

              {/* Trajectory chart — full width */}
              <TrajectoryChart trajectories={r.trajectories} phases={r.phase_segments} />

              {/* 3-column grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <QualityGauge quality={wf.quality} />
                <FaultFlags flags={r.fault_flags} />

                {/* Energy + Frequency card */}
                <div className="bg-[#171717] rounded-2xl p-6 flex flex-col gap-4">
                  <span className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Energy & Frequency</span>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: 'Work+',    val: wf.energy.work_positive.toFixed(1) },
                      { label: 'Work−',    val: wf.energy.work_negative.toFixed(1) },
                      { label: 'Eff %',   val: wf.energy.efficiency_pct.toFixed(1) + '%' },
                      { label: 'Peak W',  val: wf.energy.peak_power_w.toFixed(1) },
                      { label: 'Dom Hz',  val: wf.frequency.dominant_hz.toFixed(2) },
                      { label: 'Entropy', val: wf.frequency.spectral_entropy.toFixed(3) },
                    ].map(({ label, val }) => (
                      <div key={label} className="bg-zinc-800/50 rounded-lg p-2.5">
                        <span className="text-indigo-400 font-bold text-base block">{val}</span>
                        <span className="text-zinc-600 text-[10px] uppercase tracking-wider">{label}</span>
                      </div>
                    ))}
                  </div>

                  {/* Depth features */}
                  {r.depth_features.depth_enabled && (
                    <div className="border-t border-zinc-800 pt-4 flex flex-col gap-2">
                      <span className="text-zinc-600 text-xs uppercase tracking-wider">Depth (Depth Anything)</span>
                      {[
                        { label: 'Bar asymmetry',    val: r.depth_features.bar_depth_asymmetry.toFixed(4) },
                        { label: 'Torso shift',      val: r.depth_features.torso_depth_shift.toFixed(4) },
                        { label: 'Subj. stability',  val: r.depth_features.subject_depth_stability.toFixed(4) },
                      ].map(({ label, val }) => (
                        <div key={label} className="flex justify-between text-xs">
                          <span className="text-zinc-500">{label}</span>
                          <span className="text-zinc-300 font-mono">{val}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Coach feedback — full width */}
              <CoachFeedback
                language={r.language}
                wandbUrl={state.wandbUrl}
                cluster={r.unsupervised.consensus_cluster_name}
              />

              {/* Phase segments table */}
              {r.phase_segments.length > 0 && (
                <div className="bg-[#171717] rounded-2xl p-6">
                  <span className="text-zinc-400 text-sm font-medium uppercase tracking-wider block mb-4">Phase Segments</span>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-zinc-600 text-xs uppercase tracking-wider">
                          <th className="text-left pb-3 font-medium">Phase</th>
                          <th className="text-right pb-3 font-medium">Start</th>
                          <th className="text-right pb-3 font-medium">End</th>
                          <th className="text-right pb-3 font-medium">Duration</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-800">
                        {r.phase_segments.map((ph, i) => (
                          <tr key={i}>
                            <td className="py-2 capitalize text-zinc-300">{ph.type}</td>
                            <td className="py-2 text-right text-zinc-500 font-mono text-xs">{ph.start_frame}</td>
                            <td className="py-2 text-right text-zinc-500 font-mono text-xs">{ph.end_frame}</td>
                            <td className="py-2 text-right text-zinc-300 font-mono text-xs">{ph.duration_sec.toFixed(3)}s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

            </div>
          )
        })()}
      </div>
    </div>
  )
}
