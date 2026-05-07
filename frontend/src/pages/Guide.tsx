import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Triangle } from 'lucide-react'

export default function Guide() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[#09090b] flex flex-col">
      <nav className="px-6 py-4 flex items-center gap-3 border-b border-zinc-800/60 sticky top-0 z-10 bg-[#09090b]/90 backdrop-blur">
        <button onClick={() => navigate('/')} className="text-zinc-500 hover:text-white transition-colors">
          <ArrowLeft size={16} />
        </button>
        <div className="w-6 h-6 rounded-md bg-indigo-500 flex items-center justify-center">
          <Triangle size={11} fill="white" stroke="none" />
        </div>
        <span className="text-white font-bold">LiftLens</span>
        <span className="text-zinc-700 text-sm ml-1">/ Guide</span>
      </nav>

      <div className="flex-1 max-w-5xl mx-auto w-full px-4 py-8 flex flex-col gap-6">
        <div>
          <h1 className="text-white font-bold text-2xl">User Guide</h1>
          <p className="text-zinc-400 text-sm mt-1">
            How scoring works, what each fault means, and which parameters most affect results.
          </p>
        </div>

        <section className="bg-[#171717] rounded-2xl p-6 border border-zinc-800/60">
          <h2 className="text-white font-semibold mb-3">Scoring (A-F)</h2>
          <p className="text-zinc-400 text-sm leading-relaxed">
            Overall score is computed from four components: smoothness, control, efficiency, and consistency.
            Current grade boundaries are: A &gt;= 0.80, B &gt;= 0.70, C &gt;= 0.55, D &gt;= 0.40, else F.
          </p>
        </section>

        <section className="bg-[#171717] rounded-2xl p-6 border border-zinc-800/60">
          <h2 className="text-white font-semibold mb-3">Fault Flags</h2>
          <p className="text-zinc-400 text-sm leading-relaxed">
            Faults are threshold-based. A flag means its metric crossed the configured threshold.
            No-fault does not automatically mean A-grade; quality and faults are separate systems.
          </p>
          <p className="text-zinc-500 text-xs mt-3">
            Main faults: lockout asymmetry, bar tilt instability, lateral bar drift, uneven press timing,
            compensatory lateral shift, trunk shift under load, hip shift compensation, unstable lockout,
            forward bar drift depth proxy.
          </p>
        </section>

        <section className="bg-[#171717] rounded-2xl p-6 border border-zinc-800/60">
          <h2 className="text-white font-semibold mb-3">Parameters That Matter Most</h2>
          <div className="text-zinc-400 text-sm leading-relaxed flex flex-col gap-2">
            <p><span className="text-zinc-200">Signal Cutoff Hz:</span> most impactful for smoothness/efficiency balance.</p>
            <p><span className="text-zinc-200">Max Height / Max Frames / Resample Length:</span> controls detail level.</p>
            <p><span className="text-zinc-200">YOLO Every N / YOLO Frame Step:</span> keep at 1 for best fidelity.</p>
            <p><span className="text-zinc-200">Backend:</span> yolo_keypoints is fastest; sam2_yolo can be higher quality but slower.</p>
          </div>
        </section>

        <section className="bg-[#171717] rounded-2xl p-6 border border-zinc-800/60">
          <h2 className="text-white font-semibold mb-3">Best Practices For Better Results</h2>
          <ul className="text-zinc-400 text-sm leading-relaxed list-disc pl-5 space-y-1">
            <li>Use back-view videos with full body in frame.</li>
            <li>Keep lighting consistent and avoid strong shadows.</li>
            <li>Upload trimmed clips (single set) for stable phase segmentation.</li>
            <li>Compare runs using the same settings when evaluating improvements.</li>
          </ul>
        </section>
      </div>
    </div>
  )
}

