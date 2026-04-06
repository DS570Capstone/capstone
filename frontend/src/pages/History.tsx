import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Triangle, ArrowLeft, Loader, AlertCircle, ChevronRight, CheckCircle, Clock, XCircle } from 'lucide-react'
import { getHistory, HistoryItem } from '../api/client'

const GRADE_COLOR: Record<string, string> = {
  A: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  B: 'text-indigo-400 bg-indigo-400/10 border-indigo-400/30',
  C: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  D: 'text-orange-400 bg-orange-400/10 border-orange-400/30',
  F: 'text-red-400 bg-red-400/10 border-red-400/30',
}

function StatusBadge({ status }: { status: HistoryItem['status'] }) {
  if (status === 'done') return <CheckCircle size={14} className="text-emerald-400" />
  if (status === 'running' || status === 'queued') return <Clock size={14} className="text-indigo-400" />
  return <XCircle size={14} className="text-red-400" />
}

function formatDate(iso: string) {
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) +
    ' · ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

export default function History() {
  const navigate = useNavigate()
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getHistory()
      .then(setItems)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

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
        <span className="text-zinc-700 text-sm ml-1">/ History</span>
      </nav>

      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-white font-bold text-xl">Analysis History</h1>
            <p className="text-zinc-500 text-sm mt-0.5">{items.length} videos analysed</p>
          </div>
        </div>

        {loading && (
          <div className="flex justify-center py-24">
            <Loader size={24} className="text-indigo-400 animate-spin" />
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-red-400 py-12 justify-center">
            <AlertCircle size={18} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <div className="text-center py-24 text-zinc-600">
            <p className="text-lg font-semibold mb-2">No analyses yet</p>
            <p className="text-sm">Upload your first OHP video to get started.</p>
            <button
              onClick={() => navigate('/')}
              className="mt-6 text-sm text-indigo-400 hover:text-indigo-300 underline"
            >
              Upload a video
            </button>
          </div>
        )}

        {!loading && items.length > 0 && (
          <div className="flex flex-col gap-2">
            {items.map(item => (
              <button
                key={item.video_id}
                onClick={() => item.status === 'done' && navigate(`/dashboard/${item.video_id}`)}
                disabled={item.status !== 'done'}
                className={`w-full text-left bg-[#171717] rounded-2xl p-4 flex items-center gap-4 border border-zinc-800/60 transition-all
                  ${item.status === 'done' ? 'hover:border-zinc-600 hover:bg-zinc-800/40 cursor-pointer' : 'opacity-60 cursor-default'}`}
              >
                {/* Status icon */}
                <div className="shrink-0">
                  <StatusBadge status={item.status} />
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm font-medium truncate">
                    {item.filename
                      ? item.filename.replace(/\.[^.]+$/, '')
                      : item.video_id.slice(0, 8) + '…'}
                  </p>
                  <p className="text-zinc-500 text-xs mt-0.5">
                    {formatDate(item.created_at)}
                    {item.duration_sec != null && <> · {item.duration_sec.toFixed(1)}s</>}
                    {item.status === 'done' && item.fault_count > 0 && (
                      <> · <span className="text-orange-400">{item.fault_count} fault{item.fault_count > 1 ? 's' : ''}</span></>
                    )}
                    {item.status === 'done' && item.fault_count === 0 && (
                      <> · <span className="text-emerald-400">no faults</span></>
                    )}
                    {(item.status === 'error' || item.status === 'invalid') && item.error && (
                      <> · <span className="text-red-400 truncate">{item.error.slice(0, 60)}</span></>
                    )}
                  </p>
                </div>

                {/* Grade */}
                {item.grade && (
                  <span className={`text-sm font-bold px-2.5 py-1 rounded-lg border shrink-0 ${GRADE_COLOR[item.grade] ?? 'text-zinc-400 bg-zinc-800 border-zinc-700'}`}>
                    {item.grade}
                  </span>
                )}

                {item.status === 'done' && (
                  <ChevronRight size={16} className="text-zinc-600 shrink-0" />
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
