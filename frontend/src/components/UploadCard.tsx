import { useState, useRef, DragEvent } from 'react'
import { Upload, Loader, CheckCircle, Triangle, History, AlertTriangle } from 'lucide-react'
import { uploadVideo, checkDuplicate, triggerAnalysis } from '../api/client'
import { useNavigate } from 'react-router-dom'

type Phase = 'idle' | 'hashing' | 'uploading' | 'done' | 'error'

async function sha256(file: File): Promise<string> {
  const buf = await file.arrayBuffer()
  const digest = await crypto.subtle.digest('SHA-256', buf)
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('')
}

export default function UploadCard() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [filename, setFilename] = useState('')
  const [dragging, setDragging] = useState(false)
  const [duplicate, setDuplicate] = useState<{ videoId: string; grade: string | null; filename: string; createdAt: string } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const MAX_FILE_MB = 500
  const MAX_DURATION_SEC = 90

  const handleFile = async (file: File) => {
    if (!file.type.startsWith('video/')) {
      setError('Please select a video file.')
      setPhase('error')
      return
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`File is too large (${(file.size / 1024 / 1024).toFixed(0)} MB). Maximum is ${MAX_FILE_MB} MB.`)
      setPhase('error')
      return
    }
    // Duration check
    const durationOk = await new Promise<boolean>(resolve => {
      const url = URL.createObjectURL(file)
      const vid = document.createElement('video')
      vid.preload = 'metadata'
      vid.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(vid.duration <= MAX_DURATION_SEC) }
      vid.onerror = () => { URL.revokeObjectURL(url); resolve(true) }
      vid.src = url
    })
    if (!durationOk) {
      setError(`Video is too long. Maximum duration is ${MAX_DURATION_SEC}s. Please trim to a single OHP set.`)
      setPhase('error')
      return
    }

    // Hash the file and check for duplicates
    setFilename(file.name)
    setPhase('hashing')
    let fileHash = ''
    try {
      fileHash = await sha256(file)
      const check = await checkDuplicate(fileHash)
      if (check.duplicate && check.status === 'done') {
        setDuplicate({
          videoId: check.video_id,
          grade: check.grade,
          filename: check.filename,
          createdAt: check.created_at,
        })
        setPhase('idle')
        return
      }
    } catch {
      // Hash or check failed — continue with upload anyway
    }

    setPhase('uploading')
    setProgress(0)
    try {
      const ticker = setInterval(() => setProgress(p => Math.min(p + 8, 85)), 300)
      const { videoId } = await uploadVideo(file, fileHash)
      clearInterval(ticker)
      setProgress(100)
      setPhase('done')
      setTimeout(() => navigate(`/dashboard/${videoId}`), 600)
    } catch (e: any) {
      setError(e.message)
      setPhase('error')
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div className="min-h-screen bg-[#09090b] flex flex-col">
      {/* Nav */}
      <nav className="px-6 py-4 flex items-center gap-2.5 border-b border-zinc-800/60">
        <div className="w-7 h-7 rounded-lg bg-indigo-500 flex items-center justify-center">
          <Triangle size={13} fill="white" stroke="none" />
        </div>
        <span className="text-white font-bold">LiftLens</span>
        <button
          onClick={() => navigate('/history')}
          className="ml-auto flex items-center gap-1.5 text-xs text-zinc-400 hover:text-white transition-colors"
        >
          <History size={14} />
          History
        </button>
      </nav>

      <div className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl flex flex-col gap-6">

          {/* Hero text */}
          <div className="text-center">
            <h1 className="text-4xl font-bold text-white mb-2">
              Analyze Your <span className="text-indigo-400">Overhead Press</span>
            </h1>
            <p className="text-zinc-500 text-sm max-w-md mx-auto">
              Upload a back-view OHP video. Our pipeline extracts biomechanical trajectories,
              detects movement faults, and delivers AI coaching.
            </p>
          </div>

          {/* Duplicate banner */}
          {duplicate && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-2xl p-4 flex items-start gap-3">
              <AlertTriangle size={18} className="text-yellow-400 mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-yellow-300 font-semibold text-sm">This video was already analysed</p>
                <p className="text-zinc-400 text-xs mt-0.5 truncate">
                  {duplicate.filename} · {new Date(duplicate.createdAt).toLocaleDateString()}
                  {duplicate.grade && <> · Grade <span className="text-white font-bold">{duplicate.grade}</span></>}
                </p>
              </div>
              <div className="flex gap-2 shrink-0">
                <button
                  onClick={() => navigate(`/dashboard/${duplicate.videoId}`)}
                  className="text-xs bg-indigo-500 hover:bg-indigo-400 text-white px-3 py-1.5 rounded-lg transition-colors"
                >
                  View result
                </button>
                <button
                  onClick={async () => {
                    const id = duplicate.videoId
                    setDuplicate(null)
                    try { await triggerAnalysis(id) } catch { /* ignore — dashboard will poll */ }
                    navigate(`/dashboard/${id}`)
                  }}
                  className="text-xs text-zinc-400 hover:text-white px-3 py-1.5 rounded-lg transition-colors"
                >
                  Re-analyse
                </button>
              </div>
            </div>
          )}

          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => phase === 'idle' && inputRef.current?.click()}
            className={`
              relative rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer
              flex flex-col items-center justify-center gap-4 p-12
              ${dragging ? 'border-indigo-400 bg-indigo-500/10' : 'border-zinc-700 bg-[#171717] hover:border-zinc-500 hover:bg-zinc-800/40'}
              ${phase === 'error' ? 'border-red-500/50 bg-red-500/5' : ''}
              ${phase === 'done' ? 'border-emerald-500/50 bg-emerald-500/5 cursor-default' : ''}
            `}
          >
            <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />

            {phase === 'idle' && (
              <>
                <div className="w-14 h-14 rounded-xl bg-indigo-500/15 flex items-center justify-center">
                  <Upload size={24} className="text-indigo-400" />
                </div>
                <div className="text-center">
                  <p className="text-white font-semibold mb-1">Drop your video here</p>
                  <p className="text-zinc-500 text-sm">MP4, MOV, AVI · back-view OHP · max 90s · max 500 MB</p>
                </div>
                <span className="text-xs text-zinc-600 bg-zinc-800 px-3 py-1.5 rounded-lg">Browse files</span>
              </>
            )}

            {phase === 'hashing' && (
              <>
                <Loader size={32} className="text-zinc-500 animate-spin" />
                <div className="text-center">
                  <p className="text-white font-semibold mb-1">Checking for duplicates…</p>
                  <p className="text-zinc-500 text-sm font-mono truncate max-w-xs">{filename}</p>
                </div>
              </>
            )}

            {phase === 'uploading' && (
              <>
                <Loader size={32} className="text-indigo-400 animate-spin" />
                <div className="text-center">
                  <p className="text-white font-semibold mb-1">Uploading…</p>
                  <p className="text-zinc-500 text-sm font-mono truncate max-w-xs">{filename}</p>
                </div>
                <div className="w-48 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-500 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
                </div>
              </>
            )}

            {phase === 'done' && (
              <>
                <CheckCircle size={32} className="text-emerald-400" />
                <p className="text-white font-semibold">Upload complete — redirecting…</p>
              </>
            )}

            {phase === 'error' && (
              <>
                <div className="w-14 h-14 rounded-xl bg-red-500/15 flex items-center justify-center">
                  <Upload size={24} className="text-red-400" />
                </div>
                <div className="text-center">
                  <p className="text-red-300 font-semibold mb-1">Upload failed</p>
                  <p className="text-zinc-500 text-sm">{error}</p>
                </div>
                <button
                  onClick={e => { e.stopPropagation(); setPhase('idle'); setError('') }}
                  className="text-xs text-zinc-400 hover:text-white underline"
                >Try again</button>
              </>
            )}
          </div>

          {/* Info chips */}
          <div className="flex gap-3 flex-wrap justify-center">
            {['Depth Anything V2', 'HDBSCAN clustering', 'WandB logging', 'AI coaching'].map(t => (
              <span key={t} className="text-xs text-zinc-500 bg-zinc-800/60 border border-zinc-700/50 rounded-full px-3 py-1">{t}</span>
            ))}
          </div>

        </div>
      </div>
    </div>
  )
}
