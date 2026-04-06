import { useState, useEffect, useRef } from 'react'
import { triggerAnalysis, getJobStatus, getResults, JobStatus, AnalysisResult } from '../api/client'

type State =
  | { phase: 'idle' }
  | { phase: 'queuing' }
  | { phase: 'polling'; jobId: string; status: JobStatus }
  | { phase: 'done'; result: AnalysisResult; wandbUrl?: string; filename?: string }
  | { phase: 'error'; message: string }

export function useAnalysis(videoId: string | null) {
  const [state, setState] = useState<State>({ phase: 'idle' })
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const start = async () => {
    if (!videoId) return
    setState({ phase: 'queuing' })
    try {
      const { jobId } = await triggerAnalysis(videoId)
      const initialStatus: JobStatus = { jobId, videoId, status: 'queued', progress: 0, stage: 'Queued' }
      setState({ phase: 'polling', jobId, status: initialStatus })
    } catch (e: any) {
      setState({ phase: 'error', message: e.message })
    }
  }

  // Poll when in polling phase
  useEffect(() => {
    if (state.phase !== 'polling') return
    const { jobId } = state

    const poll = async () => {
      try {
        const status = await getJobStatus(jobId)
        setState(prev => prev.phase === 'polling' ? { ...prev, status } : prev)

        if (status.status === 'done') {
          clearInterval(intervalRef.current!)
          const result = await getResults(status.videoId)
          setState({ phase: 'done', result, wandbUrl: status.wandbUrl, filename: status.filename })
        } else if (status.status === 'invalid') {
          clearInterval(intervalRef.current!)
          setState({ phase: 'error', message: status.error || 'Video is not suitable for analysis.' })
        } else if (status.status === 'error') {
          clearInterval(intervalRef.current!)
          setState({ phase: 'error', message: status.error || 'Analysis failed' })
        }
      } catch (e: any) {
        clearInterval(intervalRef.current!)
        setState({ phase: 'error', message: e.message })
      }
    }

    poll()
    intervalRef.current = setInterval(poll, 2000)
    return () => clearInterval(intervalRef.current!)
  }, [state.phase === 'polling' ? state.jobId : null])

  const reset = () => {
    clearInterval(intervalRef.current!)
    setState({ phase: 'idle' })
  }

  return { state, start, reset }
}
