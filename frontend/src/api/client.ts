const BASE = '/api'

export async function checkDuplicate(hash: string): Promise<DuplicateCheck> {
  const res = await fetch(`${BASE}/check-duplicate?hash=${encodeURIComponent(hash)}`)
  if (!res.ok) return { duplicate: false }
  return res.json()
}

export async function uploadVideo(file: File, fileHash?: string): Promise<{ videoId: string; objectName: string }> {
  const form = new FormData()
  form.append('video', file)
  if (fileHash) form.append('file_hash', fileHash)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return {
    videoId: data.video_id,
    objectName: data.object_name,
  }
}

export async function getHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${BASE}/history`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function triggerAnalysis(videoId: string): Promise<{ jobId: string }> {
  const res = await fetch(`${BASE}/analyze/${videoId}`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return { jobId: videoId }
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/status/${jobId}`)
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return {
    jobId,
    videoId: jobId,
    status: data.status,
    progress: data.progress,
    stage: data.stage,
    error: data.error,
    wandbUrl: data.wandb_url,
    filename: data.filename ?? undefined,
  }
}

export async function getResults(videoId: string): Promise<AnalysisResult> {
  const res = await fetch(`${BASE}/results/${videoId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getVideoUrl(videoId: string): Promise<string> {
  const res = await fetch(`${BASE}/video-url/${videoId}`)
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return data.url
}

// ── Types ──────────────────────────────────────────────────────

export type JobStatus = {
  jobId: string
  videoId: string
  status: 'queued' | 'running' | 'done' | 'error' | 'invalid'
  progress: number        // 0–100
  stage: string
  error?: string
  wandbUrl?: string
  filename?: string
}

export type AnalysisResult = {
  video_id: string
  exercise: string
  camera_position: string
  fps: number
  n_frames: number
  duration_sec: number
  trajectories: {
    bar_path_trajectory: number[]
    arm_trajectory: number[]
    legs_trajectory: number[]
    core_trajectory: number[]
  }
  raw_signals: Record<string, (number | null)[]>
  phase_segments: PhaseSegment[]
  wave_features: WaveFeatures
  unsupervised: UnsupervisedResult
  fault_flags: Record<string, boolean>
  depth_features: DepthFeatures
  language: Language
  wandb_url?: string
}

export type PhaseSegment = {
  type: string
  start_frame: number
  end_frame: number
  duration_sec: number
}

export type WaveFeatures = {
  quality: {
    grade: string
    overall: number
    smoothness: number
    control: number
    efficiency: number
    consistency: number
    symmetry: number
  }
  energy: {
    work_positive: number
    work_negative: number
    efficiency_pct: number
    peak_power_w: number
  }
  damping: { ratio: number; control_quality: string }
  frequency: {
    dominant_hz: number
    band_power: { slow: number; medium: number; fast: number; harmonic: number }
    spectral_entropy: number
  }
  harmonic: { oscillation_count: number; is_harmonic: boolean }
  waves: { type: string; duration_sec: number; mean_velocity: number; smoothness: number }[]
  wave_count: number
}

export type UnsupervisedResult = {
  feature_cluster_id: number
  feature_cluster_name: string
  latent_cluster_id: number
  latent_cluster_name: string
  consensus_cluster_name: string
  cluster_confidence: number
  anomaly_score: number
  disagreement_score: number
}

export type DepthFeatures = {
  depth_enabled: boolean
  bar_forward_drift_depth: number
  bar_depth_asymmetry: number
  torso_depth_shift: number
  subject_depth_stability: number
}

export type Language = {
  summary: string
  coach_feedback: string
  reasoning_trace_short: string
}

export type HistoryItem = {
  video_id: string
  filename: string | null
  status: 'queued' | 'running' | 'done' | 'error' | 'invalid'
  created_at: string
  grade: string | null
  duration_sec: number | null
  fault_count: number
  error: string | null
}

export type DuplicateCheck =
  | { duplicate: false }
  | { duplicate: true; video_id: string; filename: string; status: string; grade: string | null; created_at: string }
