# LiftLens — Overhead Press Form Analysis

LiftLens is a full-stack web application that analyses Overhead Press (OHP) technique from a back-view video. Upload a short clip, and the pipeline extracts pose keypoints, tracks the barbell, segments rep phases, detects biomechanical faults, and delivers rule-based coaching cues.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Fault Detection](#fault-detection)
- [Video Requirements](#video-requirements)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## How It Works

1. Upload a back-view OHP video (MP4/MOV/AVI, max 90 s, max 500 MB)
2. The Flask API stores the video in MinIO and creates a job in PostgreSQL
3. The ML pipeline runs in a background thread (no Kafka needed for local dev):
   - Extracts 33-point pose skeleton via MediaPipe
   - Infers barbell position from wrist heuristic
   - Segments the rep into concentric / lockout / eccentric / rest phases
   - Computes 50+ biomechanical features (bilateral symmetry, trunk shift, bar path, etc.)
   - Detects faults against tuned thresholds
   - Generates rule-based coaching cues
4. Results appear on the dashboard with trajectory charts, quality grade, and fault flags

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  React Frontend │────▶│   Flask API     │────▶│  ML Pipeline     │
│  (Vite + TS)    │     │  (Python 3.11)  │     │  (Python 3.11)   │
│  port 3000      │     │  port 5050      │     │  inline thread   │
└─────────────────┘     └────────┬────────┘     └──────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │PostgreSQL│ │  MinIO   │ │  Volume  │
             │ jobs /   │ │ (videos) │ │ outputs/ │
             │ results  │ │          │ │          │
             └──────────┘ └──────────┘ └──────────┘
```

All services run in Docker. The API falls back to an inline background thread when Kafka is unavailable (default for local development).

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine + Compose v2
- 8 GB RAM minimum, 16 GB recommended
- 4 GB free disk space (Docker images + model weights)

No GPU required — the pipeline runs on CPU only.

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd Capstone

# 2. Copy environment file
cp .env.example .env

# 3. Build and start all services
docker compose up --build

# 4. Open the app
#    Frontend:  http://localhost:3000
#    API:       http://localhost:5050/api/health
#    MinIO UI:  http://localhost:9001  (user: minioadmin / pass: minioadmin)
```

First startup takes 3–5 minutes while Docker pulls images and installs Python packages.

---

## Configuration

All ML pipeline settings live in `ml/ohp_form_pipeline/configs/default.yaml`.

### Key settings for CPU / 16 GB RAM

```yaml
pipeline:
  max_height: 480      # resize frames to 480px tall — reduces RAM ~55%
  max_frames: 64       # resample trajectories to 64 points
  frame_step: 2        # process every 2nd frame — halves pose estimation time

depth:
  enabled: false       # keep false — Depth Anything V2 needs GPU

vlm:
  enabled: false       # keep false — Qwen2.5 VLM needs GPU
  device: "cpu"        # safe fallback if accidentally enabled

clustering:
  n_jobs: 2            # limit CPU cores — prevents thermal throttle on laptop
```

### Re-enabling optional features (GPU only)

```bash
# Install extra dependencies first
pip install -r backend/requirements-extras.txt
pip install -r ml/ohp_form_pipeline/requirements-train.txt
```

Then set `depth.enabled: true` or `vlm.enabled: true` in `default.yaml`.

---

## Pipeline Stages

| Stage | Description | Key output |
|-------|-------------|------------|
| **1 — Ingest** | Load video, downsample frames | `VideoMeta`, frame list |
| **2 — Pose + Bar** | MediaPipe 33-point skeleton, wrist-heuristic bar position | `poses`, `bars` |
| **3 — Depth** | Depth Anything V2 (disabled by default) | depth maps |
| **4 — Trajectories** | Resample joint paths to fixed length, compute raw signals | `trajectories`, `raw_signals` |
| **5 — Segmentation** | Detect concentric / lockout / eccentric / rest phases from bar velocity | `phase_segments` |
| **6 — Features** | 50+ biomechanical metrics (bilateral symmetry, trunk shift, bar path, wave analysis) | `fault_flags`, `wave_features` |
| **7 — Feedback** | Rule-based coaching cues matched against fault flags | `language` |
| **8 — Outputs** | Annotated video, signal plots, JSON artifact, text report | saved to `outputs/` |

---

## Fault Detection

The pipeline detects 9 biomechanical faults defined in `configs/thresholds.yaml`:

| Fault | Metric | Threshold |
|-------|--------|-----------|
| Left/right lockout asymmetry | Wrist height diff at lockout | > 5% shoulder width |
| Bar tilt instability | Bar tilt std dev | > 3° |
| Lateral bar drift | Bar lateral drift | > 8% shoulder width |
| Uneven press timing | Lockout delay between sides | > 100 ms |
| Compensatory lateral shift | Trunk lateral shift | > 6% shoulder width |
| Trunk shift under load | Peak trunk shift | > 8% shoulder width |
| Hip shift compensation | Hip lateral shift | > 7% shoulder width |
| Unstable lockout | Bar oscillation at lockout | > 4% |
| Forward bar drift (depth) | Depth proxy drift | > 10% (GPU only) |

Quality is graded A–F from an overall score combining smoothness, control, efficiency, consistency, and bilateral symmetry.

---

## Video Requirements

| Requirement | Detail |
|-------------|--------|
| **Camera angle** | Back-view — camera positioned behind the lifter |
| **Exercise** | Overhead press (standing or seated) |
| **Duration** | 5–90 seconds (one set) |
| **File size** | Max 500 MB |
| **Format** | MP4, MOV, AVI |
| **Resolution** | Any — frames are downscaled to 480px tall automatically |
| **Lighting** | Good lighting, full body visible in frame |

Videos that fail these requirements are rejected with a specific error message before analysis runs.

---

## Project Structure

```
Capstone/
├── frontend/                    # React 18 + TypeScript (Vite)
│   └── src/
│       ├── pages/               # Dashboard.tsx, Home.tsx
│       ├── components/          # UploadCard, QualityGauge, FaultFlags, etc.
│       ├── hooks/useAnalysis.ts # Polling state machine
│       └── api/client.ts        # Typed API client
│
├── backend/
│   ├── api/server.py            # Flask endpoints + inline worker
│   ├── worker/                  # Kafka consumer (production)
│   ├── migrations/init.sql      # PostgreSQL schema
│   ├── requirements.txt         # Inference-only deps (~380 MB)
│   ├── requirements-extras.txt  # torch/transformers (GPU features)
│   └── Dockerfile
│
├── ml/ohp_form_pipeline/
│   ├── configs/
│   │   ├── default.yaml         # All pipeline settings
│   │   ├── thresholds.yaml      # Fault detection thresholds
│   │   └── rules_ohp.yaml       # Coaching cue library (9 rules)
│   ├── models/
│   │   └── pose_landmarker_heavy.task   # MediaPipe model (30 MB)
│   ├── checkpoints/
│   │   ├── vision_adapter.pt    # Fine-tuned vision adapter (83 MB)
│   │   └── language_lora/       # LoRA weights for Qwen2.5 (34 MB)
│   ├── src/
│   │   ├── app/run_single_video.py   # Main pipeline orchestrator
│   │   ├── cv/                  # Pose estimation, bar detection, depth
│   │   ├── signals/             # Segmentation, feature engineering, wave analysis
│   │   ├── reasoning/           # Rule engine, feedback generator
│   │   ├── viz/                 # Annotated video, signal plots
│   │   └── io/                  # Video loader, JSON writer
│   ├── requirements.txt         # Inference-only ML deps
│   └── requirements-train.txt   # Training / batch deps (wandb, peft, etc.)
│
└── docker-compose.yml
```

---

## Development

### Rebuild after changing Python dependencies

```bash
docker compose build api
docker compose up -d api
```

### Reload Python code without rebuilding

The `ml/` directory is bind-mounted into the container. Any change to pipeline Python files takes effect after restarting the API container:

```bash
docker compose restart api
```

### View logs

```bash
docker compose logs -f api        # ML pipeline output + errors
docker compose logs -f frontend   # Vite dev server
```

### Run the pipeline directly (without Docker)

```bash
cd ml/ohp_form_pipeline
pip install -r requirements.txt
python run_single.py --video path/to/clip.mp4 --config configs/default.yaml
```

### Reset all data

```bash
docker compose down -v   # removes all volumes including the database
docker compose up -d
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'yaml'` | `pyyaml` missing from backend deps | Already fixed — rebuild image |
| `Read-only file system: '/app/ml/data'` | ML volume was mounted `:ro` | Already fixed in `docker-compose.yml` |
| `Image size ... is too large` | Corrupted HEVC frame from phone video | Already fixed — frames are validated before MediaPipe |
| `Token "NaN" is invalid` | NumPy NaN in artifact JSON | Already fixed — sanitized before DB insert |
| `No person detected` | Full body not in frame, or wrong exercise | Ensure back-view, full body visible |
| `No overhead press movement` | Static video or wrong exercise | Upload a video of an actual OHP set |
| `Front-view video` | Camera facing the lifter | Move camera behind the lifter |
| Container OOM killed | Video too long / large | Max 90s, max 500 MB |
| Slow analysis (3–5 min) | CPU-only mode, normal | Expected for 30–60s video on laptop CPU |
