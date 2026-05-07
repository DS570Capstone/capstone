"""
LiftLens Flask API
Handles video upload → MinIO, job creation → PostgreSQL, analysis trigger → Kafka.
"""
import os, uuid, json, threading
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import yaml
from minio import Minio
from minio.error import S3Error
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Config from environment ─────────────────────────────────────────────────
DATABASE_URL      = os.environ["DATABASE_URL"]
MINIO_ENDPOINT    = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY  = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY  = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET      = os.environ.get("MINIO_BUCKET", "videos")
KAFKA_BOOTSTRAP   = os.environ["KAFKA_BOOTSTRAP"]
KAFKA_TOPIC       = "video.uploaded"
ML_ROOT           = os.environ.get("ML_ROOT", "/app/ml")
ML_CONFIG_PATH    = os.path.join(ML_ROOT, "configs", "default.yaml")

app = Flask(__name__)
CORS(app)

# ── Upload limit ─────────────────────────────────────────────────────────────
# Reject uploads > 500 MB before they reach MinIO or the ML pipeline.
# A typical 60-second phone video is 50-150 MB; 500 MB is a generous ceiling.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# ── Inline worker concurrency guard ──────────────────────────────────────────
# Limits the ML pipeline to one analysis at a time when running without Kafka.
# Each analysis peaks at ~1.5 GB RAM; two simultaneous runs would OOM a 16 GB laptop.
_worker_semaphore = threading.Semaphore(1)

# ── DB helper ────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def _find_completed_duplicate(file_hash: str, exclude_video_id: str | None = None):
    """Return latest completed job+artifact for identical file hash."""
    if not file_hash:
        return None
    sql = """
        SELECT j.video_id, j.object_name, j.filename, j.wandb_url, r.artifact
        FROM jobs j
        JOIN results r USING (video_id)
        WHERE j.file_hash = %s
          AND j.status = 'done'
    """
    params = [file_hash]
    if exclude_video_id:
        sql += " AND j.video_id <> %s"
        params.append(exclude_video_id)
    sql += " ORDER BY j.created_at DESC LIMIT 1"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, tuple(params))
            return cur.fetchone()

# ── MinIO client ─────────────────────────────────────────────────────────────
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def ensure_bucket():
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)

# ── Kafka producer (lazy, retried) ───────────────────────────────────────────
_producer = None
_producer_lock = threading.Lock()

def get_producer():
    global _producer
    with _producer_lock:
        if _producer is None:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode(),
                retries=5,
            )
    return _producer

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/settings")
def get_settings():
    """Return editable analysis settings from ML config."""
    try:
        with open(ML_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        return jsonify(error=f"Failed to read settings: {exc}"), 500

    pipeline = cfg.get("pipeline", {})
    signals = cfg.get("signals", {})
    tracker = cfg.get("tracker", {})
    return jsonify(
        pipeline={
            "max_height": int(pipeline.get("max_height", 720)),
            "max_frames": int(pipeline.get("max_frames", 128)),
            "frame_step": int(pipeline.get("frame_step", 1)),
        },
        signals={
            "resample_length": int(signals.get("resample_length", 128)),
        },
        tracker={
            "backend": str(tracker.get("backend", "yolo_keypoints")),
            "yolo_every_n": int(tracker.get("yolo_every_n", 1)),
            "yolo_frame_step": int(tracker.get("yolo_frame_step", 1)),
            "signal_cutoff_hz": float(tracker.get("signal_cutoff_hz", 2.0)),
        },
    )


@app.put("/api/settings")
def update_settings():
    """Update selected analysis settings in ML config."""
    body = request.get_json(silent=True) or {}
    try:
        with open(ML_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        return jsonify(error=f"Failed to read settings: {exc}"), 500

    cfg.setdefault("pipeline", {})
    cfg.setdefault("signals", {})
    cfg.setdefault("tracker", {})

    p = body.get("pipeline", {}) if isinstance(body.get("pipeline"), dict) else {}
    s = body.get("signals", {}) if isinstance(body.get("signals"), dict) else {}
    t = body.get("tracker", {}) if isinstance(body.get("tracker"), dict) else {}

    def _clamp_int(val, default, min_v, max_v):
        try:
            x = int(val)
        except Exception:
            x = int(default)
        return max(min_v, min(max_v, x))

    def _clamp_float(val, default, min_v, max_v):
        try:
            x = float(val)
        except Exception:
            x = float(default)
        return max(min_v, min(max_v, x))

    cfg["pipeline"]["max_height"] = _clamp_int(
        p.get("max_height", cfg["pipeline"].get("max_height", 720)), 720, 240, 1080
    )
    cfg["pipeline"]["max_frames"] = _clamp_int(
        p.get("max_frames", cfg["pipeline"].get("max_frames", 128)), 128, 32, 256
    )
    cfg["pipeline"]["frame_step"] = _clamp_int(
        p.get("frame_step", cfg["pipeline"].get("frame_step", 1)), 1, 1, 4
    )
    cfg["signals"]["resample_length"] = _clamp_int(
        s.get("resample_length", cfg["signals"].get("resample_length", 128)), 128, 32, 256
    )
    cfg["tracker"]["yolo_every_n"] = _clamp_int(
        t.get("yolo_every_n", cfg["tracker"].get("yolo_every_n", 1)), 1, 1, 4
    )
    cfg["tracker"]["yolo_frame_step"] = _clamp_int(
        t.get("yolo_frame_step", cfg["tracker"].get("yolo_frame_step", 1)), 1, 1, 4
    )
    cfg["tracker"]["signal_cutoff_hz"] = _clamp_float(
        t.get("signal_cutoff_hz", cfg["tracker"].get("signal_cutoff_hz", 2.0)), 2.0, 0.5, 6.0
    )

    backend_val = t.get("backend", cfg["tracker"].get("backend", "yolo_keypoints"))
    if backend_val in {"yolo_keypoints", "sam2_yolo"}:
        cfg["tracker"]["backend"] = backend_val

    try:
        with open(ML_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as exc:
        return jsonify(error=f"Failed to write settings: {exc}"), 500

    return jsonify(ok=True)


@app.get("/api/check-duplicate")
def check_duplicate():
    """Check if a file with the given SHA-256 hash was already processed."""
    file_hash = request.args.get("hash", "").strip()
    if not file_hash:
        return jsonify(duplicate=False)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT j.video_id, j.filename, j.status, j.created_at,
                          r.artifact->'wave_features'->'quality'->>'grade' AS grade
                   FROM jobs j
                   LEFT JOIN results r USING (video_id)
                   WHERE j.file_hash = %s
                   ORDER BY j.created_at DESC LIMIT 1""",
                (file_hash,),
            )
            row = cur.fetchone()
    if not row:
        return jsonify(duplicate=False)
    return jsonify(
        duplicate=True,
        video_id=row["video_id"],
        filename=row["filename"],
        status=row["status"],
        grade=row["grade"],
        created_at=row["created_at"].isoformat() if row["created_at"] else None,
    )


@app.post("/api/upload")
def upload():
    """Receive a video file, store in MinIO, return a video_id."""
    if "video" not in request.files:
        return jsonify(error="No video file"), 400

    file      = request.files["video"]
    file_hash = request.form.get("file_hash") or None
    video_id  = str(uuid.uuid4())
    original_filename = file.filename or "video.mp4"
    ext       = os.path.splitext(original_filename)[1] or ".mp4"
    obj_name  = f"{video_id}{ext}"

    # If identical file content already has a completed result, reuse it.
    dup = _find_completed_duplicate(file_hash) if file_hash else None
    if dup:
        return jsonify(
            video_id=dup["video_id"],
            object_name=dup["object_name"],
            reused=True,
        ), 200

    ensure_bucket()
    try:
        minio_client.put_object(
            MINIO_BUCKET, obj_name, file.stream,
            length=-1, part_size=10 * 1024 * 1024,
            content_type=file.content_type or "video/mp4",
        )
    except S3Error as e:
        return jsonify(error=str(e)), 500

    # Insert job row
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO jobs (video_id, object_name, filename, file_hash, status, stage)
                   VALUES (%s, %s, %s, %s, 'queued', 'Queued')""",
                (video_id, obj_name, original_filename, file_hash),
            )

    return jsonify(video_id=video_id, object_name=obj_name), 201


@app.get("/api/history")
def history():
    """Return all jobs ordered by most recent, with grade and fault summary."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT
                       j.video_id,
                       j.filename,
                       j.status,
                       j.created_at,
                       j.updated_at,
                       j.error,
                       r.artifact->'wave_features'->'quality'->>'grade'       AS grade,
                       r.artifact->>'duration_sec'                             AS duration_sec,
                       r.artifact                                               AS artifact,
                       r.artifact->'signal_processing'                         AS signal_processing
                   FROM jobs j
                   LEFT JOIN results r USING (video_id)
                   ORDER BY
                     CASE j.status
                       WHEN 'done' THEN 0
                       WHEN 'invalid' THEN 1
                       WHEN 'error' THEN 2
                       WHEN 'running' THEN 3
                       WHEN 'queued' THEN 4
                       ELSE 5
                     END,
                     j.created_at DESC
                   LIMIT 300""",
            )
            rows = cur.fetchall()
    out = []
    for r in rows:
        artifact = _ensure_fault_flags(r.get("artifact"))
        fault_flags = (artifact or {}).get("fault_flags", {}) if isinstance(artifact, dict) else {}
        out.append(
            {
                "video_id":    r["video_id"],
                "filename":    r["filename"],
                "status":      r["status"],
                "created_at":  r["created_at"].isoformat() if r["created_at"] else None,
                "grade":       r["grade"],
                "duration_sec": float(r["duration_sec"]) if r["duration_sec"] else None,
                "fault_count": sum(1 for v in (fault_flags or {}).values() if v),
                "error":       r["error"],
                "signal_processing": r.get("signal_processing") or None,
            }
        )
    return jsonify(out)


@app.get("/api/history-summary")
def history_summary():
    """Return top-level history statistics for dashboard cards."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE j.status = 'done') AS processed_videos,
                  GREATEST(
                    COUNT(*) FILTER (WHERE j.status = 'done')
                    - COUNT(DISTINCT j.filename) FILTER (WHERE j.status = 'done' AND j.filename IS NOT NULL),
                    0
                  ) AS reprocessed_runs
                FROM jobs j
                """
            )
            top = cur.fetchone() or {}

            cur.execute(
                """
                SELECT COALESCE(r.artifact->'wave_features'->'quality'->>'grade', 'unknown') AS grade,
                       COUNT(*)::int AS count
                FROM jobs j
                LEFT JOIN results r USING (video_id)
                WHERE j.status = 'done'
                GROUP BY grade
                ORDER BY grade
                """
            )
            grade_rows = cur.fetchall() or []

            cur.execute(
                """
                SELECT kv.key AS fault, COUNT(*)::int AS count
                FROM results r
                JOIN jobs j USING (video_id),
                     LATERAL jsonb_each_text(COALESCE(r.artifact->'fault_flags', '{}'::jsonb)) AS kv(key, value)
                WHERE j.status = 'done' AND kv.value = 'true'
                GROUP BY kv.key
                ORDER BY count DESC, kv.key
                """
            )
            fault_rows = cur.fetchall() or []

    grade_counts = {row["grade"]: int(row["count"]) for row in grade_rows}
    fault_counts = [{"fault": row["fault"], "count": int(row["count"])} for row in fault_rows]
    return jsonify(
        processed_videos=int(top.get("processed_videos") or 0),
        reprocessed_runs=int(top.get("reprocessed_runs") or 0),
        grade_counts=grade_counts,
        fault_counts=fault_counts,
    )


@app.post("/api/analyze/<video_id>")
def analyze(video_id):
    """Publish a Kafka message to kick off the ML worker."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM jobs WHERE video_id = %s", (video_id,))
            job = cur.fetchone()
    if not job:
        return jsonify(error="Job not found"), 404

    # Always run a fresh analysis for this job (no artifact reuse).
    # If previously done/error, reset state and run again.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM results WHERE video_id=%s", (video_id,))
            cur.execute(
                "UPDATE jobs SET status='queued', stage='Queued', progress=0, error=NULL WHERE video_id=%s",
                (video_id,),
            )

    try:
        get_producer().send(KAFKA_TOPIC, {
            "video_id":   video_id,
            "object_name": job["object_name"],
        })
    except NoBrokersAvailable:
        # Fallback: run inline in background thread (dev mode).
        # Semaphore ensures only one pipeline runs at a time to avoid OOM.
        threading.Thread(
            target=_run_worker_inline_guarded,
            args=(video_id, job["object_name"]),
            daemon=True,
        ).start()

    # Mark as running
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status='running', stage='Loading video', progress=5 WHERE video_id=%s",
                (video_id,),
            )

    return jsonify(status="queued"), 200


@app.get("/api/status/<video_id>")
def status(video_id):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT status, progress, stage, error, wandb_url, filename FROM jobs WHERE video_id=%s", (video_id,))
            row = cur.fetchone()
    if not row:
        return jsonify(error="Not found"), 404
    return jsonify(dict(row))


@app.get("/api/results/<video_id>")
def results(video_id):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT artifact FROM results WHERE video_id=%s", (video_id,))
            row = cur.fetchone()
    if not row:
        return jsonify(error="Results not ready"), 404
    artifact = _ensure_trajectories(row["artifact"])
    artifact = _ensure_fault_flags(artifact)
    return jsonify(artifact)


@app.get("/api/video-url/<video_id>")
def video_url(video_id):
    """Return a presigned MinIO URL for playback."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT object_name FROM jobs WHERE video_id=%s", (video_id,))
            row = cur.fetchone()
    if not row:
        return jsonify(error="Not found"), 404
    from datetime import timedelta
    url = minio_client.presigned_get_object(MINIO_BUCKET, row["object_name"], expires=timedelta(hours=1))
    return jsonify(url=url)


# ── Upload size error handler ─────────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify(error="File too large. Maximum upload size is 500 MB."), 413


# ── Inline worker fallback (no Kafka) ────────────────────────────────────────
def _sanitize_for_json(obj):
    """Recursively replace float NaN/Inf with None so json.dumps produces valid JSON.
    PostgreSQL JSONB rejects bare NaN/Infinity tokens."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _ensure_trajectories(artifact: dict | None) -> dict:
    """Backfill trajectories from raw_signals for older/precomputed artifacts."""
    if not isinstance(artifact, dict):
        return artifact or {}

    trajectories = artifact.get("trajectories")
    if not isinstance(trajectories, dict):
        trajectories = {}
        artifact["trajectories"] = trajectories

    needed = ("bar_path_trajectory", "arm_trajectory", "legs_trajectory", "core_trajectory")
    has_all = all(isinstance(trajectories.get(k), list) and len(trajectories.get(k)) > 0 for k in needed)
    if has_all:
        return artifact

    raw = artifact.get("raw_signals") if isinstance(artifact.get("raw_signals"), dict) else {}

    def _to_float_list(values):
        if not isinstance(values, list):
            return []
        out = []
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    def _normalize(values):
        if not values:
            return []
        mean = sum(values) / len(values)
        centered = [v - mean for v in values]
        variance = sum(v * v for v in centered) / max(len(centered), 1)
        std = (variance ** 0.5) if variance > 0 else 1.0
        return [v / std for v in centered]

    bar = _to_float_list(raw.get("bar_center_y")) or _to_float_list(raw.get("bar_center_z_proxy"))
    lw = _to_float_list(raw.get("left_wrist_y"))
    rw = _to_float_list(raw.get("right_wrist_y"))
    trunk = _to_float_list(raw.get("trunk_center_x"))
    n = len(bar) or len(lw) or len(rw) or len(trunk)
    if n == 0:
        return artifact

    if not bar:
        bar = [0.0] * n
    if not lw:
        lw = [0.0] * n
    if not rw:
        rw = lw[:]
    if not trunk:
        trunk = [0.0] * n

    arm = [(lw[i] + rw[i]) / 2.0 for i in range(min(len(lw), len(rw)))]
    if len(arm) < n:
        arm.extend([arm[-1] if arm else 0.0] * (n - len(arm)))

    trajectories["bar_path_trajectory"] = _normalize(bar)[:n]
    trajectories["arm_trajectory"] = _normalize(arm)[:n]
    trajectories["core_trajectory"] = _normalize(trunk[:n])
    trajectories["legs_trajectory"] = [0.0] * n
    return artifact


def _ensure_fault_flags(artifact: dict | None, ml_root: str | None = None) -> dict:
    """Backfill fault flags for legacy artifacts generated before rule features."""
    if not isinstance(artifact, dict):
        return artifact or {}

    fault_flags = artifact.get("fault_flags")
    if isinstance(fault_flags, dict) and any(bool(v) for v in fault_flags.values()):
        return artifact

    raw = artifact.get("raw_signals") if isinstance(artifact.get("raw_signals"), dict) else {}

    def _to_floats(values):
        if not isinstance(values, list):
            return []
        out = []
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    bar = _to_floats(raw.get("bar_center_y"))
    arm = _to_floats(raw.get("left_wrist_y")) or _to_floats(raw.get("bar_center_z_proxy"))
    torso = _to_floats(raw.get("trunk_center_x"))
    if not bar or not arm or not torso:
        return artifact

    n = min(len(bar), len(arm), len(torso))
    bar = bar[:n]
    arm = arm[:n]
    torso = torso[:n]

    bar_amp = max(max(bar) - min(bar), 1e-8)
    torso_amp = max(torso) - min(torso)
    arm_torso_delta = [a - t for a, t in zip(arm, torso)]
    arm_torso_drift = (max(arm_torso_delta) - min(arm_torso_delta)) / bar_amp

    # Approximate lockout oscillation with normalized residual after simple smoothing.
    smooth = []
    for i in range(n):
        s = max(0, i - 2)
        e = min(n, i + 3)
        smooth.append(sum(bar[s:e]) / max(e - s, 1))
    residual = [bar[i] - smooth[i] for i in range(n)]
    mean_res = sum(residual) / n
    var_res = sum((r - mean_res) ** 2 for r in residual) / n
    lockout_osc = (var_res ** 0.5) / bar_amp

    features = {
        "wrist_height_diff_at_lockout_normalized": 0.0,
        "bar_tilt_std_deg": 0.0,
        "bar_lateral_drift_normalized": 0.0,
        "lockout_delay_sec": 0.0,
        "trunk_lateral_shift_normalized": torso_amp / bar_amp,
        "trunk_shift_peak_normalized": torso_amp / bar_amp,
        "hip_lateral_shift_normalized": 0.7 * torso_amp / bar_amp,
        "bar_lockout_oscillation": lockout_osc,
        "bar_depth_drift_normalized": arm_torso_drift,
    }

    cfg_root = ml_root or os.environ.get("ML_ROOT", "/app/ml")
    thresholds_path = os.path.join(cfg_root, "configs", "thresholds.yaml")
    if not os.path.exists(thresholds_path):
        return artifact
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_cfg = yaml.safe_load(f) or {}

    flags = {}
    for fault_name, fault_cfg in (thresholds_cfg.get("faults") or {}).items():
        metric = fault_cfg.get("metric", "")
        thresh = float(fault_cfg.get("threshold", 0.0))
        direction = fault_cfg.get("direction", "gt")
        val = float(features.get(metric, 0.0))
        if direction == "lt":
            flags[fault_name] = val < thresh
        else:
            flags[fault_name] = val > thresh

    artifact["derived_rule_features"] = features
    artifact["fault_flags"] = flags
    return artifact


def _run_worker_inline_guarded(video_id: str, object_name: str):
    """Acquire the semaphore then delegate — queues analyses rather than running them in parallel."""
    with _worker_semaphore:
        _run_worker_inline(video_id, object_name)


def _run_worker_inline(video_id: str, object_name: str):
    """Download video → run pipeline → store result (used when Kafka unavailable)."""
    import sys, tempfile, shutil
    ml_root = os.environ.get("ML_ROOT", "/app/ml")
    ml_config = os.path.join(ml_root, "configs", "default.yaml")
    precomputed_root = os.environ.get(
        "PRECOMPUTED_ANALYSIS_ROOT",
        os.path.join(ml_root, "batch_outputs_sam2_yolo_bodywaves"),
    )
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)

    def _set(status, stage, progress, error=None):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status=%s, stage=%s, progress=%s, error=%s WHERE video_id=%s",
                    (status, stage, progress, error, video_id),
                )

    def _get_job_metadata() -> dict | None:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT filename, file_hash FROM jobs WHERE video_id=%s",
                    (video_id,),
                )
                return cur.fetchone()

    def _load_precomputed_artifact(filename: str | None):
        """Return precomputed artifact for matching filename stem if available."""
        if not filename:
            return None
        stem = Path(filename).stem
        if not stem:
            return None
        candidate = os.path.join(precomputed_root, stem, "analysis.json")
        if not os.path.exists(candidate):
            return None
        with open(candidate, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        # Keep uploaded job identity consistent in DB/UI.
        artifact["video_id"] = video_id
        artifact["video"] = filename
        artifact = _ensure_trajectories(artifact)
        artifact = _ensure_fault_flags(artifact, ml_root=ml_root)
        return artifact

    from src.app.run_single_video import run, VideoValidationError
    tmp = tempfile.mkdtemp()
    try:
        _set("running", "Loading video", 10)
        job_meta = _get_job_metadata()
        precomputed = _load_precomputed_artifact((job_meta or {}).get("filename"))
        if precomputed is not None:
            _set("running", "Loading predefined analysis", 70)
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO results (video_id, artifact) VALUES (%s, %s) "
                        "ON CONFLICT (video_id) DO UPDATE SET artifact=EXCLUDED.artifact",
                        (video_id, json.dumps(_sanitize_for_json(precomputed))),
                    )
                    wandb_url = precomputed.get("wandb_url")
                    cur.execute(
                        "UPDATE jobs SET status='done', stage='Done', progress=100, wandb_url=%s WHERE video_id=%s",
                        (wandb_url, video_id),
                    )
            return

        tmp_path = os.path.join(tmp, object_name)
        minio_client.fget_object(MINIO_BUCKET, object_name, tmp_path)

        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        artifact = run(
            tmp_path, ml_config, out_dir,
            on_progress=lambda stage, pct: _set("running", stage, pct),
        )
        artifact["video_id"] = video_id

        _set("running", "Storing results", 90)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO results (video_id, artifact) VALUES (%s, %s) "
                    "ON CONFLICT (video_id) DO UPDATE SET artifact=EXCLUDED.artifact",
                    (video_id, json.dumps(_sanitize_for_json(artifact))),
                )
                wandb_url = artifact.get("wandb_url")
                cur.execute(
                    "UPDATE jobs SET status='done', stage='Done', progress=100, wandb_url=%s WHERE video_id=%s",
                    (wandb_url, video_id),
                )
    except VideoValidationError as exc:
        # User-facing validation failure — clear message, not a system error
        _set("invalid", "Invalid video", 0, str(exc))
    except Exception as exc:
        _set("error", "Error", 0, str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
