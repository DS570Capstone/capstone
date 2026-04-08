"""
LiftLens Flask API
Handles video upload → MinIO, job creation → PostgreSQL, analysis trigger → Kafka.
"""
import os, uuid, json, threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import psycopg2.extras
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
                       r.artifact->'fault_flags'                               AS fault_flags
                   FROM jobs j
                   LEFT JOIN results r USING (video_id)
                   ORDER BY j.created_at DESC
                   LIMIT 100""",
            )
            rows = cur.fetchall()
    return jsonify([
        {
            "video_id":    r["video_id"],
            "filename":    r["filename"],
            "status":      r["status"],
            "created_at":  r["created_at"].isoformat() if r["created_at"] else None,
            "grade":       r["grade"],
            "duration_sec": float(r["duration_sec"]) if r["duration_sec"] else None,
            "fault_count": sum(1 for v in (r["fault_flags"] or {}).values() if v),
            "error":       r["error"],
        }
        for r in rows
    ])


@app.post("/api/analyze/<video_id>")
def analyze(video_id):
    """Publish a Kafka message to kick off the ML worker."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM jobs WHERE video_id = %s", (video_id,))
            job = cur.fetchone()
    if not job:
        return jsonify(error="Job not found"), 404

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
    return jsonify(row["artifact"])


@app.get("/api/video-url/<video_id>")
def video_url(video_id):
    """Return a presigned MinIO URL for the original uploaded video."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT object_name FROM jobs WHERE video_id=%s", (video_id,))
            row = cur.fetchone()
    if not row:
        return jsonify(error="Not found"), 404
    from datetime import timedelta
    url = minio_client.presigned_get_object(MINIO_BUCKET, row["object_name"], expires=timedelta(hours=1))
    return jsonify(url=url)


@app.get("/api/compare-url/<video_id>")
def compare_url(video_id):
    """Return a presigned MinIO URL for the VP3D comparison video."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT artifact->>'compare_object' AS compare_object FROM results WHERE video_id=%s", (video_id,))
            row = cur.fetchone()
    if not row or not row["compare_object"]:
        return jsonify(error="Comparison video not available"), 404
    from datetime import timedelta
    url = minio_client.presigned_get_object(MINIO_BUCKET, row["compare_object"], expires=timedelta(hours=1))
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


def _run_worker_inline_guarded(video_id: str, object_name: str):
    """Acquire the semaphore then delegate — queues analyses rather than running them in parallel."""
    with _worker_semaphore:
        _run_worker_inline(video_id, object_name)


def _run_worker_inline(video_id: str, object_name: str):
    """Download video → run pipeline → store result (used when Kafka unavailable)."""
    import sys, tempfile, shutil
    ml_root = os.environ.get("ML_ROOT", "/app/ml")
    ml_config = os.path.join(ml_root, "configs", "default.yaml")
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)

    def _set(status, stage, progress, error=None):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status=%s, stage=%s, progress=%s, error=%s WHERE video_id=%s",
                    (status, stage, progress, error, video_id),
                )

    tmp = tempfile.mkdtemp()
    try:
        _set("running", "Loading video", 10)
        tmp_path = os.path.join(tmp, object_name)
        minio_client.fget_object(MINIO_BUCKET, object_name, tmp_path)

        _set("running", "Estimating pose", 30)
        from src.app.run_single_video import run, VideoValidationError
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        artifact = run(tmp_path, ml_config, out_dir)
        artifact["video_id"] = video_id

        # Upload annotated video to MinIO if produced
        ann_candidates = [
            os.path.join(out_dir, "videos", f"{artifact.get('video_id', video_id)}_annotated.mp4"),
            os.path.join(out_dir, "annotated.mp4"),
        ]
        for candidate in ann_candidates:
            if os.path.exists(candidate):
                ann_obj = f"annotated/{video_id}.mp4"
                minio_client.fput_object(MINIO_BUCKET, ann_obj, candidate, content_type="video/mp4")
                artifact["annotated_object"] = ann_obj
                break

        # Upload VP3D comparison video to MinIO if produced
        compare_candidate = os.path.join(out_dir, "videos", f"{artifact.get('video_id', video_id)}_compare_vp3d.mp4")
        if os.path.exists(compare_candidate):
            cmp_obj = f"compare/{video_id}.mp4"
            minio_client.fput_object(MINIO_BUCKET, cmp_obj, compare_candidate, content_type="video/mp4")
            artifact["compare_object"] = cmp_obj

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
