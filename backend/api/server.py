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


@app.post("/api/upload")
def upload():
    """Receive a video file, store in MinIO, return a video_id."""
    if "video" not in request.files:
        return jsonify(error="No video file"), 400

    file      = request.files["video"]
    video_id  = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
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
                """INSERT INTO jobs (video_id, object_name, status, stage)
                   VALUES (%s, %s, 'queued', 'Queued')""",
                (video_id, obj_name),
            )

    return jsonify(video_id=video_id, object_name=obj_name), 201


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
        # Fallback: run inline in background thread (dev mode)
        threading.Thread(target=_run_worker_inline, args=(video_id, job["object_name"]), daemon=True).start()

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
            cur.execute("SELECT status, progress, stage, error, wandb_url FROM jobs WHERE video_id=%s", (video_id,))
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


# ── Inline worker fallback (no Kafka) ────────────────────────────────────────
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
        from src.app.run_single_video import run
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        artifact = run(tmp_path, ml_config, out_dir)
        artifact["video_id"] = video_id

        _set("running", "Storing results", 90)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO results (video_id, artifact) VALUES (%s, %s) "
                    "ON CONFLICT (video_id) DO UPDATE SET artifact=EXCLUDED.artifact",
                    (video_id, json.dumps(artifact)),
                )
                wandb_url = artifact.get("wandb_url")
                cur.execute(
                    "UPDATE jobs SET status='done', stage='Done', progress=100, wandb_url=%s WHERE video_id=%s",
                    (wandb_url, video_id),
                )
    except Exception as exc:
        _set("error", "Error", 0, str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
