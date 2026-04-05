"""
LiftLens ML Worker
Consumes video.uploaded Kafka messages → runs OHP pipeline → stores results in PostgreSQL.
"""
import os, sys, json, tempfile, logging, shutil
import psycopg2
import psycopg2.extras
from minio import Minio
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL     = os.environ["DATABASE_URL"]
MINIO_ENDPOINT   = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET", "videos")
KAFKA_BOOTSTRAP  = os.environ["KAFKA_BOOTSTRAP"]
KAFKA_TOPIC      = "video.uploaded"
ML_ROOT          = os.environ.get("ML_ROOT", "/app/ml")
ML_CONFIG        = os.path.join(ML_ROOT, "configs", "default.yaml")

if ML_ROOT not in sys.path:
    sys.path.insert(0, ML_ROOT)

# ── Clients ───────────────────────────────────────────────────────────────────
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def get_conn():
    return psycopg2.connect(DATABASE_URL)


# ── Job state helpers ─────────────────────────────────────────────────────────
def set_status(video_id, status, stage, progress, error=None, wandb_url=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status=%s, stage=%s, progress=%s, error=%s, wandb_url=%s WHERE video_id=%s",
                (status, stage, progress, error, wandb_url, video_id),
            )


def store_result(video_id, artifact):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO results (video_id, artifact) VALUES (%s, %s) "
                "ON CONFLICT (video_id) DO UPDATE SET artifact=EXCLUDED.artifact",
                (video_id, json.dumps(artifact)),
            )


# ── Pipeline execution ────────────────────────────────────────────────────────
def process(video_id: str, object_name: str):
    log.info("Processing video_id=%s object=%s", video_id, object_name)
    set_status(video_id, "running", "Loading video", 5)

    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, object_name)

    try:
        # 1. Download from MinIO
        minio_client.fget_object(MINIO_BUCKET, object_name, local_path)
        set_status(video_id, "running", "Estimating pose", 15)

        # 2. Run pipeline
        out_dir = os.path.join(tmp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        from src.app.run_single_video import run
        set_status(video_id, "running", "Estimating depth", 35)
        artifact = run(local_path, ML_CONFIG, out_dir)
        artifact["video_id"] = video_id

        # 3. Upload annotated video back to MinIO (if produced)
        vid_id = artifact.get("video_id", video_id)
        annotated_candidates = [
            os.path.join(out_dir, "videos", f"{vid_id}_annotated.mp4"),
            os.path.join(out_dir, "annotated.mp4"),
        ]
        for candidate in annotated_candidates:
            if os.path.exists(candidate):
                ann_obj = f"annotated/{video_id}.mp4"
                minio_client.fput_object(MINIO_BUCKET, ann_obj, candidate, content_type="video/mp4")
                artifact["annotated_object"] = ann_obj
                break

        # 4. Store result
        set_status(video_id, "running", "Storing results", 95)
        store_result(video_id, artifact)
        wandb_url = artifact.get("wandb_url")
        set_status(video_id, "done", "Done", 100, wandb_url=wandb_url)
        log.info("Done video_id=%s wandb=%s", video_id, wandb_url)

    except Exception as exc:
        log.exception("Pipeline failed for video_id=%s", video_id)
        set_status(video_id, "error", "Error", 0, error=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Kafka consumer loop ───────────────────────────────────────────────────────
def main():
    log.info("Worker starting, connecting to Kafka at %s", KAFKA_BOOTSTRAP)
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="liftlens-worker",
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode()),
        enable_auto_commit=True,
    )
    log.info("Listening on topic: %s", KAFKA_TOPIC)
    for msg in consumer:
        payload = msg.value
        video_id    = payload.get("video_id")
        object_name = payload.get("object_name")
        if not video_id or not object_name:
            log.warning("Malformed message: %s", payload)
            continue
        process(video_id, object_name)


if __name__ == "__main__":
    main()
