"""
run_full_pipeline.py — End-to-end pipeline: process videos → build dataset → train model.

This script orchestrates the full workflow:
  1. Batch-process all 500 videos (pose, depth, features, annotated outputs)
  2. Extract training data from the S&C manual PDF + video artifacts
  3. Train the ExerciseAware vision adapter
  4. LoRA fine-tune the language model
  5. Run inference on a sample video

Usage:
    python run_full_pipeline.py --step all
    python run_full_pipeline.py --step process   # just batch process videos
    python run_full_pipeline.py --step dataset    # just build training dataset
    python run_full_pipeline.py --step train      # just train the model
    python run_full_pipeline.py --step infer --video path/to/video.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)
os.chdir(PIPELINE_ROOT)

# Paths
VIDEO_DIR = r"C:\Users\josep\Desktop\Capstone_new\videos\First 500 Vids"
PDF_PATH = r"C:\Users\josep\Desktop\Capstone_new\basics_of_strength_and_conditioning_manual.pdf"
BATCH_OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "batch_outputs")
DATASET_PATH = os.path.join(PIPELINE_ROOT, "data", "training_data.jsonl")
CHECKPOINT_DIR = os.path.join(PIPELINE_ROOT, "checkpoints")
CONFIG_PATH = os.path.join(PIPELINE_ROOT, "configs", "default.yaml")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def step_process(max_videos: int = 0):
    """Step 1: Batch-process all videos."""
    print("\n" + "=" * 70)
    print("STEP 1: BATCH VIDEO PROCESSING")
    print("=" * 70)
    from batch_process import main as batch_main
    sys.argv = [
        "batch_process.py",
        "--input_dir", VIDEO_DIR,
        "--output_dir", BATCH_OUTPUT_DIR,
        "--config", CONFIG_PATH,
    ]
    if max_videos > 0:
        sys.argv.extend(["--max_videos", str(max_videos)])
    batch_main()


def step_dataset():
    """Step 2: Build training dataset from PDF + artifacts."""
    print("\n" + "=" * 70)
    print("STEP 2: BUILD TRAINING DATASET")
    print("=" * 70)
    from src.model.pdf_dataset_builder import build_dataset
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    build_dataset(
        pdf_path=PDF_PATH,
        artifact_dir=BATCH_OUTPUT_DIR,
        output_path=DATASET_PATH,
    )


def step_train(epochs: int = 10, batch_size: int = 4):
    """Step 3: Train vision adapter + language LoRA + baseline. All logged to W&B."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING (with adapter + baseline comparison)")
    print("=" * 70)
    from src.model.train import main as train_main
    sys.argv = [
        "train.py",
        "--stage", "all",
        "--data_dir", BATCH_OUTPUT_DIR,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--output_dir", CHECKPOINT_DIR,
        "--model_id", MODEL_ID,
    ]
    train_main()


def step_infer(video_path: str):
    """Step 4: Run inference on a video."""
    print("\n" + "=" * 70)
    print("STEP 4: INFERENCE")
    print("=" * 70)
    from src.model.inference import ExerciseAwareCoach

    adapter_path = os.path.join(CHECKPOINT_DIR, "vision_adapter.pt")
    lora_path = os.path.join(CHECKPOINT_DIR, "language_lora")

    coach = ExerciseAwareCoach(
        model_id=MODEL_ID,
        adapter_path=adapter_path if os.path.exists(adapter_path) else None,
        lora_path=lora_path if os.path.isdir(lora_path) else None,
    )
    result = coach.generate_feedback(video_path)

    print(f"\nQuality: {result['quality'].get('grade', '?')}")
    print(f"Faults: {result['faults']}")
    print(f"\n{result['feedback']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Full ExerciseAware pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "process", "dataset", "train", "infer"],
        default="all",
    )
    parser.add_argument("--video", default=None, help="Video path for inference step")
    parser.add_argument("--max_videos", type=int, default=0, help="Limit videos to process (0=all)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.step in ("all", "process"):
        step_process(args.max_videos)

    if args.step in ("all", "dataset"):
        step_dataset()

    if args.step in ("all", "train"):
        step_train(args.epochs, args.batch_size)

    if args.step == "infer":
        if not args.video:
            # Pick first video from input dir
            import glob
            vids = sorted(glob.glob(os.path.join(VIDEO_DIR, "**/*.mp4"), recursive=True))
            if vids:
                args.video = vids[0]
                print(f"No --video specified, using: {args.video}")
            else:
                print("No video found. Specify --video path.")
                return
        step_infer(args.video)

    if args.step == "all":
        # Run inference on first video as demo
        import glob
        vids = sorted(glob.glob(os.path.join(VIDEO_DIR, "**/*.mp4"), recursive=True))
        if vids:
            step_infer(vids[0])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
