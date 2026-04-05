"""Top-level entry point — works from any directory."""
import os
import sys

# Always resolve to the ohp_form_pipeline directory
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)
os.chdir(PIPELINE_ROOT)

from src.app.run_single_video import run, load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single back-view OHP video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    run(args.video, args.config, args.output_dir)
