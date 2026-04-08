"""SAM2 frame-by-frame video segmentation runner.

This script applies SAM2 on every frame of a video using a point prompt.
It is intentionally standalone so it can be invoked from WSL directly.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch


@dataclass
class SAM2FrameResult:
    frame_idx: int
    score: float
    mask_area: int


class SAM2PerFrameEstimator:
    """Runs SAM2 image predictor on each frame independently."""

    def __init__(self, checkpoint: str, model_cfg: str, device: Optional[str] = None):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise RuntimeError(
                "SAM2 imports failed. Install SAM2 in the active environment first."
            ) from e

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor(
            build_sam2(config_file=model_cfg, ckpt_path=checkpoint, device=self.device)
        )

    def process_frame(
        self, bgr_frame: np.ndarray, point_xy: tuple[float, float]
    ) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])
        return best_mask, best_score


def _overlay_mask(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.zeros_like(out)
    color[:, :, 1] = 255
    alpha = 0.35
    mask = mask_u8.astype(bool)
    out[mask] = cv2.addWeighted(out[mask], 1.0 - alpha, color[mask], alpha, 0)
    return out


def run_video(
    video_path: str,
    checkpoint: str,
    model_cfg: str,
    output_dir: str,
    prompt_x: Optional[float] = None,
    prompt_y: Optional[float] = None,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    px = float(prompt_x) if prompt_x is not None else width * 0.5
    py = float(prompt_y) if prompt_y is not None else height * 0.5

    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    out_video = os.path.join(output_dir, "sam2_overlay.mp4")
    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    estimator = SAM2PerFrameEstimator(checkpoint=checkpoint, model_cfg=model_cfg)
    per_frame: list[SAM2FrameResult] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, score = estimator.process_frame(frame, point_xy=(px, py))
        np.save(os.path.join(masks_dir, f"frame_{idx:06d}.npy"), mask)
        overlay = _overlay_mask(frame, mask)
        writer.write(overlay)

        per_frame.append(
            SAM2FrameResult(frame_idx=idx, score=score, mask_area=int(mask.sum()))
        )
        idx += 1

    cap.release()
    writer.release()

    summary = {
        "video": video_path,
        "checkpoint": checkpoint,
        "model_cfg": model_cfg,
        "frames_expected": n_frames,
        "frames_processed": idx,
        "prompt_point": [px, py],
        "overlay_video": out_video,
        "avg_score": float(np.mean([r.score for r in per_frame])) if per_frame else 0.0,
        "avg_mask_area": (
            float(np.mean([r.mask_area for r in per_frame])) if per_frame else 0.0
        ),
    }

    summary_path = os.path.join(output_dir, "sam2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM2 per-frame segmentation on a video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--checkpoint",
        default=r"C:\Users\josep\Desktop\cappy\capstone\ml\models\sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_x", type=float, default=None)
    parser.add_argument("--prompt_y", type=float, default=None)
    args = parser.parse_args()

    summary = run_video(
        video_path=args.video,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        output_dir=args.output_dir,
        prompt_x=args.prompt_x,
        prompt_y=args.prompt_y,
    )
    print("SAM2 run complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
