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
    prompt_x: float
    prompt_y: float
    yolo_conf: float


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
        self,
        bgr_frame: np.ndarray,
        point_xy: tuple[float, float],
        prev_mask: Optional[np.ndarray] = None,
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

        best_idx = self._select_mask_idx(masks, scores, prev_mask)
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])
        return best_mask, best_score

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / max(union, 1))

    def _select_mask_idx(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        prev_mask: Optional[np.ndarray],
    ) -> int:
        if prev_mask is None:
            return int(np.argmax(scores))

        best_idx = 0
        best_value = -1e9
        prev_area = max(int(prev_mask.astype(bool).sum()), 1)
        for idx in range(len(scores)):
            m = masks[idx].astype(np.uint8)
            sam_score = float(scores[idx])
            iou = self._mask_iou(m, prev_mask)
            area = max(int(m.astype(bool).sum()), 1)
            area_ratio = min(area, prev_area) / max(area, prev_area)
            value = 0.65 * sam_score + 0.25 * iou + 0.10 * area_ratio
            if value > best_value:
                best_value = value
                best_idx = idx
        return int(best_idx)


class YOLOPosePromptTracker:
    """Generates a stable SAM2 prompt from YOLO pose detections."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        fallback_model_path: Optional[str] = None,
    ):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "Ultralytics YOLO is required for robust tracking. Install with `pip install ultralytics`."
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_model_path = ""

        candidate_paths = [model_path]
        if fallback_model_path:
            candidate_paths.append(fallback_model_path)

        load_errors: list[str] = []
        for path in candidate_paths:
            if not os.path.exists(path):
                load_errors.append(f"missing: {path}")
                continue
            try:
                self.model = YOLO(path)
                self.loaded_model_path = path
                break
            except Exception as e:
                load_errors.append(f"{path}: {e}")
        else:
            raise RuntimeError(
                "Could not load any YOLO model for tracking. "
                + " | ".join(load_errors)
            )

    @staticmethod
    def _valid_points(xy: np.ndarray, conf: np.ndarray, min_conf: float) -> np.ndarray:
        keep = conf >= min_conf
        return xy[keep]

    def estimate_prompt(
        self,
        frame_bgr: np.ndarray,
        fallback_xy: tuple[float, float],
        min_kp_conf: float = 0.25,
    ) -> tuple[float, float, float]:
        res = self.model.predict(
            source=frame_bgr,
            verbose=False,
            device=self.device,
            imgsz=640,
            conf=0.2,
            max_det=1,
        )

        if not res or res[0].keypoints is None or len(res[0].keypoints) == 0:
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxy = res[0].boxes.xyxy[0].detach().cpu().numpy()
                conf = float(res[0].boxes.conf[0].detach().cpu().numpy())
                cx = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
                cy = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
                return cx, cy, conf
            return float(fallback_xy[0]), float(fallback_xy[1]), 0.0

        kpts_xy = res[0].keypoints.xy[0].detach().cpu().numpy()
        kpts_conf = res[0].keypoints.conf[0].detach().cpu().numpy()

        # COCO indices: shoulders 5,6; hips 11,12. Torso center is a robust prompt.
        torso_ids = np.array([5, 6, 11, 12], dtype=np.int32)
        torso_xy = kpts_xy[torso_ids]
        torso_conf = kpts_conf[torso_ids]
        valid_torso = self._valid_points(torso_xy, torso_conf, min_kp_conf)

        if len(valid_torso) > 0:
            center = valid_torso.mean(axis=0)
            return float(center[0]), float(center[1]), float(torso_conf.max())

        valid_all = self._valid_points(kpts_xy, kpts_conf, min_kp_conf)
        if len(valid_all) > 0:
            center = valid_all.mean(axis=0)
            return float(center[0]), float(center[1]), float(kpts_conf.max())

        return float(fallback_xy[0]), float(fallback_xy[1]), 0.0


def _overlay_mask(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.zeros_like(out)
    color[:, :, 1] = 255
    alpha = 0.35
    mask = mask_u8.astype(bool)
    out[mask] = cv2.addWeighted(out[mask], 1.0 - alpha, color[mask], alpha, 0)
    return out


def _smooth_prompt(
    prev_xy: tuple[float, float], curr_xy: tuple[float, float], alpha: float
) -> tuple[float, float]:
    x = alpha * curr_xy[0] + (1.0 - alpha) * prev_xy[0]
    y = alpha * curr_xy[1] + (1.0 - alpha) * prev_xy[1]
    return float(x), float(y)


def run_video(
    video_path: str,
    checkpoint: str,
    model_cfg: str,
    yolo_model: str,
    yolo_fallback_model: Optional[str],
    output_dir: str,
    prompt_x: Optional[float] = None,
    prompt_y: Optional[float] = None,
    yolo_every_n: int = 1,
    prompt_ema_alpha: float = 0.45,
    on_frame_progress=None,
    frame_step: int = 1,
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
    yolo_prompt = YOLOPosePromptTracker(
        model_path=yolo_model,
        fallback_model_path=yolo_fallback_model,
    )
    per_frame: list[SAM2FrameResult] = []

    step = max(1, int(frame_step))
    idx = 0
    prev_mask: Optional[np.ndarray] = None
    prompt_xy = (px, py)
    yolo_detections = 0
    last_processed_mask: Optional[np.ndarray] = None
    last_processed_idx: int = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if on_frame_progress:
            on_frame_progress(idx, n_frames)

        # Skip frames: only run SAM2 every `step` frames
        if step > 1 and idx > 0 and idx % step != 0:
            # Reuse last mask for skipped frames
            if last_processed_mask is not None:
                np.save(os.path.join(masks_dir, f"frame_{idx:06d}.npy"), last_processed_mask)
            per_frame.append(
                SAM2FrameResult(
                    frame_idx=idx,
                    score=0.0,
                    mask_area=int(last_processed_mask.sum()) if last_processed_mask is not None else 0,
                    prompt_x=float(prompt_xy[0]),
                    prompt_y=float(prompt_xy[1]),
                    yolo_conf=0.0,
                )
            )
            idx += 1
            continue

        if idx == 0 or (yolo_every_n > 0 and idx % yolo_every_n == 0):
            raw_x, raw_y, yolo_conf = yolo_prompt.estimate_prompt(
                frame_bgr=frame,
                fallback_xy=prompt_xy,
            )
            if yolo_conf > 0:
                yolo_detections += 1
            prompt_xy = _smooth_prompt(prompt_xy, (raw_x, raw_y), prompt_ema_alpha)
        else:
            yolo_conf = 0.0

        mask, score = estimator.process_frame(
            frame,
            point_xy=prompt_xy,
            prev_mask=prev_mask,
        )
        prev_mask = mask
        last_processed_mask = mask
        last_processed_idx = idx

        np.save(os.path.join(masks_dir, f"frame_{idx:06d}.npy"), mask)
        overlay = _overlay_mask(frame, mask)
        cv2.circle(
            overlay,
            (int(round(prompt_xy[0])), int(round(prompt_xy[1]))),
            4,
            (0, 0, 255),
            -1,
        )
        writer.write(overlay)

        per_frame.append(
            SAM2FrameResult(
                frame_idx=idx,
                score=score,
                mask_area=int(mask.sum()),
                prompt_x=float(prompt_xy[0]),
                prompt_y=float(prompt_xy[1]),
                yolo_conf=float(yolo_conf),
            )
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
        "yolo_model": yolo_prompt.loaded_model_path,
        "yolo_detection_frames": yolo_detections,
        "avg_score": float(np.mean([r.score for r in per_frame])) if per_frame else 0.0,
        "avg_mask_area": (
            float(np.mean([r.mask_area for r in per_frame])) if per_frame else 0.0
        ),
        "avg_yolo_conf": (
            float(np.mean([r.yolo_conf for r in per_frame])) if per_frame else 0.0
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
        default="/scratch/jnolas77/fitness/capstone/ml/models/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--yolo_model",
        default="/scratch/jnolas77/fitness/capstone/ml/models/yolo26x-pose.pt",
    )
    parser.add_argument(
        "--yolo_fallback_model",
        default="/scratch/jnolas77/Question_Generator_Model/Question_Generator_Model/yolov8n.pt",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_x", type=float, default=None)
    parser.add_argument("--prompt_y", type=float, default=None)
    parser.add_argument("--yolo_every_n", type=int, default=1)
    parser.add_argument("--prompt_ema_alpha", type=float, default=0.45)
    args = parser.parse_args()

    summary = run_video(
        video_path=args.video,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        yolo_model=args.yolo_model,
        yolo_fallback_model=args.yolo_fallback_model,
        output_dir=args.output_dir,
        prompt_x=args.prompt_x,
        prompt_y=args.prompt_y,
        yolo_every_n=args.yolo_every_n,
        prompt_ema_alpha=args.prompt_ema_alpha,
    )
    print("SAM2 run complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
