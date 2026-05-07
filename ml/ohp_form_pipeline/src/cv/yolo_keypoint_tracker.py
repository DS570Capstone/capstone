"""CPU-optimized YOLO keypoint tracker — no SAM2, no masks.

Extracts body-wave signal directly from YOLO-pose keypoints:
  - arm proxy  : midpoint of wrist Y (COCO indices 9, 10)
  - torso proxy: midpoint of hip   Y (COCO indices 11, 12)

For a 150-frame video this runs in ~3-10 s on CPU with yolov8n-pose.pt,
vs 3-15 min for the SAM2+yolo26x combination.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch


# COCO keypoint indices used for signal extraction
_WRIST_IDS  = [9, 10]   # left wrist, right wrist
_HIP_IDS    = [11, 12]  # left hip,   right hip
_TORSO_IDS  = [5, 6, 11, 12]  # shoulders + hips (for bounding box centre fallback)
_MIN_KP_CONF = 0.25


def _load_yolo(model_path: str, device: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from e
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YOLO(model_path)


def _frame_keypoints(
    model,
    frame_bgr: np.ndarray,
    device: str,
    fallback_xy: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    """Run YOLO on one frame.

    Returns (arm_y, torso_y, cx, cy, conf) where:
      arm_y   — mean wrist Y (image pixels, top=0)
      torso_y — mean hip Y
      cx, cy  — body centre (for overlay / debugging)
      conf    — detection confidence (0 if no person found)
    """
    h, w = frame_bgr.shape[:2]
    res = model.predict(
        source=frame_bgr,
        verbose=False,
        device=device,
        imgsz=640,
        conf=0.2,
        max_det=1,
    )

    if not res or res[0].keypoints is None or len(res[0].keypoints) == 0:
        # No keypoints — try bounding box centre
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            xyxy = res[0].boxes.xyxy[0].detach().cpu().numpy()
            cx = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
            cy = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
            conf = float(res[0].boxes.conf[0].detach().cpu().numpy())
            return cy, cy, cx, cy, conf
        return float("nan"), float("nan"), fallback_xy[0], fallback_xy[1], 0.0

    kp_xy   = res[0].keypoints.xy[0].detach().cpu().numpy()    # (17, 2)
    kp_conf = res[0].keypoints.conf[0].detach().cpu().numpy()  # (17,)

    def _mean_y(ids: list[int]) -> float:
        ys = [kp_xy[i][1] for i in ids if i < len(kp_xy) and kp_conf[i] >= _MIN_KP_CONF]
        return float(np.mean(ys)) if ys else float("nan")

    arm_y   = _mean_y(_WRIST_IDS)
    torso_y = _mean_y(_HIP_IDS)

    # Body centre from torso keypoints
    torso_pts = [kp_xy[i] for i in _TORSO_IDS if i < len(kp_xy) and kp_conf[i] >= _MIN_KP_CONF]
    if torso_pts:
        cx, cy = np.mean(torso_pts, axis=0)
    else:
        cx = w * 0.5
        cy = h * 0.5

    conf = float(kp_conf.max())
    return arm_y, torso_y, float(cx), float(cy), conf


def run_yolo_keypoints(
    video_path: str,
    yolo_model: str,
    output_dir: str,
    yolo_every_n: int = 1,
    on_frame_progress=None,
    frame_step: int = 1,
    max_height: int = 720,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process a video and return per-frame keypoint signals.

    Args:
        video_path:   Path to input .mp4 / .mov
        yolo_model:   Path to YOLO-pose .pt file (e.g. yolov8n-pose.pt)
        output_dir:   Where to write overlay video + summary JSON
        yolo_every_n: Run YOLO every N frames; interpolate arm_y/torso_y between
        on_frame_progress: optional callback(frame_idx: int, total_frames: int)
        frame_step:   Skip frames (run YOLO on frame 0, frame_step, 2*frame_step …)
        device:       "cpu" | "cuda" (auto-detected if None)

    Returns:
        arm_y_arr   — float64 array length == processed_frames
        torso_y_arr — float64 array length == processed_frames
        summary     — dict with diagnostics
    """
    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_yolo(yolo_model, _device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)
    out_video = os.path.join(output_dir, "yolo_overlay.mp4")
    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    step = max(1, int(frame_step))
    arm_vals:   list[float] = []
    torso_vals: list[float] = []
    yolo_detections = 0

    # Cache last good detection for interpolation
    last_arm_y:   float = float("nan")
    last_torso_y: float = float("nan")
    last_cx: float = width * 0.5
    last_cy: float = height * 0.5

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if on_frame_progress:
            on_frame_progress(idx, n_frames)

        # Only run YOLO on selected frames
        run_yolo = (idx % step == 0)

        if run_yolo:
            infer_frame = frame
            scale_back = 1.0
            if max_height and frame.shape[0] > int(max_height):
                scale = float(max_height) / float(frame.shape[0])
                infer_w = int(round(frame.shape[1] * scale))
                infer_h = int(max_height)
                infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                scale_back = 1.0 / scale
            arm_y, torso_y, cx, cy, conf = _frame_keypoints(
                model, infer_frame, _device, (last_cx, last_cy)
            )
            if scale_back != 1.0:
                if not np.isnan(arm_y):
                    arm_y *= scale_back
                if not np.isnan(torso_y):
                    torso_y *= scale_back
                cx *= scale_back
                cy *= scale_back
            if conf > 0:
                yolo_detections += 1
            if not np.isnan(arm_y):
                last_arm_y = arm_y
            if not np.isnan(torso_y):
                last_torso_y = torso_y
            last_cx, last_cy = cx, cy
        else:
            # Reuse last values for skipped frames
            arm_y, torso_y = last_arm_y, last_torso_y
            cx, cy = last_cx, last_cy

        arm_vals.append(arm_y)
        torso_vals.append(torso_y)

        # Draw a simple overlay dot
        overlay = frame.copy()
        if not (np.isnan(cx) or np.isnan(cy)):
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 6, (0, 255, 0), -1)
        writer.write(overlay)
        idx += 1

    cap.release()
    writer.release()

    arm_arr   = np.array(arm_vals,   dtype=np.float64)
    torso_arr = np.array(torso_vals, dtype=np.float64)

    summary = {
        "video": video_path,
        "yolo_model": yolo_model,
        "frames_expected": n_frames,
        "frames_processed": idx,
        "yolo_detection_frames": yolo_detections,
        "overlay_video": out_video,
        "avg_arm_y": float(np.nanmean(arm_arr)) if len(arm_arr) else 0.0,
        "avg_torso_y": float(np.nanmean(torso_arr)) if len(torso_arr) else 0.0,
        "max_height": int(max_height),
    }

    summary_path = os.path.join(output_dir, "yolo_kp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return arm_arr, torso_arr, summary
