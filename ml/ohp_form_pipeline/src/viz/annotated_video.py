"""Write annotated output video with skeleton, bar path, and fault overlays."""
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from ..cv.pose_estimator import PoseResult, KP
from ..cv.bar_detector import BarDetection

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def draw_skeleton(frame: np.ndarray, pose: PoseResult,
                  color=(0, 255, 0), radius: int = 4, thickness: int = 2) -> np.ndarray:
    """Draw skeleton in-place. Caller must pass a writable copy if the original must be preserved."""
    for (a, b) in SKELETON_EDGES:
        pa = pose.keypoints[KP[a]]
        pb = pose.keypoints[KP[b]]
        if not (np.isnan(pa).any() or np.isnan(pb).any()):
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, thickness)
    for i in range(len(pose.keypoints)):
        pt = pose.keypoints[i]
        if not np.isnan(pt).any() and pose.visible[i]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
    return frame


def draw_bar(frame: np.ndarray, bar: BarDetection,
             color=(0, 0, 255), thickness: int = 3) -> np.ndarray:
    """Draw bar in-place. Caller must pass a writable copy if the original must be preserved."""
    if np.isnan(bar.center_x):
        return frame
    le, re = bar.left_end, bar.right_end
    if not any(np.isnan(v) for v in [*le, *re]):
        cv2.line(frame, (int(le[0]), int(le[1])), (int(re[0]), int(re[1])), color, thickness)
    cv2.circle(frame, (int(bar.center_x), int(bar.center_y)), 5, color, -1)
    return frame


def draw_fault_flags(frame: np.ndarray, fault_flags: dict[str, bool],
                     phase_label: str = "") -> np.ndarray:
    """Draw fault text in-place. Caller must pass a writable copy if the original must be preserved."""
    active = [k for k, v in fault_flags.items() if v]
    y = 30
    for fault in active:
        cv2.putText(frame, f"! {fault}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        y += 20
    if phase_label:
        cv2.putText(frame, f"Phase: {phase_label}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return frame


def write_annotated_video(
    frames: list[np.ndarray],
    poses: list[PoseResult],
    bar_detections: list[BarDetection],
    fault_flags: dict[str, bool],
    phase_per_frame: list[str],
    out_path: str,
    fps: float = 30.0,
) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    bar_trail = []

    _TRAIL_MAX = 30  # cap trail length to keep drawing O(1) per frame

    for i, (frame, pose, bar) in enumerate(zip(frames, poses, bar_detections)):
        annotated = frame.copy()  # one copy per frame shared by all draw calls
        draw_skeleton(annotated, pose)
        draw_bar(annotated, bar)
        # Draw bar path trail (capped to avoid O(n²) drawing over long videos)
        if not np.isnan(bar.center_x):
            bar_trail.append((int(bar.center_x), int(bar.center_y)))
            if len(bar_trail) > _TRAIL_MAX:
                bar_trail.pop(0)
        for j in range(1, len(bar_trail)):
            cv2.line(annotated, bar_trail[j-1], bar_trail[j], (0, 165, 255), 1)
        phase = phase_per_frame[i] if i < len(phase_per_frame) else ""
        draw_fault_flags(annotated, fault_flags, phase)
        writer.write(annotated)

    writer.release()
