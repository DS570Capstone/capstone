"""Write annotated output video with skeleton, bar path, and fault overlays."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional
import cv2
import numpy as np


def _reencode_h264(src: str, dst: str) -> bool:
    """Re-encode src → dst as H.264 in-place using ffmpeg. Returns True on success."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", src,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                dst,
            ],
            capture_output=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def _write_video_h264(out_path: str, writer_fn) -> None:
    """Write video via writer_fn to a temp file, re-encode to H.264, move to out_path."""
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, "raw.mp4")
        writer_fn(tmp_path)
        if not _reencode_h264(tmp_path, out_path):
            # ffmpeg unavailable — just move the raw file so something is written
            shutil.move(tmp_path, out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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


def write_comparison_video(
    frames: list[np.ndarray],
    poses_raw: list[PoseResult],
    poses_vp3d: list[PoseResult],
    out_path: str,
    fps: float = 30.0,
) -> None:
    """
    Side-by-side video: left = raw MediaPipe (red), right = VP3D-cleaned (cyan).
    Output is H.264 MP4 (browser-compatible). Falls back to mp4v if ffmpeg unavailable.
    """
    if not frames:
        return

    def _write_raw(tmp_path: str) -> None:
        h, w = frames[0].shape[:2]
        canvas_w = w * 2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (canvas_w, h))

        RAW_COLOR  = (60, 60, 255)   # red-ish  — raw MediaPipe
        VP3D_COLOR = (255, 220, 0)   # cyan     — VP3D cleaned

        for i, frame in enumerate(frames):
            raw  = poses_raw[i]  if i < len(poses_raw)  else None
            vp3d = poses_vp3d[i] if i < len(poses_vp3d) else None

            left  = frame.copy()
            right = frame.copy()

            if raw is not None:
                draw_skeleton(left, raw, color=RAW_COLOR, radius=4, thickness=2)
                n_vis = int(raw.visible.sum())
                cv2.putText(left, f"MediaPipe raw  {n_vis}/33 kp",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RAW_COLOR, 1, cv2.LINE_AA)

            if vp3d is not None:
                draw_skeleton(right, vp3d, color=VP3D_COLOR, radius=4, thickness=2)
                n_vis = int(vp3d.visible.sum())
                cv2.putText(right, f"VP3D cleaned  {n_vis}/33 kp",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, VP3D_COLOR, 1, cv2.LINE_AA)

            h2 = frames[0].shape[0]
            cv2.putText(left,  f"frame {i}", (8, h2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(right, f"frame {i}", (8, h2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

            canvas = np.concatenate([left, right], axis=1)
            writer.write(canvas)

        writer.release()

    _write_video_h264(out_path, _write_raw)


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

    def _write_raw(tmp_path: str) -> None:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        bar_trail: list[tuple[int, int]] = []
        _TRAIL_MAX = 30

        for i, (frame, pose, bar) in enumerate(zip(frames, poses, bar_detections)):
            annotated = frame.copy()
            draw_skeleton(annotated, pose)
            draw_bar(annotated, bar)
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

    _write_video_h264(out_path, _write_raw)
