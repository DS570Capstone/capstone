"""Video ingestion and preprocessing for OHP pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoMeta:
    video_id: str
    path: str
    fps: float
    n_frames: int
    width: int
    height: int
    duration_sec: float
    frame_timestamps: list[float] = field(default_factory=list)


def load_video_meta(path: str, video_id: Optional[str] = None) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    vid_id = video_id or os.path.splitext(os.path.basename(path))[0]
    timestamps = [i / fps for i in range(n_frames)]
    return VideoMeta(
        video_id=vid_id,
        path=path,
        fps=fps,
        n_frames=n_frames,
        width=width,
        height=height,
        duration_sec=n_frames / fps,
        frame_timestamps=timestamps,
    )


_MAX_DIM = 65535  # PIL / MediaPipe hard limit (2^16 - 1)


def iter_frames(
    path: str,
    max_height: int = 720,
    frame_step: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, bgr_frame) with optional resize and decimation."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            # Guard against corrupted frames from HEVC/VFR phone videos where
            # OpenCV can return a frame with garbage dimensions.
            if frame is None or frame.ndim != 3:
                idx += 1
                continue
            h, w = frame.shape[:2]
            if h == 0 or w == 0 or h > _MAX_DIM or w > _MAX_DIM:
                idx += 1
                continue
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
            yield idx, frame
        idx += 1
    cap.release()


def load_all_frames(
    path: str,
    max_height: int = 720,
    frame_step: int = 1,
) -> tuple[list[np.ndarray], VideoMeta]:
    meta = load_video_meta(path)
    frames = [f for _, f in iter_frames(path, max_height=max_height, frame_step=frame_step)]
    return frames, meta
