"""Barbell detection — wrist heuristic fallback (primary for back-view OHP)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose_estimator import PoseResult, KP


@dataclass
class BarDetection:
    center_x: float        # pixel
    center_y: float        # pixel
    tilt_deg: float        # angle of bar from horizontal
    left_end: tuple[float, float]
    right_end: tuple[float, float]
    confidence: float
    frame_idx: int
    method: str            # "wrist_heuristic" | "detector"


def _wrist_heuristic(pose: PoseResult, wrist_span_scale: float = 0.5) -> BarDetection:
    """
    Infer bar from wrist positions.
    Bar center = midpoint of wrists.
    Bar tilt = angle of wrist-to-wrist line.
    Bar endpoints extended by wrist_span_scale * wrist_distance.
    """
    lw = pose.keypoints[KP["left_wrist"]]
    rw = pose.keypoints[KP["right_wrist"]]
    lw_conf = pose.confidences[KP["left_wrist"]]
    rw_conf = pose.confidences[KP["right_wrist"]]
    conf = float((lw_conf + rw_conf) / 2.0)

    if np.isnan(lw).any() or np.isnan(rw).any():
        return BarDetection(
            center_x=float("nan"),
            center_y=float("nan"),
            tilt_deg=float("nan"),
            left_end=(float("nan"), float("nan")),
            right_end=(float("nan"), float("nan")),
            confidence=0.0,
            frame_idx=pose.frame_idx,
            method="wrist_heuristic",
        )

    cx = float((lw[0] + rw[0]) / 2.0)
    cy = float((lw[1] + rw[1]) / 2.0)
    dx = rw[0] - lw[0]
    dy = rw[1] - lw[1]
    tilt = float(np.degrees(np.arctan2(dy, dx)))
    dist = float(np.linalg.norm([dx, dy]))
    ext = dist * wrist_span_scale
    norm_vec = np.array([dx, dy]) / (dist + 1e-8)
    left_end = (float(lw[0] - norm_vec[0] * ext), float(lw[1] - norm_vec[1] * ext))
    right_end = (float(rw[0] + norm_vec[0] * ext), float(rw[1] + norm_vec[1] * ext))

    return BarDetection(
        center_x=cx,
        center_y=cy,
        tilt_deg=tilt,
        left_end=left_end,
        right_end=right_end,
        confidence=conf,
        frame_idx=pose.frame_idx,
        method="wrist_heuristic",
    )


class BarDetector:
    def __init__(self, backend: str = "wrist_heuristic", wrist_span_scale: float = 0.5,
                 min_confidence: float = 0.3):
        self.backend = backend
        self.wrist_span_scale = wrist_span_scale
        self.min_confidence = min_confidence

    def detect(self, frame: np.ndarray, pose: PoseResult) -> BarDetection:
        if self.backend == "wrist_heuristic":
            det = _wrist_heuristic(pose, self.wrist_span_scale)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' not implemented.")
        return det

    def detect_sequence(
        self, frames: list[np.ndarray], poses: list[PoseResult]
    ) -> list[BarDetection]:
        return [self.detect(f, p) for f, p in zip(frames, poses)]
