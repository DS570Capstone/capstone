"""Coordinate normalization and scale computation."""
from __future__ import annotations

import numpy as np
from ..cv.pose_estimator import PoseResult, KP


def compute_scale(poses: list[PoseResult]) -> float:
    """
    Compute median shoulder width across frames as normalization scale.
    Falls back to hip width, then frame width.
    """
    widths = []
    for p in poses:
        ls = p.keypoints[KP["left_shoulder"]]
        rs = p.keypoints[KP["right_shoulder"]]
        if not (np.isnan(ls).any() or np.isnan(rs).any()):
            widths.append(float(np.linalg.norm(ls - rs)))
    if widths:
        return float(np.median(widths))
    hip_widths = []
    for p in poses:
        lh = p.keypoints[KP["left_hip"]]
        rh = p.keypoints[KP["right_hip"]]
        if not (np.isnan(lh).any() or np.isnan(rh).any()):
            hip_widths.append(float(np.linalg.norm(lh - rh)))
    if hip_widths:
        return float(np.median(hip_widths))
    return 100.0  # fallback


def normalize_signal(signal: np.ndarray, scale: float) -> np.ndarray:
    return signal / (scale + 1e-8)


def compute_midline_x(poses: list[PoseResult]) -> float:
    """Compute median x-position of body midline (average of shoulder and hip centers)."""
    xs = []
    for p in poses:
        pts = [p.keypoints[KP[k]] for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]]
        valid = [pt for pt in pts if not np.isnan(pt).any()]
        if valid:
            xs.append(float(np.mean([pt[0] for pt in valid])))
    return float(np.median(xs)) if xs else 0.0
