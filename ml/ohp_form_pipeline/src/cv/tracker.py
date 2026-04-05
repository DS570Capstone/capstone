"""Temporal smoothing of detected keypoints and bar positions."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .pose_estimator import PoseResult, KEYPOINT_NAMES, KP
from .bar_detector import BarDetection


class OneEuroFilter:
    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smooth_1d(signal: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing with NaN handling via linear interpolation."""
    arr = signal.copy().astype(np.float64)
    # Interpolate NaNs
    nans = np.isnan(arr)
    if nans.all():
        return arr
    if nans.any():
        xs = np.where(~nans)[0]
        arr[nans] = np.interp(np.where(nans)[0], xs, arr[xs])
    if len(arr) < window:
        return arr
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if w < poly + 2:
        return arr
    return savgol_filter(arr, w, poly)


def smooth_poses(
    poses: list[PoseResult],
    window: int = 5,
    poly: int = 2,
    method: str = "savgol",
    fps: float = 30.0,
    one_euro_min_cutoff: float = 1.0,
    one_euro_beta: float = 0.02,
    one_euro_d_cutoff: float = 1.0,
    min_visibility: float = 0.5,
    max_jump_px: float = 60.0,
    max_interp_gap: int = 8,
) -> list[PoseResult]:
    """Smooth all keypoint coordinates over time."""
    if method == "none":
        return [
            PoseResult(
                keypoints=p.keypoints.copy(),
                confidences=p.confidences.copy(),
                visible=p.visible.copy(),
                frame_idx=p.frame_idx,
            )
            for p in poses
        ]

    if method == "one_euro":
        return smooth_poses_one_euro(
            poses,
            fps=fps,
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
            min_visibility=min_visibility,
            max_jump_px=max_jump_px,
            max_interp_gap=max_interp_gap,
        )

    n = len(poses)
    n_kp = len(KEYPOINT_NAMES)
    kp_x = np.array(
        [[p.keypoints[k, 0] for p in poses] for k in range(n_kp)]
    )  # (n_kp, T)
    kp_y = np.array([[p.keypoints[k, 1] for p in poses] for k in range(n_kp)])

    smoothed_x = np.stack([smooth_1d(kp_x[k], window, poly) for k in range(n_kp)])
    smoothed_y = np.stack([smooth_1d(kp_y[k], window, poly) for k in range(n_kp)])

    out = []
    for t, p in enumerate(poses):
        kp = np.stack([smoothed_x[:, t], smoothed_y[:, t]], axis=1)
        out.append(
            PoseResult(
                keypoints=kp,
                confidences=p.confidences.copy(),
                visible=p.visible.copy(),
                frame_idx=p.frame_idx,
            )
        )
    return out


def smooth_poses_one_euro(
    poses: list[PoseResult],
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.02,
    d_cutoff: float = 1.0,
    min_visibility: float = 0.5,
    max_jump_px: float = 60.0,
    max_interp_gap: int = 8,
) -> list[PoseResult]:
    if not poses:
        return []

    n_kp = len(KEYPOINT_NAMES)
    tracks_x = np.array(
        [[p.keypoints[k, 0] for p in poses] for k in range(n_kp)], dtype=np.float64
    )
    tracks_y = np.array(
        [[p.keypoints[k, 1] for p in poses] for k in range(n_kp)], dtype=np.float64
    )
    tracks_vis = np.array(
        [[p.confidences[k] for p in poses] for k in range(n_kp)], dtype=np.float64
    )

    # Robust combine: drop low-confidence points, reject sudden jumps, then interpolate short gaps.
    for k in range(n_kp):
        x = tracks_x[k].copy()
        y = tracks_y[k].copy()
        vis = tracks_vis[k]

        valid = (~np.isnan(x)) & (~np.isnan(y)) & (vis >= min_visibility)
        prev_idx = None
        for t in range(len(x)):
            if not valid[t]:
                continue
            if prev_idx is not None:
                jump = np.hypot(x[t] - x[prev_idx], y[t] - y[prev_idx])
                if jump > max_jump_px:
                    valid[t] = False
                    continue
            prev_idx = t

        x[~valid] = np.nan
        y[~valid] = np.nan
        tracks_x[k] = _interp_with_gap_limit(x, max_interp_gap)
        tracks_y[k] = _interp_with_gap_limit(y, max_interp_gap)

    filters_x = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n_kp)]
    filters_y = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n_kp)]

    out = []
    for t_idx, p in enumerate(poses):
        t = float(t_idx) / max(float(fps), 1e-6)
        kp = p.keypoints.copy().astype(np.float64)
        for k in range(n_kp):
            x, y = tracks_x[k, t_idx], tracks_y[k, t_idx]
            if np.isnan(x) or np.isnan(y):
                kp[k, 0] = np.nan
                kp[k, 1] = np.nan
                continue
            kp[k, 0] = filters_x[k](float(x), t)
            kp[k, 1] = filters_y[k](float(y), t)

        out.append(
            PoseResult(
                keypoints=kp,
                confidences=p.confidences.copy(),
                visible=p.visible.copy(),
                frame_idx=p.frame_idx,
            )
        )
    return out


def _interp_with_gap_limit(arr: np.ndarray, max_gap: int) -> np.ndarray:
    out = arr.copy().astype(np.float64)
    valid_idx = np.where(~np.isnan(out))[0]
    if len(valid_idx) < 2:
        return out

    n = len(out)
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue
        start = i
        while i < n and np.isnan(out[i]):
            i += 1
        end = i - 1
        left = start - 1
        right = i
        gap_len = end - start + 1
        if (
            left >= 0
            and right < n
            and gap_len <= max_gap
            and not np.isnan(out[left])
            and not np.isnan(out[right])
        ):
            out[start : end + 1] = np.interp(
                np.arange(start, end + 1), [left, right], [out[left], out[right]]
            )
    return out


def smooth_bar_detections(
    detections: list[BarDetection], window: int = 5, poly: int = 2
) -> list[BarDetection]:
    n = len(detections)
    cx = np.array([d.center_x for d in detections])
    cy = np.array([d.center_y for d in detections])
    tilt = np.array([d.tilt_deg for d in detections])

    cx_s = smooth_1d(cx, window, poly)
    cy_s = smooth_1d(cy, window, poly)
    tilt_s = smooth_1d(tilt, window, poly)

    return [
        BarDetection(
            center_x=float(cx_s[i]),
            center_y=float(cy_s[i]),
            tilt_deg=float(tilt_s[i]),
            left_end=detections[i].left_end,
            right_end=detections[i].right_end,
            confidence=detections[i].confidence,
            frame_idx=detections[i].frame_idx,
            method=detections[i].method,
        )
        for i in range(n)
    ]
