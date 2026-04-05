"""Build the 4 canonical trajectory signals from raw pose + bar detections."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ..cv.pose_estimator import PoseResult, KP
from ..cv.bar_detector import BarDetection
from .smoothing import savgol_smooth
from .normalization import compute_scale, normalize_signal, compute_midline_x


def _resample(signal: np.ndarray, target_len: int) -> np.ndarray:
    if len(signal) == target_len:
        return signal
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    nans = np.isnan(signal)
    if nans.any():
        valid = ~nans
        if valid.sum() < 2:
            return np.zeros(target_len)
        f = interp1d(x_old[valid], signal[valid], kind="linear", fill_value="extrapolate")
        resampled = f(x_new)
    else:
        f = interp1d(x_old, signal, kind="linear")
        resampled = f(x_new)
    return resampled.astype(np.float32)


def build_bar_path_trajectory(
    bar_detections: list[BarDetection],
    scale: float,
    midline_x: float,
    resample_len: int = 128,
    smooth_window: int = 7,
) -> np.ndarray:
    """
    bar_path_trajectory: normalized vertical displacement of bar center.
    Inverted so up = positive value.
    """
    cy = np.array([d.center_y for d in bar_detections], dtype=np.float64)
    cy_smooth = savgol_smooth(cy, smooth_window)
    # Invert and normalize: bar going up = positive
    cy_inv = -(cy_smooth - np.nanmean(cy_smooth)) / (scale + 1e-8)
    return _resample(cy_inv.astype(np.float32), resample_len)


def build_arm_trajectory(
    poses: list[PoseResult],
    scale: float,
    resample_len: int = 128,
    smooth_window: int = 7,
) -> np.ndarray:
    """
    arm_trajectory: symmetry-weighted bilateral wrist elevation.
    Positive = both wrists high; asymmetry reduces the signal.
    """
    lw_y = np.array([p.keypoints[KP["left_wrist"], 1] for p in poses])
    rw_y = np.array([p.keypoints[KP["right_wrist"], 1] for p in poses])
    ls_y = np.array([p.keypoints[KP["left_shoulder"], 1] for p in poses])

    # Elevation = distance above shoulder line, inverted
    lw_elev = -(lw_y - ls_y) / (scale + 1e-8)
    rw_elev = -(rw_y - ls_y) / (scale + 1e-8)

    # Bilateral mean weighted by symmetry (1 - normalized diff)
    diff = np.abs(lw_y - rw_y) / (scale + 1e-8)
    sym_weight = np.clip(1.0 - diff, 0.0, 1.0)
    mean_elev = (lw_elev + rw_elev) / 2.0
    arm_traj = mean_elev * sym_weight

    smoothed = savgol_smooth(arm_traj, smooth_window)
    return _resample(smoothed.astype(np.float32), resample_len)


def build_legs_trajectory(
    poses: list[PoseResult],
    scale: float,
    resample_len: int = 128,
    smooth_window: int = 7,
) -> np.ndarray:
    """
    legs_trajectory: bilateral knee stability proxy.
    Near-zero = stable; high values = leg movement under load.
    """
    lk_y = np.array([p.keypoints[KP["left_knee"], 1] for p in poses])
    rk_y = np.array([p.keypoints[KP["right_knee"], 1] for p in poses])

    # Deviation from median (stability signal)
    lk_dev = np.abs(lk_y - np.nanmedian(lk_y)) / (scale + 1e-8)
    rk_dev = np.abs(rk_y - np.nanmedian(rk_y)) / (scale + 1e-8)
    legs_traj = (lk_dev + rk_dev) / 2.0

    smoothed = savgol_smooth(legs_traj, smooth_window)
    return _resample(smoothed.astype(np.float32), resample_len)


def build_core_trajectory(
    poses: list[PoseResult],
    scale: float,
    resample_len: int = 128,
    smooth_window: int = 7,
) -> np.ndarray:
    """
    core_trajectory: lateral trunk shift signal.
    Near-zero = trunk stable; positive = lateral sway.
    """
    ls_x = np.array([p.keypoints[KP["left_shoulder"], 0] for p in poses])
    rs_x = np.array([p.keypoints[KP["right_shoulder"], 0] for p in poses])
    lh_x = np.array([p.keypoints[KP["left_hip"], 0] for p in poses])
    rh_x = np.array([p.keypoints[KP["right_hip"], 0] for p in poses])

    # Trunk center x = mean of shoulder and hip centers
    trunk_cx = np.nanmean(np.stack([
        (ls_x + rs_x) / 2.0,
        (lh_x + rh_x) / 2.0,
    ], axis=0), axis=0)

    # Lateral shift from median
    shift = np.abs(trunk_cx - np.nanmedian(trunk_cx)) / (scale + 1e-8)
    smoothed = savgol_smooth(shift, smooth_window)
    return _resample(smoothed.astype(np.float32), resample_len)


def build_all_trajectories(
    poses: list[PoseResult],
    bar_detections: list[BarDetection],
    resample_len: int = 128,
    smooth_window: int = 7,
) -> dict[str, list[float]]:
    scale = compute_scale(poses)
    midline_x = compute_midline_x(poses)

    bar_traj = build_bar_path_trajectory(bar_detections, scale, midline_x, resample_len, smooth_window)
    arm_traj = build_arm_trajectory(poses, scale, resample_len, smooth_window)
    legs_traj = build_legs_trajectory(poses, scale, resample_len, smooth_window)
    core_traj = build_core_trajectory(poses, scale, resample_len, smooth_window)

    return {
        "bar_path_trajectory": [round(float(v), 6) for v in bar_traj],
        "arm_trajectory": [round(float(v), 6) for v in arm_traj],
        "legs_trajectory": [round(float(v), 6) for v in legs_traj],
        "core_trajectory": [round(float(v), 6) for v in core_traj],
        "_scale": float(scale),
        "_midline_x": float(midline_x),
    }
