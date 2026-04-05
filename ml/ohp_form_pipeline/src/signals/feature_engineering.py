"""Engineer biomechanical features from smoothed trajectories — back-view OHP."""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from .smoothing import savgol_smooth, compute_derivatives


def _safe(arr: np.ndarray, fn) -> float:
    valid = arr[~np.isnan(arr)]
    return float(fn(valid)) if len(valid) > 0 else 0.0


def spectral_features(signal: np.ndarray, fps: float) -> dict:
    n = len(signal)
    if n < 8:
        return {"dominant_hz": 0.0, "spectral_entropy": 0.0,
                "band_slow": 0.0, "band_medium": 0.0, "band_fast": 0.0, "band_harmonic": 0.0}
    freqs = rfftfreq(n, d=1.0 / fps)
    power = np.abs(rfft(signal - np.nanmean(signal))) ** 2
    total = power.sum() + 1e-10
    dominant_hz = float(freqs[np.argmax(power[1:]) + 1]) if len(power) > 1 else 0.0

    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum())

    slow = band(0.0, 1.0)
    medium = band(1.0, 3.0)
    fast = band(3.0, 8.0)
    harmonic = band(8.0, fps / 2)
    norm_power = power / total
    sp_entropy = float(-np.sum(norm_power[norm_power > 0] * np.log(norm_power[norm_power > 0] + 1e-10)))
    return {
        "dominant_hz": dominant_hz,
        "spectral_entropy": round(sp_entropy, 6),
        "band_slow": round(slow, 4),
        "band_medium": round(medium, 4),
        "band_fast": round(fast, 4),
        "band_harmonic": round(harmonic, 4),
    }


def smoothness_score(jerk: np.ndarray, velocity: np.ndarray) -> float:
    """SPARC-inspired: lower jerk cost → higher smoothness."""
    jerk_cost = float(np.nanmean(jerk ** 2))
    vel_scale = float(np.nanmax(np.abs(velocity)) + 1e-8)
    normalized = jerk_cost / (vel_scale ** 2 + 1e-8)
    return float(np.exp(-normalized * 0.001))


def count_oscillations(signal: np.ndarray) -> int:
    """Count zero crossings of velocity (direction reversals)."""
    vel = np.diff(signal)
    crossings = int(np.sum(np.diff(np.sign(vel)) != 0))
    return crossings


def compute_bar_features(
    bar_cx: np.ndarray, bar_cy: np.ndarray, bar_tilt: np.ndarray,
    fps: float, scale: float, midline_x: float
) -> dict:
    bar_cx_n = (bar_cx - midline_x) / (scale + 1e-8)
    derivs = compute_derivatives(bar_cy, fps)
    vel = derivs["velocity"]
    jerk = derivs["jerk"]

    lateral_drift = np.abs(bar_cx_n)
    path_len = float(np.nansum(np.sqrt(np.diff(bar_cx) ** 2 + np.diff(bar_cy) ** 2)))
    vertical_range = float(np.nanmax(bar_cy) - np.nanmin(bar_cy))
    straightness = vertical_range / (path_len + 1e-8)
    spec = spectral_features(bar_cy[~np.isnan(bar_cy)], fps)
    smooth = smoothness_score(jerk, vel)
    osc = count_oscillations(bar_cy[~np.isnan(bar_cy)])

    # Work estimates (proportional, not physical Joules)
    up_vel = vel[vel < 0]
    down_vel = vel[vel > 0]
    work_pos = float(np.sum(up_vel ** 2)) * 0.5
    work_neg = float(np.sum(down_vel ** 2)) * 0.5
    eff_pct = (work_pos / (work_pos + work_neg + 1e-8)) * 100.0
    peak_power = float(np.nanmax(np.abs(vel) ** 2))

    # Lockout stability: variance in final 20% of frames
    lock_idx = int(len(bar_cy) * 0.8)
    lock_var = float(np.nanstd(bar_cy[lock_idx:]))

    return {
        "bar_lateral_drift_normalized": round(_safe(lateral_drift, np.mean), 6),
        "bar_lateral_drift_max_normalized": round(_safe(lateral_drift, np.max), 6),
        "bar_tilt_std_deg": round(_safe(bar_tilt, np.std), 4),
        "bar_tilt_range_deg": round(_safe(bar_tilt, lambda x: x.max() - x.min()), 4),
        "bar_path_straightness": round(straightness, 4),
        "bar_path_curvature_mean": round(float(np.nanmean(np.abs(np.diff(bar_cx, 2)))), 6),
        "bar_lockout_oscillation": round(lock_var / (scale + 1e-8), 6),
        "oscillation_count": osc,
        "smoothness": round(smooth, 4),
        "work_positive": round(work_pos, 4),
        "work_negative": round(work_neg, 4),
        "efficiency_pct": round(eff_pct, 2),
        "peak_power_w": round(peak_power, 4),
        **{f"bar_{k}": v for k, v in spec.items()},
    }


def compute_bilateral_features(
    left_y: np.ndarray, right_y: np.ndarray,
    left_elbow: np.ndarray, right_elbow: np.ndarray,
    fps: float, scale: float,
) -> dict:
    """Bilateral symmetry features for arm trajectory."""
    height_diff = np.abs(left_y - right_y) / (scale + 1e-8)
    elbow_diff = np.abs(left_elbow - right_elbow)

    # Lockout detection: find frame of minimum left and right y (highest position)
    if not np.isnan(left_y).all() and not np.isnan(right_y).all():
        l_lockout = int(np.nanargmin(left_y))
        r_lockout = int(np.nanargmin(right_y))
        lockout_delay = abs(l_lockout - r_lockout) / fps
        h_diff_at_lockout = abs(
            left_y[l_lockout] - right_y[r_lockout]
        ) / (scale + 1e-8)
    else:
        lockout_delay = 0.0
        h_diff_at_lockout = 0.0

    sym_score = float(1.0 - np.clip(_safe(height_diff, np.mean), 0.0, 1.0))
    return {
        "wrist_height_diff_mean_normalized": round(_safe(height_diff, np.mean), 6),
        "wrist_height_diff_at_lockout_normalized": round(h_diff_at_lockout, 6),
        "lockout_delay_sec": round(lockout_delay, 4),
        "elbow_angle_diff_mean_deg": round(_safe(elbow_diff, np.mean), 4),
        "symmetry_score": round(sym_score, 4),
    }


def compute_trunk_features(
    shoulder_tilt: np.ndarray, hip_tilt: np.ndarray,
    trunk_cx: np.ndarray, scale: float,
) -> dict:
    trunk_lateral = np.abs(trunk_cx - np.nanmedian(trunk_cx)) / (scale + 1e-8)
    return {
        "shoulder_line_tilt_mean_deg": round(_safe(shoulder_tilt, np.mean), 4),
        "shoulder_line_tilt_std_deg": round(_safe(shoulder_tilt, np.std), 4),
        "hip_line_tilt_mean_deg": round(_safe(hip_tilt, np.mean), 4),
        "hip_line_tilt_std_deg": round(_safe(hip_tilt, np.std), 4),
        "trunk_lateral_shift_normalized": round(_safe(trunk_lateral, np.mean), 6),
        "trunk_shift_peak_normalized": round(_safe(trunk_lateral, np.max), 6),
    }


def compute_hip_features(hip_cx: np.ndarray, scale: float) -> dict:
    hip_shift = np.abs(hip_cx - np.nanmedian(hip_cx)) / (scale + 1e-8)
    return {
        "hip_lateral_shift_normalized": round(_safe(hip_shift, np.mean), 6),
        "hip_lateral_shift_max_normalized": round(_safe(hip_shift, np.max), 6),
    }


def compute_leg_features(
    left_knee: np.ndarray, right_knee: np.ndarray
) -> dict:
    knee_diff = np.abs(left_knee - right_knee)
    return {
        "left_knee_angle_mean_deg": round(_safe(left_knee, np.mean), 4),
        "right_knee_angle_mean_deg": round(_safe(right_knee, np.mean), 4),
        "knee_angle_diff_mean_deg": round(_safe(knee_diff, np.mean), 4),
        "knee_angle_diff_max_deg": round(_safe(knee_diff, np.max), 4),
    }


def assemble_feature_vector(features: dict) -> np.ndarray:
    """Flatten feature dict to a 1D numpy array for clustering."""
    vals = []
    for v in features.values():
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return np.array(vals, dtype=np.float32)
