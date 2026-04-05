"""Wave/phase-level analysis and quality scoring."""
from __future__ import annotations

import numpy as np
from .smoothing import savgol_smooth, compute_derivatives
from .segmentation import PhaseSegment


def analyze_wave(signal: np.ndarray, fps: float, phase_type: str) -> dict:
    """Compute wave-level metrics for a single phase segment."""
    if len(signal) < 3:
        return {"type": phase_type, "duration_sec": len(signal) / fps,
                "mean_velocity": 0.0, "smoothness": 0.0}
    smoothed = savgol_smooth(signal)
    derivs = compute_derivatives(smoothed, fps)
    vel = derivs["velocity"]
    jerk = derivs["jerk"]
    mean_vel = float(np.nanmean(np.abs(vel)))
    vel_scale = float(np.nanmax(np.abs(vel)) + 1e-8)
    jerk_cost = float(np.nanmean(jerk ** 2)) / (vel_scale ** 2 + 1e-8)
    smoothness = max(0.0, float(2.0 - jerk_cost * 0.1))
    return {
        "type": phase_type,
        "duration_sec": round(len(signal) / fps, 4),
        "mean_velocity": round(mean_vel, 6),
        "smoothness": round(smoothness, 4),
    }


def compute_wave_features(
    bar_cy: np.ndarray,
    fps: float,
    phases: list[PhaseSegment],
    scale: float,
) -> dict:
    """Compute full wave_features dict matching the JSON schema."""
    from .feature_engineering import spectral_features, smoothness_score, count_oscillations

    smoothed = savgol_smooth(bar_cy)
    derivs = compute_derivatives(smoothed, fps)
    vel = derivs["velocity"]
    jerk = derivs["jerk"]
    acc = derivs["acceleration"]

    # Energy
    up = vel[vel < 0]
    down = vel[vel > 0]
    work_pos = float(np.sum(up ** 2)) * 0.5
    work_neg = float(np.sum(down ** 2)) * 0.5
    eff_pct = work_pos / (work_pos + work_neg + 1e-8) * 100.0
    peak_power = float(np.nanmax(np.abs(vel) ** 2))

    # Damping proxy
    damping_ratio = min(1.0, float(np.std(acc) / (np.std(vel) + 1e-8)))

    # Spectral
    spec = spectral_features(bar_cy[~np.isnan(bar_cy)], fps)

    # Smoothness / oscillations
    smooth = smoothness_score(jerk, vel)
    osc = count_oscillations(bar_cy[~np.isnan(bar_cy)])
    is_harmonic = osc >= 2

    # Quality
    control = float(np.clip(1.0 - np.nanstd(acc) / (np.nanmax(np.abs(acc)) + 1e-8), 0.0, 1.0))
    consistency = float(np.clip(1.0 - np.nanstd(vel) / (np.nanmax(np.abs(vel)) + 1e-8), 0.0, 1.0))
    efficiency = round(eff_pct / 100.0, 4)
    overall = round(float(np.mean([smooth, control, efficiency, consistency])), 4)

    # Grade
    if overall >= 0.85:
        grade = "A"
    elif overall >= 0.70:
        grade = "B"
    elif overall >= 0.55:
        grade = "C"
    elif overall >= 0.40:
        grade = "D"
    else:
        grade = "F"

    # Per-phase waves
    waves = []
    for phase in phases:
        seg = bar_cy[phase.start_frame: phase.end_frame + 1]
        waves.append(analyze_wave(seg, fps, phase.phase_type))

    return {
        "quality": {
            "grade": grade,
            "overall": overall,
            "smoothness": round(smooth, 4),
            "control": round(control, 4),
            "efficiency": round(efficiency, 4),
            "consistency": round(consistency, 4),
        },
        "energy": {
            "work_positive": round(work_pos, 4),
            "work_negative": round(work_neg, 4),
            "efficiency_pct": round(eff_pct, 2),
            "peak_power_w": round(peak_power, 4),
        },
        "damping": {
            "ratio": round(damping_ratio, 4),
            "control_quality": "good" if damping_ratio < 0.3 else ("fair" if damping_ratio < 0.6 else "poor"),
        },
        "frequency": {
            "dominant_hz": spec["dominant_hz"],
            "band_power": {
                "slow": spec["band_slow"],
                "medium": spec["band_medium"],
                "fast": spec["band_fast"],
                "harmonic": spec["band_harmonic"],
            },
            "spectral_entropy": spec["spectral_entropy"],
        },
        "harmonic": {
            "oscillation_count": osc,
            "is_harmonic": is_harmonic,
        },
        "waves": waves,
        "wave_count": len(waves),
    }
