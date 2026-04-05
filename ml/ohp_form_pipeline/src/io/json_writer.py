"""Write structured JSON artifacts for each processed clip."""
from __future__ import annotations

import json
import os
from typing import Any


def _to_serializable(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def write_clip_json(artifact: dict, output_dir: str, video_id: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{video_id}.json")
    with open(path, "w") as f:
        json.dump(_to_serializable(artifact), f, indent=2)
    return path


def build_empty_artifact(video_id: str, video_path: str, fps: float, n_frames: int) -> dict:
    """Return a fully-specified artifact dict with all fields populated to safe defaults."""
    return {
        "video_id": video_id,
        "exercise": "ohp",
        "camera_position": "BACK",
        "expert": None,
        "video": os.path.basename(video_path),
        "fps": fps,
        "n_frames": n_frames,
        "duration_sec": round(n_frames / fps, 3),
        "trajectories": {
            "arm_trajectory": [],
            "legs_trajectory": [],
            "core_trajectory": [],
            "bar_path_trajectory": [],
        },
        "raw_signals": {
            "bar_center_x": [],
            "bar_center_y": [],
            "bar_center_z_proxy": [],
            "bar_tilt_deg": [],
            "left_wrist_y": [],
            "right_wrist_y": [],
            "left_elbow_angle_deg": [],
            "right_elbow_angle_deg": [],
            "shoulder_line_tilt_deg": [],
            "hip_line_tilt_deg": [],
            "trunk_center_x": [],
            "left_knee_angle_deg": [],
            "right_knee_angle_deg": [],
            "left_right_wrist_depth_diff": [],
            "bar_depth_relative_to_shoulder_plane": [],
        },
        "phase_segments": [],
        "wave_features": {
            "quality": {
                "overall": 0.0,
                "smoothness": 0.0,
                "control": 0.0,
                "efficiency": 0.0,
                "consistency": 0.0,
                "symmetry": 0.0,
            },
            "energy": {
                "work_positive": 0.0,
                "work_negative": 0.0,
                "efficiency_pct": 0.0,
                "peak_power_w": 0.0,
            },
            "damping": {"ratio": 0.0, "control_quality": "unknown"},
            "frequency": {
                "dominant_hz": 0.0,
                "band_power": {"slow": 0.0, "medium": 0.0, "fast": 0.0, "harmonic": 0.0},
                "spectral_entropy": 0.0,
            },
            "harmonic": {"oscillation_count": 0, "is_harmonic": False},
            "waves": [],
            "wave_count": 0,
        },
        "unsupervised": {
            "feature_cluster_id": -1,
            "feature_cluster_name": "unknown",
            "latent_cluster_id": -1,
            "latent_cluster_name": "unknown",
            "consensus_cluster_name": "unknown",
            "cluster_confidence": 0.0,
            "anomaly_score": 0.0,
            "nearest_cluster_distance": 0.0,
            "disagreement_score": 0.0,
            "latent_embedding": [],
        },
        "fault_flags": {
            "left_right_lockout_asymmetry": False,
            "bar_tilt_instability": False,
            "lateral_bar_drift": False,
            "uneven_press_timing": False,
            "compensatory_lateral_shift": False,
            "trunk_shift_under_load": False,
            "hip_shift_compensation": False,
            "unstable_lockout": False,
            "forward_bar_drift_depth_proxy": False,
        },
        "depth_features": {
            "depth_enabled": False,
            "bar_forward_drift_depth": 0.0,
            "bar_depth_asymmetry": 0.0,
            "torso_depth_shift": 0.0,
            "subject_depth_stability": 0.0,
        },
        "signal_source": "bar_path_trajectory",
        "language": {
            "summary": "",
            "coach_feedback": "",
            "reasoning_trace_short": "",
        },
    }
