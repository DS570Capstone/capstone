"""Generate per-clip HTML/text summary reports."""
from __future__ import annotations

import json
import os


def generate_text_report(artifact: dict, out_path: str) -> None:
    vid = artifact.get("video_id", "unknown")
    quality = artifact.get("wave_features", {}).get("quality", {})
    unsup = artifact.get("unsupervised", {})
    fault_flags = artifact.get("fault_flags", {})
    language = artifact.get("language", {})
    depth = artifact.get("depth_features", {})

    active_faults = [k for k, v in fault_flags.items() if v]

    lines = [
        "=" * 60,
        f"OHP FORM ANALYSIS REPORT — {vid}",
        "=" * 60,
        f"Exercise      : {artifact.get('exercise', 'ohp').upper()}",
        f"Camera        : {artifact.get('camera_position', 'BACK')}",
        f"Duration      : {artifact.get('duration_sec', 0):.2f}s  |  Frames: {artifact.get('n_frames', 0)}",
        f"FPS           : {artifact.get('fps', 30.0)}",
        "",
        "--- QUALITY ---",
        f"  Grade       : {quality.get('grade', '?')}",
        f"  Overall     : {quality.get('overall', 0):.3f}",
        f"  Smoothness  : {quality.get('smoothness', 0):.3f}",
        f"  Efficiency  : {quality.get('efficiency', 0):.3f}",
        f"  Symmetry    : {quality.get('symmetry', 0):.3f}",
        "",
        "--- CLUSTER ---",
        f"  Feature cluster   : {unsup.get('feature_cluster_name', 'unknown')}",
        f"  Latent cluster    : {unsup.get('latent_cluster_name', 'unknown')}",
        f"  Consensus         : {unsup.get('consensus_cluster_name', 'unknown')}",
        f"  Anomaly score     : {unsup.get('anomaly_score', 0):.3f}",
        f"  Disagreement      : {unsup.get('disagreement_score', 0):.3f}",
        "",
        "--- DEPTH ---",
        f"  Enabled           : {depth.get('depth_enabled', False)}",
        f"  Bar depth asym.   : {depth.get('bar_depth_asymmetry', 0):.4f}",
        f"  Torso depth shift : {depth.get('torso_depth_shift', 0):.4f}",
        "",
        "--- FAULT FLAGS ---",
    ]
    if active_faults:
        for f in active_faults:
            lines.append(f"  [X] {f}")
    else:
        lines.append("  No faults detected.")
    lines += [
        "",
        "--- COACHING FEEDBACK ---",
        f"  Summary      : {language.get('summary', '')}",
        f"  Feedback     : {language.get('coach_feedback', '')}",
        f"  Reasoning    : {language.get('reasoning_trace_short', '')}",
        "=" * 60,
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
