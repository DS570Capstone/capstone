"""
run_single_video.py — Process one OHP back-view .mp4.

Backends
--------
  yolo_keypoints  (default) — YOLO-pose only, no SAM2.
                               ~3-10 s on CPU with yolov8n-pose.pt.
  sam2_yolo       (GPU)     — SAM2 mask segmentation + YOLO prompt.
                               3-15 min on CPU; intended for GPU clusters.

Usage:
    python -m src.app.run_single_video --video path/to/clip.mp4 [--config configs/default.yaml]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml

# Ensure ohp_form_pipeline root is on sys.path regardless of invocation method
_PIPELINE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)


class VideoValidationError(Exception):
    """User-facing validation failure — shown directly in the UI."""


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Signal helpers ──────────────────────────────────────────────────────────

def _fill_nans_1d(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64).copy()
    nans = np.isnan(out)
    if not nans.any():
        return out
    if nans.all():
        out[:] = 0.0
        return out
    idx = np.arange(len(out))
    valid = ~nans
    out[nans] = np.interp(idx[nans], idx[valid], out[valid])
    return out


def _rolling_median(signal: np.ndarray, window: int = 5) -> np.ndarray:
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    out = np.empty_like(signal, dtype=np.float64)
    for i in range(len(signal)):
        out[i] = float(np.median(padded[i: i + w]))
    return out


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0 or len(signal) == target_len:
        return signal.astype(float)
    if len(signal) == 0:
        return np.zeros(target_len, dtype=float)
    x_old = np.linspace(0.0, 1.0, len(signal))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, signal.astype(float)).astype(float)


def _build_body_wave_from_arrays(
    arm_y_raw: np.ndarray,
    torso_y_raw: np.ndarray,
    fps: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Blend arm + torso Y signals → filtered body-wave signal."""
    from src.signals.smoothing import lowpass_smooth, savgol_smooth

    arm_filled   = _fill_nans_1d(arm_y_raw)
    torso_filled = _fill_nans_1d(torso_y_raw)

    arm_med   = _rolling_median(arm_filled,   window=5)
    torso_med = _rolling_median(torso_filled, window=5)

    raw = 0.65 * arm_med + 0.35 * torso_med
    med = _rolling_median(raw, window=5)
    low = lowpass_smooth(med, fps=fps, cutoff_hz=cutoff_hz)
    filt = savgol_smooth(low, window=11, poly=3)
    return filt.astype(float)


def _build_ui_trajectories(
    body_wave_y: np.ndarray,
    arm_y_raw: np.ndarray,
    torso_y_raw: np.ndarray,
) -> dict:
    """Build chart trajectories from tracked 1D signals for the dashboard UI."""
    n = len(body_wave_y)
    if n == 0:
        return {
            "bar_path_trajectory": [],
            "arm_trajectory": [],
            "legs_trajectory": [],
            "core_trajectory": [],
        }

    # Center and scale so curves are visually comparable on a shared axis.
    bar = np.asarray(body_wave_y, dtype=np.float64)
    bar = bar - np.nanmean(bar)
    bar_scale = float(np.nanstd(bar) + 1e-8)
    bar_traj = (bar / bar_scale).astype(float)

    arm = _fill_nans_1d(np.asarray(arm_y_raw, dtype=np.float64))
    arm = arm - np.nanmean(arm)
    arm_scale = float(np.nanstd(arm) + 1e-8)
    arm_traj = (arm / arm_scale).astype(float)

    torso = _fill_nans_1d(np.asarray(torso_y_raw, dtype=np.float64))
    torso = torso - np.nanmedian(torso)
    torso_scale = float(np.nanstd(torso) + 1e-8)
    core_traj = (torso / torso_scale).astype(float)

    # Legs are unavailable in yolo_keypoints fast path; keep a safe zero baseline.
    legs_traj = np.zeros(n, dtype=float)

    return {
        "bar_path_trajectory": [float(x) for x in bar_traj.tolist()],
        "arm_trajectory": [float(x) for x in arm_traj.tolist()],
        "legs_trajectory": [float(x) for x in legs_traj.tolist()],
        "core_trajectory": [float(x) for x in core_traj.tolist()],
    }


# ── SAM2 mask-based helpers (kept for sam2_yolo backend) ────────────────────

def _mask_percentile_y(mask: np.ndarray, percentile: float) -> float:
    import glob as _glob
    ys, _ = np.where(mask.astype(bool))
    if ys.size == 0:
        return float("nan")
    return float(np.percentile(ys, float(np.clip(percentile, 0.0, 100.0))))


def _build_body_wave_from_masks(
    masks_dir: str,
    fps: float,
    arm_percentile: float,
    torso_percentile: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import glob as _glob
    mask_paths = sorted(_glob.glob(os.path.join(masks_dir, "frame_*.npy")))
    if not mask_paths:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    arm_raw = np.array(
        [_mask_percentile_y(np.load(p), arm_percentile) for p in mask_paths],
        dtype=np.float64,
    )
    torso_raw = np.array(
        [_mask_percentile_y(np.load(p), torso_percentile) for p in mask_paths],
        dtype=np.float64,
    )
    filt = _build_body_wave_from_arrays(arm_raw, torso_raw, fps, cutoff_hz)
    return arm_raw.astype(float), torso_raw.astype(float), filt


def _extract_rule_features(
    arm_y_raw: np.ndarray,
    torso_y_raw: np.ndarray,
    body_wave_y: np.ndarray,
) -> dict:
    from src.signals.smoothing import savgol_smooth

    eps = 1e-8
    if len(body_wave_y) == 0:
        return {}

    body_amp = float(np.ptp(body_wave_y)) + eps
    torso_amp = float(np.ptp(torso_y_raw))
    arm_torso_delta = np.asarray(arm_y_raw, dtype=float) - np.asarray(torso_y_raw, dtype=float)
    arm_torso_drift = float(np.ptp(arm_torso_delta)) / body_amp

    smooth = savgol_smooth(np.asarray(body_wave_y, dtype=float), window=11, poly=3)
    residual = np.asarray(body_wave_y, dtype=float) - np.asarray(smooth, dtype=float)
    lockout_osc = float(np.std(residual)) / body_amp

    return {
        "wrist_height_diff_at_lockout_normalized": 0.0,
        "bar_tilt_std_deg": 0.0,
        "bar_lateral_drift_normalized": 0.0,
        "lockout_delay_sec": 0.0,
        "trunk_lateral_shift_normalized": float(torso_amp / body_amp),
        "trunk_shift_peak_normalized": float(torso_amp / body_amp),
        "hip_lateral_shift_normalized": float(0.7 * torso_amp / body_amp),
        "bar_lockout_oscillation": lockout_osc,
        "bar_depth_drift_normalized": arm_torso_drift,
    }


# ── Main entry point ────────────────────────────────────────────────────────

def run(video_path: str, config_path: str, output_dir: str = None, on_progress=None) -> dict:
    from src.io.json_writer import build_empty_artifact
    from src.io.video_loader import load_video_meta
    from src.reasoning.feedback_generator import VLMFeedbackGenerator
    from src.reasoning.rule_engine import load_rules, select_rules, format_coaching_feedback
    from src.signals.segmentation import segment_phases, phases_to_dicts
    from src.signals.wave_analysis import compute_wave_features
    from src.unsupervised.cluster_naming import assign_clip_fault_flags

    def _progress(stage: str, pct: int) -> None:
        if on_progress:
            on_progress(stage, pct)

    t0 = time.time()
    cfg = load_config(config_path)
    out_dir = output_dir or os.path.join(_PIPELINE_ROOT, cfg["pipeline"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    tracker_cfg = cfg.get("tracker", {})
    backend = tracker_cfg.get("backend", "yolo_keypoints")

    # ── Stage 1: Validate video ──────────────────────────────────────────────
    _progress("Loading video", 10)
    print(f"[1/5] Loading video: {video_path}")
    meta = load_video_meta(video_path)
    print(f"      {meta.n_frames} frames @ {meta.fps:.1f} fps  ({meta.duration_sec:.2f}s)")

    _MAX_DURATION_SEC = 90.0
    if meta.duration_sec > _MAX_DURATION_SEC:
        raise VideoValidationError(
            f"Video is {meta.duration_sec:.0f}s long. "
            f"Maximum supported duration is {_MAX_DURATION_SEC:.0f}s. "
            "Please trim your video to a single OHP set before uploading."
        )

    # ── Stage 2: Tracking ────────────────────────────────────────────────────
    if backend == "yolo_keypoints":
        arm_y_raw, torso_y_raw, tracker_summary = _run_yolo_keypoints(
            video_path, meta, cfg, out_dir, _progress
        )
    else:
        arm_y_raw, torso_y_raw, tracker_summary = _run_sam2_yolo(
            video_path, meta, cfg, out_dir, _progress
        )

    # ── Stage 3: Body-wave signal ────────────────────────────────────────────
    _progress("Extracting body-wave signal", 72)
    print("[3/5] Building body-wave signal ...")
    cutoff_hz = float(tracker_cfg.get("signal_cutoff_hz", 2.0))
    body_wave_y = _build_body_wave_from_arrays(arm_y_raw, torso_y_raw, meta.fps, cutoff_hz)
    target_frames = int(cfg.get("pipeline", {}).get("max_frames", 0) or 0)
    if target_frames > 0 and len(body_wave_y) > 0:
        arm_y_raw = _resample_1d(arm_y_raw, target_frames)
        torso_y_raw = _resample_1d(torso_y_raw, target_frames)
        body_wave_y = _resample_1d(body_wave_y, target_frames)

    if len(body_wave_y) < 6:
        raise VideoValidationError(
            "Not enough frames were tracked. "
            "Make sure the person is fully visible throughout the lift."
        )

    # ── Stage 4: Phase segmentation + wave features ──────────────────────────
    _progress("Segmenting phases", 80)
    print("[4/5] Segmenting phases and computing wave features ...")
    phases = segment_phases(body_wave_y, meta.fps)
    wave_features = compute_wave_features(body_wave_y, meta.fps, phases, scale=1.0)

    artifact = build_empty_artifact(meta.video_id, video_path, meta.fps, len(body_wave_y))
    artifact["trajectories"] = _build_ui_trajectories(body_wave_y, arm_y_raw, torso_y_raw)
    artifact["phase_segments"] = phases_to_dicts(phases)
    artifact["wave_features"] = wave_features
    artifact["raw_signals"]["bar_center_y"] = [float(x) for x in body_wave_y.tolist()]
    artifact["raw_signals"]["bar_center_x"] = [0.0] * len(body_wave_y)
    artifact["raw_signals"]["bar_center_z_proxy"] = [float(x) for x in arm_y_raw.tolist()]
    artifact["raw_signals"]["left_wrist_y"] = [float(x) for x in arm_y_raw.tolist()]
    artifact["raw_signals"]["right_wrist_y"] = [float(x) for x in arm_y_raw.tolist()]
    artifact["raw_signals"]["trunk_center_x"] = [float(x) for x in torso_y_raw.tolist()]
    artifact["signal_source"] = f"{backend}_wave_filtered"
    artifact["tracker_summary"] = tracker_summary
    artifact["signal_processing"] = {
        "backend": backend,
        "max_height": int(cfg.get("pipeline", {}).get("max_height", 720)),
        "max_frames": int(cfg.get("pipeline", {}).get("max_frames", len(body_wave_y))),
        "frame_step": int(cfg.get("pipeline", {}).get("frame_step", 1)),
        "resample_length": int(cfg.get("signals", {}).get("resample_length", len(body_wave_y))),
        "yolo_every_n": int(tracker_cfg.get("yolo_every_n", 1)),
        "yolo_frame_step": int(tracker_cfg.get("yolo_frame_step", 1)),
        "blend_weights": {"arm": 0.65, "torso": 0.35},
        "median_window": 5,
        "lowpass_cutoff_hz": cutoff_hz,
        "savgol_window": 11,
        "savgol_poly": 3,
    }

    # ── Stage 5: Rules engine + coaching feedback ────────────────────────────
    _progress("Generating coaching feedback", 86)
    print("[5/5] Generating coaching feedback ...")
    thresholds_path = os.path.join(_PIPELINE_ROOT, "configs", "thresholds.yaml")
    rules_path = os.path.join(_PIPELINE_ROOT, "configs", "rules_ohp.yaml")

    features_for_rules = _extract_rule_features(arm_y_raw, torso_y_raw, body_wave_y)
    artifact["derived_rule_features"] = {k: float(v) for k, v in features_for_rules.items()}

    fault_flags: dict = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds_cfg = yaml.safe_load(f)
        fault_flags = assign_clip_fault_flags(features_for_rules, thresholds_cfg)
    artifact["fault_flags"] = fault_flags

    rule_feedback = "Analysis complete."
    if os.path.exists(rules_path):
        rules = load_rules(rules_path)
        triggered = select_rules(fault_flags, rules)
        rule_feedback = format_coaching_feedback(
            triggered_rules=triggered,
            cluster_name=artifact.get("unsupervised", {}).get("consensus_cluster_name", "unknown"),
            wave_quality=wave_features.get("quality", {}),
            rules=rules,
            uncertainty=False,
        )

    vlm_cfg = cfg.get("vlm", {})
    vlm_gen = VLMFeedbackGenerator(vlm_cfg)
    fallback_language = {
        "summary": rule_feedback[:200] if len(rule_feedback) > 200 else rule_feedback,
        "coach_feedback": rule_feedback,
        "reasoning_trace_short": "rule-based fallback",
    }
    artifact["language"] = vlm_gen.generate(
        artifact,
        key_frames=None,
        rule_based_fallback=fallback_language,
    )

    elapsed = time.time() - t0
    grade = wave_features.get("quality", {}).get("grade", "?")
    print(f"      Done in {elapsed:.1f}s — Grade: {grade}")
    return artifact


# ── Backend implementations ──────────────────────────────────────────────────

def _run_yolo_keypoints(video_path, meta, cfg, out_dir, _progress):
    """YOLO-pose keypoint tracker — CPU-friendly, no SAM2."""
    from src.cv.yolo_keypoint_tracker import run_yolo_keypoints

    tracker_cfg = cfg.get("tracker", {})
    yolo_model = tracker_cfg.get("yolo_model", "")

    if not os.path.exists(yolo_model):
        raise RuntimeError(
            f"YOLO model not found: {yolo_model}\n"
            "Set tracker.yolo_model in configs/default.yaml\n"
            "Download: python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt')\""
        )

    _progress("Starting YOLO pose tracker", 15)
    print(f"[2/5] Running YOLO keypoint tracker ({os.path.basename(yolo_model)}) ...")

    def _frame_cb(frame_idx: int, total_frames: int) -> None:
        if total_frames > 0:
            pct = 15 + int(55 * frame_idx / total_frames)
            _progress(f"Tracking frame {frame_idx + 1}/{total_frames}", pct)

    arm_y_raw, torso_y_raw, summary = run_yolo_keypoints(
        video_path=video_path,
        yolo_model=yolo_model,
        output_dir=out_dir,
        yolo_every_n=int(tracker_cfg.get("yolo_every_n", 1)),
        on_frame_progress=_frame_cb,
        frame_step=int(tracker_cfg.get("yolo_frame_step", 1)),
        max_height=int(cfg.get("pipeline", {}).get("max_height", 720)),
    )
    return arm_y_raw, torso_y_raw, summary


def _run_sam2_yolo(video_path, meta, cfg, out_dir, _progress):
    """SAM2 + YOLO tracker — best accuracy but slow on CPU."""
    from src.cv.pose_estimator import run_video as run_sam2_yolo_video

    tracker_cfg = cfg.get("tracker", {})
    sam2_checkpoint = tracker_cfg.get("sam2_checkpoint", "")
    sam2_model_cfg  = tracker_cfg.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
    yolo_model      = tracker_cfg.get("yolo_model", "")

    if not os.path.exists(sam2_checkpoint):
        raise RuntimeError(
            f"SAM2 checkpoint not found: {sam2_checkpoint}\n"
            "Set tracker.sam2_checkpoint in configs/default.yaml"
        )
    if not os.path.exists(yolo_model):
        raise RuntimeError(
            f"YOLO model not found: {yolo_model}\n"
            "Set tracker.yolo_model in configs/default.yaml"
        )

    _progress("Starting SAM2+YOLO tracker", 15)
    print("[2/5] Running SAM2+YOLO tracker ...")

    def _frame_cb(frame_idx: int, total_frames: int) -> None:
        if total_frames > 0:
            pct = 15 + int(55 * frame_idx / total_frames)
            _progress(f"Tracking frame {frame_idx + 1}/{total_frames}", pct)

    tracker_summary = run_sam2_yolo_video(
        video_path=video_path,
        checkpoint=sam2_checkpoint,
        model_cfg=sam2_model_cfg,
        yolo_model=yolo_model,
        yolo_fallback_model=None,
        output_dir=out_dir,
        prompt_x=None,
        prompt_y=None,
        yolo_every_n=int(tracker_cfg.get("yolo_every_n", 2)),
        prompt_ema_alpha=float(tracker_cfg.get("prompt_ema_alpha", 0.45)),
        on_frame_progress=_frame_cb,
        frame_step=int(tracker_cfg.get("sam2_frame_step", 1)),
    )

    masks_dir = os.path.join(out_dir, "masks")
    arm_percentile   = float(tracker_cfg.get("arm_y_percentile",   18.0))
    torso_percentile = float(tracker_cfg.get("torso_y_percentile", 55.0))
    cutoff_hz        = float(tracker_cfg.get("signal_cutoff_hz",    2.0))

    arm_y_raw, torso_y_raw, _ = _build_body_wave_from_masks(
        masks_dir=masks_dir,
        fps=meta.fps,
        arm_percentile=arm_percentile,
        torso_percentile=torso_percentile,
        cutoff_hz=cutoff_hz,
    )
    return arm_y_raw, torso_y_raw, tracker_summary
