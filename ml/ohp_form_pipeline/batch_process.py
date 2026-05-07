"""
batch_process.py - SAM2+YOLO-only batch processing and wave analysis.

Processes all videos recursively and saves per-video:
  - sam2_overlay.mp4
  - masks/frame_XXXXXX.npy
  - analysis.json
  - plots/dashboard.png
  - plots/harmonic_wave.png
  - report.txt

Usage:
  python batch_process.py \
      --input_dir "/scratch/jnolas77/fitness/capstone/First 500 Vids" \
      --output_dir "/scratch/jnolas77/fitness/capstone/ml/ohp_form_pipeline/batch_outputs"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)
os.chdir(PIPELINE_ROOT)

from src.cv.pose_estimator import run_video as run_sam2_yolo_video
from src.io.json_writer import build_empty_artifact, _to_serializable
from src.io.video_loader import load_video_meta
from src.reasoning.feedback_generator import VLMFeedbackGenerator
from src.reasoning.rule_engine import load_rules, select_rules, format_coaching_feedback
from src.signals.segmentation import segment_phases, phases_to_dicts
from src.signals.smoothing import lowpass_smooth, savgol_smooth
from src.signals.wave_analysis import compute_wave_features
from src.unsupervised.cluster_naming import assign_clip_fault_flags
from src.viz.signal_plots import plot_signal_dashboard, plot_harmonic_wave_patterns


def _discover_videos(input_dir: str) -> list[str]:
    patterns = ["**/*.mp4", "**/*.MP4", "**/*.avi", "**/*.AVI", "**/*.mov", "**/*.MOV", "**/*.mkv", "**/*.MKV"]
    found: list[str] = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(input_dir, pat), recursive=True))
    found = sorted(set(found))
    print(f"Discovered {len(found)} video(s) in {input_dir}")
    return found


def _mask_centroid_y(mask: np.ndarray) -> float:
    ys, _ = np.where(mask.astype(bool))
    if ys.size == 0:
        return float("nan")
    return float(np.mean(ys))


def _mask_percentile_y(mask: np.ndarray, percentile: float) -> float:
    ys, _ = np.where(mask.astype(bool))
    if ys.size == 0:
        return float("nan")
    q = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(ys, q))


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
        out[i] = float(np.median(padded[i : i + w]))
    return out


def _build_body_wave_signal(
    masks_dir: str,
    fps: float,
    arm_percentile: float,
    torso_percentile: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "frame_*.npy")))
    if not mask_paths:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Arm proxy: higher mask region, torso proxy: central mask region.
    arm_raw = np.array(
        [_mask_percentile_y(np.load(p), percentile=arm_percentile) for p in mask_paths],
        dtype=np.float64,
    )
    torso_raw = np.array(
        [_mask_percentile_y(np.load(p), percentile=torso_percentile) for p in mask_paths],
        dtype=np.float64,
    )
    arm_raw = _fill_nans_1d(arm_raw)
    torso_raw = _fill_nans_1d(torso_raw)

    arm_med = _rolling_median(arm_raw, window=5)
    torso_med = _rolling_median(torso_raw, window=5)

    # Weighted blend emphasizes arms while preserving torso stability cues.
    raw = 0.65 * arm_med + 0.35 * torso_med

    med = _rolling_median(raw, window=5)
    low = lowpass_smooth(med, fps=fps, cutoff_hz=cutoff_hz)
    filt = savgol_smooth(low, window=11, poly=3)
    return arm_raw.astype(float), torso_raw.astype(float), filt.astype(float)


def _load_bar_y_from_masks(masks_dir: str) -> np.ndarray:
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "frame_*.npy")))
    if not mask_paths:
        return np.array([], dtype=float)

    y_vals = np.array([_mask_centroid_y(np.load(p)) for p in mask_paths], dtype=float)

    # Fill NaN gaps by interpolation so phase segmentation remains stable.
    if np.isnan(y_vals).any():
        idx = np.arange(len(y_vals))
        valid = ~np.isnan(y_vals)
        if np.count_nonzero(valid) >= 2:
            y_vals[~valid] = np.interp(idx[~valid], idx[valid], y_vals[valid])
        elif np.count_nonzero(valid) == 1:
            y_vals[:] = y_vals[valid][0]
        else:
            y_vals[:] = 0.0

    return y_vals


def _write_report(path: str, video_name: str, wave_features: dict, phase_count: int) -> None:
    quality = wave_features.get("quality", {})
    lines = [
        f"Video: {video_name}",
        f"Grade: {quality.get('grade', '?')}",
        f"Overall: {quality.get('overall', 0.0):.4f}",
        f"Smoothness: {quality.get('smoothness', 0.0):.4f}",
        f"Control: {quality.get('control', 0.0):.4f}",
        f"Efficiency: {quality.get('efficiency', 0.0):.4f}",
        f"Consistency: {quality.get('consistency', 0.0):.4f}",
        f"Wave segments: {phase_count}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _extract_rule_features(arm_y_raw: np.ndarray, torso_y_raw: np.ndarray, body_wave_y: np.ndarray) -> dict:
    """Build threshold-compatible feature dict from mask-derived body signals."""
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
        # Not observable from mask-only body-wave extraction.
        "wrist_height_diff_at_lockout_normalized": 0.0,
        "bar_tilt_std_deg": 0.0,
        "bar_lateral_drift_normalized": 0.0,
        "lockout_delay_sec": 0.0,
        # Proxies derived from torso movement signal.
        "trunk_lateral_shift_normalized": float(torso_amp / body_amp),
        "trunk_shift_peak_normalized": float(torso_amp / body_amp),
        "hip_lateral_shift_normalized": float(0.7 * torso_amp / body_amp),
        # Oscillation and depth-drift proxies.
        "bar_lockout_oscillation": lockout_osc,
        "bar_depth_drift_normalized": arm_torso_drift,
    }


def process_single_video(video_path: str, output_root: str, args: argparse.Namespace) -> dict | None:
    t0 = time.time()
    vid_name = Path(video_path).stem
    vid_dir = os.path.join(output_root, vid_name)
    os.makedirs(vid_dir, exist_ok=True)

    analysis_path = os.path.join(vid_dir, "analysis.json")
    if os.path.exists(analysis_path):
        print(f"  [SKIP] {vid_name} - already processed")
        return None

    try:
        meta = load_video_meta(video_path)

        tracker_summary = run_sam2_yolo_video(
            video_path=video_path,
            checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            yolo_model=args.yolo_model,
            yolo_fallback_model=args.yolo_fallback_model,
            output_dir=vid_dir,
            prompt_x=None,
            prompt_y=None,
            yolo_every_n=args.yolo_every_n,
            prompt_ema_alpha=args.prompt_ema_alpha,
        )

        arm_y_raw, torso_y_raw, body_wave_y = _build_body_wave_signal(
            masks_dir=os.path.join(vid_dir, "masks"),
            fps=meta.fps,
            arm_percentile=args.arm_y_percentile,
            torso_percentile=args.torso_y_percentile,
            cutoff_hz=args.signal_cutoff_hz,
        )
        if len(body_wave_y) == 0:
            raise RuntimeError("No masks were produced by SAM2 tracker")

        phases = segment_phases(body_wave_y, meta.fps)
        wave_features = compute_wave_features(body_wave_y, meta.fps, phases, scale=1.0)

        artifact = build_empty_artifact(meta.video_id, video_path, meta.fps, len(body_wave_y))
        artifact["video"] = os.path.basename(video_path)
        artifact["phase_segments"] = phases_to_dicts(phases)
        artifact["wave_features"] = wave_features
        artifact["raw_signals"]["bar_center_y"] = [float(x) for x in body_wave_y.tolist()]
        artifact["raw_signals"]["bar_center_x"] = [0.0 for _ in range(len(body_wave_y))]
        artifact["raw_signals"]["bar_center_z_proxy"] = [
            float(x) for x in arm_y_raw.tolist()
        ]
        artifact["raw_signals"]["left_wrist_y"] = [float(x) for x in arm_y_raw.tolist()]
        artifact["raw_signals"]["right_wrist_y"] = [float(x) for x in arm_y_raw.tolist()]
        artifact["raw_signals"]["trunk_center_x"] = [float(x) for x in torso_y_raw.tolist()]
        artifact["signal_source"] = "sam2_mask_arm_torso_wave_filtered"
        artifact["tracker_summary"] = tracker_summary
        artifact["signal_processing"] = {
            "arm_y_percentile": float(args.arm_y_percentile),
            "torso_y_percentile": float(args.torso_y_percentile),
            "blend_weights": {"arm": 0.65, "torso": 0.35},
            "median_window": 5,
            "lowpass_cutoff_hz": float(args.signal_cutoff_hz),
            "savgol_window": 11,
            "savgol_poly": 3,
        }

        # ── Reasoning: fault flags + rule engine + feedback generator ──
        with open(args.thresholds_path, "r", encoding="utf-8") as f:
            thresholds_cfg = yaml.safe_load(f)
        features_for_rules = _extract_rule_features(arm_y_raw, torso_y_raw, body_wave_y)
        artifact["derived_rule_features"] = {k: float(v) for k, v in features_for_rules.items()}
        fault_flags = assign_clip_fault_flags(features_for_rules, thresholds_cfg)
        artifact["fault_flags"] = fault_flags

        rules = load_rules(args.rules_path)
        triggered = select_rules(fault_flags, rules)
        rule_feedback = format_coaching_feedback(
            triggered_rules=triggered,
            cluster_name=artifact.get("unsupervised", {}).get("consensus_cluster_name", "unknown"),
            wave_quality=wave_features.get("quality", {}),
            rules=rules,
            uncertainty=False,
        )
        vlm_cfg = {
            "enabled": bool(args.vlm_enabled),
            "model_id": args.vlm_model_id,
            "max_new_tokens": int(args.vlm_max_new_tokens),
            "temperature": float(args.vlm_temperature),
            "device": args.vlm_device,
            "lora_path": args.vlm_lora_path,
        }
        vlm_gen = VLMFeedbackGenerator(vlm_cfg)
        artifact["language"] = vlm_gen.generate(
            artifact,
            key_frames=None,
            rule_based_fallback=rule_feedback,
        )

        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(artifact), f, indent=2)

        plots_dir = os.path.join(vid_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        phase_dicts = phases_to_dicts(phases)
        plot_signal_dashboard(
            body_wave_y,
            meta.fps,
            phase_dicts,
            os.path.join(plots_dir, "dashboard.png"),
        )
        plot_harmonic_wave_patterns(
            body_wave_y,
            meta.fps,
            wave_features,
            os.path.join(plots_dir, "harmonic_wave.png"),
        )

        _write_report(
            os.path.join(vid_dir, "report.txt"),
            vid_name,
            wave_features,
            len(phase_dicts),
        )

        elapsed = time.time() - t0
        grade = wave_features.get("quality", {}).get("grade", "?")
        print(f"  [OK] {vid_name} - {elapsed:.1f}s - Grade: {grade}")
        return artifact

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERR] {vid_name} - {elapsed:.1f}s - {e}")
        traceback.print_exc()
        with open(os.path.join(vid_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"{e}\n\n{traceback.format_exc()}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process OHP videos using SAM2+YOLO only.")
    parser.add_argument(
        "--input_dir",
        default="/scratch/jnolas77/fitness/capstone/First 500 Vids",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PIPELINE_ROOT, "batch_outputs"),
    )
    parser.add_argument(
        "--checkpoint",
        default="/scratch/jnolas77/fitness/capstone/ml/models/sam2.1_hiera_large.pt",
    )
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--yolo_model",
        default="/scratch/jnolas77/fitness/capstone/ml/models/yolo26x-pose.pt",
    )
    parser.add_argument(
        "--yolo_fallback_model",
        default="/scratch/jnolas77/Question_Generator_Model/Question_Generator_Model/yolov8n.pt",
    )
    parser.add_argument("--yolo_every_n", type=int, default=2)
    parser.add_argument("--prompt_ema_alpha", type=float, default=0.45)
    parser.add_argument(
        "--arm_y_percentile",
        type=float,
        default=18.0,
        help="Mask y-percentile for arm-dominant motion proxy",
    )
    parser.add_argument(
        "--torso_y_percentile",
        type=float,
        default=55.0,
        help="Mask y-percentile for torso motion proxy",
    )
    parser.add_argument(
        "--signal_cutoff_hz",
        type=float,
        default=2.0,
        help="Low-pass cutoff for grading signal",
    )
    parser.add_argument(
        "--thresholds_path",
        default=os.path.join(PIPELINE_ROOT, "configs", "thresholds.yaml"),
        help="Path to fault-threshold YAML",
    )
    parser.add_argument(
        "--rules_path",
        default=os.path.join(PIPELINE_ROOT, "configs", "rules_ohp.yaml"),
        help="Path to coaching rules YAML",
    )
    parser.add_argument("--vlm_enabled", action="store_true", help="Enable VLM feedback generation")
    parser.add_argument("--vlm_model_id", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--vlm_device", default="cpu")
    parser.add_argument("--vlm_lora_path", default="")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=512)
    parser.add_argument("--vlm_temperature", type=float, default=0.3)
    parser.add_argument("--max_videos", type=int, default=0, help="Limit videos (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos = _discover_videos(args.input_dir)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    summary = {"total": len(videos), "success": 0, "failed": 0, "skipped": 0}
    t_start = time.time()

    for i, vpath in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] Processing {os.path.basename(vpath)} ...")
        result = process_single_video(vpath, args.output_dir, args)
        vid_name = Path(vpath).stem

        if result is None:
            err_file = os.path.join(args.output_dir, vid_name, "error.txt")
            analysis = os.path.join(args.output_dir, vid_name, "analysis.json")
            if os.path.exists(analysis):
                summary["skipped"] += 1
            elif os.path.exists(err_file):
                summary["failed"] += 1
            else:
                summary["skipped"] += 1
        else:
            summary["success"] += 1

    elapsed = time.time() - t_start
    summary["elapsed_sec"] = round(elapsed, 1)

    print("\n" + "=" * 60)
    print(
        f"Batch complete: {summary['success']} OK / {summary['failed']} failed / {summary['skipped']} skipped"
    )
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    with open(os.path.join(args.output_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
