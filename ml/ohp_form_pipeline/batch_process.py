"""
batch_process.py — Process all videos in a directory tree end-to-end.

Creates per-video output folders, each containing:
  - annotated_video.mp4   (skeleton + bar overlay)
  - depth_maps/           (per-frame .npy + colourised .jpg previews)
  - analysis.json         (full 80+ metric artifact)
  - keypoints.json        (per-frame MediaPipe keypoints)
  - report.txt            (text coaching report)
  - plots/                (trajectory, bilateral, harmonic, dashboard PNGs)

Usage:
    python batch_process.py \
        --input_dir  "C:/Users/josep/Desktop/Capstone_new/videos/First 500 Vids" \
        --output_dir "C:/Users/josep/Desktop/Capstone_new/ml/ohp_form_pipeline/batch_outputs" \
        --config     configs/default.yaml \
        --workers 1
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

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)
os.chdir(PIPELINE_ROOT)

import cv2
import numpy as np
import yaml

from src.io.video_loader import load_video_meta, load_all_frames
from src.io.json_writer import build_empty_artifact, write_clip_json, _to_serializable
from src.cv.pose_estimator import (
    PoseEstimator, PoseResult, extract_angles_from_pose, KP, KEYPOINT_NAMES,
)
from src.cv.bar_detector import BarDetector
from src.cv.tracker import smooth_poses, smooth_bar_detections
from src.cv.depth_estimator import DepthEstimator
from src.signals.normalization import compute_scale, compute_midline_x
from src.signals.smoothing import savgol_smooth
from src.signals.segmentation import segment_phases, phases_to_dicts
from src.signals.trajectory_builder import build_all_trajectories
from src.signals.feature_engineering import (
    compute_bar_features, compute_bilateral_features,
    compute_trunk_features, compute_hip_features, compute_leg_features,
)
from src.signals.wave_analysis import compute_wave_features
from src.unsupervised.cluster_naming import assign_clip_fault_flags
from src.reasoning.rule_engine import load_rules, select_rules, format_coaching_feedback
from src.viz.annotated_video import write_annotated_video
from src.viz.signal_plots import (
    plot_trajectories, plot_signal_dashboard,
    plot_bilateral_symmetry, plot_harmonic_wave_patterns,
)
from src.viz.report_generator import generate_text_report


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _discover_videos(input_dir: str) -> list[str]:
    """Recursively find all .mp4 files."""
    patterns = ["**/*.mp4", "**/*.MP4", "**/*.avi", "**/*.mov"]
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(input_dir, pat), recursive=True))
    found = sorted(set(found))
    print(f"Discovered {len(found)} video(s) in {input_dir}")
    return found


def _save_depth_maps(depths: list[np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, depth in enumerate(depths):
        np.save(os.path.join(out_dir, f"frame_{i:06d}.npy"), depth)
        preview = cv2.applyColorMap(
            (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO,
        )
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.jpg"), preview)


def _save_keypoints_json(poses: list[PoseResult], out_path: str) -> None:
    rows = []
    for p in poses:
        frame_kps = {}
        for idx, name in enumerate(KEYPOINT_NAMES):
            x, y = p.keypoints[idx]
            frame_kps[name] = {
                "x": None if np.isnan(x) else round(float(x), 4),
                "y": None if np.isnan(y) else round(float(y), 4),
                "confidence": round(float(p.confidences[idx]), 4),
                "visible": bool(p.visible[idx]),
            }
        rows.append({"frame_idx": int(p.frame_idx), "keypoints": frame_kps})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def process_single_video(
    video_path: str,
    output_root: str,
    cfg: dict,
    depth_est: DepthEstimator | None = None,
) -> dict | None:
    """Process one video into its own output folder."""
    t0 = time.time()
    vid_name = Path(video_path).stem
    vid_dir = os.path.join(output_root, vid_name)
    os.makedirs(vid_dir, exist_ok=True)

    # Check if already processed
    json_path = os.path.join(vid_dir, "analysis.json")
    if os.path.exists(json_path):
        print(f"  [SKIP] {vid_name} — already processed")
        return None

    try:
        # Stage 1: Load video
        meta = load_video_meta(video_path)
        frames, _ = load_all_frames(
            video_path,
            max_height=cfg["pipeline"]["max_height"],
            frame_step=cfg["pipeline"]["frame_step"],
        )
        if not frames:
            print(f"  [WARN] {vid_name} — no frames loaded, skipping")
            return None

        artifact = build_empty_artifact(meta.video_id, video_path, meta.fps, len(frames))

        # Stage 2: Pose estimation
        pose_cfg = cfg["pose"]
        estimator = PoseEstimator(
            backend=pose_cfg["backend"],
            confidence_threshold=pose_cfg["confidence_threshold"],
        )
        poses_raw = estimator.process_video(frames)
        estimator.close()

        # Smooth poses
        poses = smooth_poses(
            poses_raw,
            window=pose_cfg["smooth_window"],
            poly=pose_cfg["smooth_poly"],
            method=pose_cfg.get("smooth_method", "savgol"),
        )

        # Bar detection
        bar_cfg = cfg["bar"]
        bar_detector = BarDetector(
            backend=bar_cfg["backend"],
            wrist_span_scale=bar_cfg["wrist_span_scale"],
            min_confidence=bar_cfg["min_confidence"],
        )
        bars = smooth_bar_detections(
            bar_detector.detect_sequence(frames, poses_raw),
            window=pose_cfg["smooth_window"],
            poly=pose_cfg["smooth_poly"],
        )

        # Save keypoints
        _save_keypoints_json(poses, os.path.join(vid_dir, "keypoints.json"))

        # Stage 3: Depth estimation
        depths = []
        depth_feats = {"depth_enabled": False}
        depth_map_dir = os.path.join(vid_dir, "depth_maps")
        if cfg["depth"]["enabled"] and depth_est is not None:
            depths = depth_est.process_video(frames, video_id=vid_name)
            _save_depth_maps(depths, depth_map_dir)
            df_raw = depth_est.extract_depth_features(depths, poses, bars)
            depth_feats = {
                "depth_enabled": True,
                "bar_forward_drift_depth": df_raw.get("bar_forward_drift_depth", 0.0),
                "bar_depth_asymmetry": df_raw.get("bar_depth_asymmetry", 0.0),
                "torso_depth_shift": df_raw.get("torso_depth_shift", 0.0),
                "subject_depth_stability": df_raw.get("subject_depth_stability", 0.0),
            }
        artifact["depth_features"] = depth_feats

        # Stage 4: Trajectories
        scale = compute_scale(poses)
        midline_x = compute_midline_x(poses)
        resample_len = cfg["pipeline"]["max_frames"]
        traj_dict = build_all_trajectories(
            poses, bars, resample_len=resample_len,
            smooth_window=cfg["signals"]["smooth_window"],
        )
        artifact["trajectories"] = {
            k: v for k, v in traj_dict.items() if not k.startswith("_")
        }

        # Raw signals
        bar_cx = np.array([b.center_x for b in bars])
        bar_cy = np.array([b.center_y for b in bars])
        bar_tilt = np.array([b.tilt_deg for b in bars])
        angle_seq = [extract_angles_from_pose(p) for p in poses]
        left_elbow_deg = np.array([a.get("left_elbow_angle_deg", np.nan) for a in angle_seq])
        right_elbow_deg = np.array([a.get("right_elbow_angle_deg", np.nan) for a in angle_seq])
        shoulder_tilt = np.array([a.get("shoulder_line_tilt_deg", np.nan) for a in angle_seq])
        hip_tilt_arr = np.array([a.get("hip_line_tilt_deg", np.nan) for a in angle_seq])

        def kp_y(name):
            return np.array([p.keypoints[KP[name], 1] for p in poses])
        def kp_x(name):
            return np.array([p.keypoints[KP[name], 0] for p in poses])

        ls_x, rs_x = kp_x("left_shoulder"), kp_x("right_shoulder")
        lh_x, rh_x = kp_x("left_hip"), kp_x("right_hip")
        trunk_cx_arr = np.nanmean(np.stack([(ls_x + rs_x) / 2.0, (lh_x + rh_x) / 2.0]), axis=0)
        hip_cx_arr = (lh_x + rh_x) / 2.0
        lk_deg = np.array([a.get("left_knee_angle_deg", np.nan) for a in angle_seq])
        rk_deg = np.array([a.get("right_knee_angle_deg", np.nan) for a in angle_seq])

        # Stage 5: Segmentation
        bar_cy_smooth = savgol_smooth(bar_cy, cfg["signals"]["smooth_window"])
        phases = segment_phases(bar_cy_smooth, meta.fps)
        artifact["phase_segments"] = phases_to_dicts(phases)
        phase_per_frame = ["unknown"] * len(frames)
        for ph in phases:
            for fi in range(ph.start_frame, min(ph.end_frame + 1, len(frames))):
                phase_per_frame[fi] = ph.phase_type

        # Stage 6: Features + wave analysis
        bar_feats = compute_bar_features(bar_cx, bar_cy, bar_tilt, meta.fps, scale, midline_x)
        bilateral_feats = compute_bilateral_features(
            kp_y("left_wrist"), kp_y("right_wrist"),
            left_elbow_deg, right_elbow_deg, meta.fps, scale,
        )
        trunk_feats = compute_trunk_features(shoulder_tilt, hip_tilt_arr, trunk_cx_arr, scale)
        hip_feats = compute_hip_features(hip_cx_arr, scale)
        leg_feats = compute_leg_features(lk_deg, rk_deg)
        all_features = {**bar_feats, **bilateral_feats, **trunk_feats, **hip_feats, **leg_feats}

        wave_feats = compute_wave_features(bar_cy_smooth, meta.fps, phases, scale)
        wave_feats["quality"]["symmetry"] = bilateral_feats.get("symmetry_score", 0.0)
        artifact["wave_features"] = wave_feats

        # Fault flags
        thresholds_path = os.path.join(PIPELINE_ROOT, "configs", "thresholds.yaml")
        with open(thresholds_path) as f:
            thresholds_cfg = yaml.safe_load(f)
        fault_flags = assign_clip_fault_flags(all_features, thresholds_cfg)
        artifact["fault_flags"] = fault_flags

        # Stage 7: Rule-based feedback
        rules_path = os.path.join(PIPELINE_ROOT, "configs", "rules_ohp.yaml")
        rules = load_rules(rules_path)
        triggered = select_rules(fault_flags, rules)
        rule_feedback = format_coaching_feedback(
            triggered, "unknown", wave_feats["quality"], rules,
            uncertainty=depth_feats.get("depth_enabled", False),
        )
        artifact["language"] = rule_feedback

        # Stage 8: Save outputs
        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(artifact), f, indent=2)

        # Annotated video
        annotated_path = os.path.join(vid_dir, "annotated_video.mp4")
        write_annotated_video(
            frames, poses, bars, fault_flags, phase_per_frame,
            annotated_path, meta.fps,
        )

        # Plots
        plots_dir = os.path.join(vid_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        try:
            plot_trajectories(
                artifact["trajectories"], artifact["phase_segments"],
                os.path.join(plots_dir, "trajectories.png"), meta.fps,
            )
            plot_signal_dashboard(
                bar_cy_smooth, meta.fps, artifact["phase_segments"],
                os.path.join(plots_dir, "dashboard.png"),
            )
            plot_bilateral_symmetry(
                kp_y("left_wrist"), kp_y("right_wrist"),
                left_elbow_deg, right_elbow_deg,
                os.path.join(plots_dir, "bilateral.png"),
            )
            plot_harmonic_wave_patterns(
                bar_cy, meta.fps, wave_feats,
                os.path.join(plots_dir, "harmonic_wave.png"),
            )
        except Exception:
            pass  # plots are non-critical

        # Report
        generate_text_report(artifact, os.path.join(vid_dir, "report.txt"))

        elapsed = time.time() - t0
        grade = wave_feats["quality"].get("grade", "?")
        active_faults = [k for k, v in fault_flags.items() if v]
        print(f"  [OK] {vid_name} — {elapsed:.1f}s — Grade: {grade} — Faults: {active_faults}")
        return artifact

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERR] {vid_name} — {elapsed:.1f}s — {e}")
        traceback.print_exc()
        # Write error log
        with open(os.path.join(vid_dir, "error.txt"), "w") as f:
            f.write(f"{e}\n\n{traceback.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch-process OHP videos.")
    parser.add_argument(
        "--input_dir",
        default=r"C:\Users\josep\Desktop\Capstone_new\videos\First 500 Vids",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PIPELINE_ROOT, "batch_outputs"),
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=1, help="(reserved for future multiprocessing)")
    parser.add_argument("--max_videos", type=int, default=0, help="Limit videos (0=all)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.output_dir, exist_ok=True)

    videos = _discover_videos(args.input_dir)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    # Pre-load depth estimator (shared across videos)
    depth_est = None
    if cfg["depth"]["enabled"]:
        print("Loading depth model (shared) ...")
        depth_est = DepthEstimator(
            backend=cfg["depth"]["backend"],
            model_size=cfg["depth"]["model_size"],
            model_id=cfg["depth"].get("model_id"),
            cache_dir=os.path.join(cfg["pipeline"]["cache_dir"], "depth"),
            colorize_previews=cfg["depth"]["colorize_previews"],
            device="cuda" if _has_cuda() else "cpu",
        )

    summary = {"total": len(videos), "success": 0, "failed": 0, "skipped": 0}
    t_start = time.time()

    for i, vpath in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] Processing {os.path.basename(vpath)} ...")
        result = process_single_video(vpath, args.output_dir, cfg, depth_est)
        if result is None:
            # Distinguish skip vs error
            vid_name = Path(vpath).stem
            err_file = os.path.join(args.output_dir, vid_name, "error.txt")
            if os.path.exists(os.path.join(args.output_dir, vid_name, "analysis.json")):
                summary["skipped"] += 1
            elif os.path.exists(err_file):
                summary["failed"] += 1
            else:
                summary["skipped"] += 1
        else:
            summary["success"] += 1

    elapsed = time.time() - t_start
    summary["elapsed_sec"] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"Batch complete: {summary['success']} OK / {summary['failed']} failed / {summary['skipped']} skipped")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    with open(os.path.join(args.output_dir, "batch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
