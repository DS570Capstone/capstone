"""
run_single_video.py — Process one OHP back-view .mp4 end-to-end.

Usage:
    python -m src.app.run_single_video --video path/to/clip.mp4 [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

# Ensure ohp_form_pipeline root is on the path regardless of invocation method
_PIPELINE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.io.video_loader import load_video_meta, load_all_frames, iter_frames
from src.io.json_writer import build_empty_artifact, write_clip_json
from src.cv.pose_estimator import (
    PoseEstimator,
    PoseResult,
    extract_angles_from_pose,
    KP,
    KEYPOINT_NAMES,
)
from src.cv.bar_detector import BarDetector, BarDetection
from src.cv.tracker import smooth_poses, smooth_bar_detections, smooth_1d
from src.cv.depth_estimator import DepthEstimator
from src.signals.normalization import compute_scale, compute_midline_x
from src.signals.smoothing import savgol_smooth
from src.signals.segmentation import segment_phases, phases_to_dicts
from src.signals.trajectory_builder import build_all_trajectories
from src.signals.feature_engineering import (
    compute_bar_features,
    compute_bilateral_features,
    compute_trunk_features,
    compute_hip_features,
    compute_leg_features,
    assemble_feature_vector,
)
from src.signals.wave_analysis import compute_wave_features
from src.unsupervised.cluster_naming import assign_clip_fault_flags
from src.reasoning.rule_engine import load_rules, select_rules, format_coaching_feedback
from src.reasoning.feedback_generator import VLMFeedbackGenerator
from src.viz.annotated_video import write_annotated_video, draw_skeleton, draw_bar
from src.viz.report_generator import generate_text_report


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(video_path: str, config_path: str, output_dir: str = None) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    root_cfg = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # ── Output dirs ──
    video_out = os.path.join(output_dir or cfg["pipeline"]["output_dir"], "videos")
    plots_out = os.path.join(output_dir or cfg["pipeline"]["output_dir"], "plots")
    reports_out = os.path.join(output_dir or cfg["pipeline"]["output_dir"], "reports")
    json_out = os.path.join(output_dir or cfg["pipeline"]["output_dir"], "json")
    for d in [video_out, plots_out, reports_out, json_out]:
        os.makedirs(d, exist_ok=True)

    # ── Stage 1: Ingest ──
    print(f"[1/8] Loading video: {video_path}")
    meta = load_video_meta(video_path)
    vid_id = meta.video_id

    # ── Duration guard ───────────────────────────────────────────────────────
    # Reject before loading any frames — avoids OOM on long videos.
    # A typical OHP set is 5-30s; 90s is a very generous ceiling.
    _MAX_DURATION_SEC = 90.0
    if meta.duration_sec > _MAX_DURATION_SEC:
        raise VideoValidationError(
            f"Video is {meta.duration_sec:.0f}s long. "
            f"Maximum supported duration is {_MAX_DURATION_SEC:.0f}s. "
            "Please trim your video to a single OHP set before uploading."
        )

    # Always persist verification artifacts in processed/ for quick visual QA.
    processed_root = os.path.join(root_cfg, "data", "processed", vid_id)
    os.makedirs(processed_root, exist_ok=True)

    frames, _ = load_all_frames(
        video_path,
        max_height=cfg["pipeline"]["max_height"],
        frame_step=cfg["pipeline"]["frame_step"],
    )
    n_frames = len(frames)
    print(
        f"      {meta.n_frames} frames @ {meta.fps:.1f} fps  ({meta.duration_sec:.2f}s)"
    )

    artifact = build_empty_artifact(vid_id, video_path, meta.fps, n_frames)

    # ── Stage 2: Pose + Bar ──
    print("[2/8] Estimating pose + detecting bar ...")
    pose_cfg = cfg["pose"]
    estimator = PoseEstimator(
        backend=pose_cfg["backend"],
        confidence_threshold=pose_cfg["confidence_threshold"],
    )
    poses_raw = estimator.process_video(frames)
    estimator.close()

    camera_motion = _estimate_camera_motion(frames, poses_raw)
    _write_camera_motion(
        camera_motion, os.path.join(processed_root, f"{vid_id}_camera_motion.csv")
    )
    poses_cam = _apply_camera_compensation_to_poses(poses_raw, camera_motion)

    tracking_cfg = cfg.get("tracking_filter", {})
    poses_pre, pose_filter_stats, pose_long_gap_mask = _clean_pose_sequence(
        poses_cam,
        confidence_threshold=float(
            tracking_cfg.get("confidence_threshold", pose_cfg["confidence_threshold"])
        ),
        jump_ratio_threshold=float(tracking_cfg.get("jump_ratio_threshold", 0.25)),
        min_jump_px=float(tracking_cfg.get("min_jump_px", 6.0)),
        max_interp_gap=int(tracking_cfg.get("max_interp_gap", 5)),
    )

    use_body_norm = bool(tracking_cfg.get("body_normalized_smoothing", True))
    if pose_cfg.get("smooth_method", "savgol") == "savgol" and use_body_norm:
        poses = _smooth_poses_body_normalized(
            poses_pre,
            window=pose_cfg["smooth_window"],
            poly=pose_cfg["smooth_poly"],
        )
    else:
        poses = smooth_poses(
            poses_pre,
            window=pose_cfg["smooth_window"],
            poly=pose_cfg["smooth_poly"],
            method=pose_cfg.get("smooth_method", "savgol"),
            fps=meta.fps,
            one_euro_min_cutoff=pose_cfg.get("one_euro_min_cutoff", 1.0),
            one_euro_beta=pose_cfg.get("one_euro_beta", 0.02),
            one_euro_d_cutoff=pose_cfg.get("one_euro_d_cutoff", 1.0),
            min_visibility=pose_cfg.get(
                "one_euro_min_visibility", pose_cfg["confidence_threshold"]
            ),
            max_jump_px=pose_cfg.get("one_euro_max_jump_px", 60.0),
            max_interp_gap=pose_cfg.get("one_euro_max_interp_gap", 8),
        )
    if pose_cfg.get("smooth_method", "savgol") == "savgol":
        poses = _restore_pose_long_gaps(poses, poses_pre, pose_long_gap_mask)

    # ── Stage 2b: VP3D temporal 3D lifting ───────────────────────────────────
    vp3d_cfg = cfg.get("vp3d", {})
    if vp3d_cfg.get("enabled", False):
        model_path = os.path.join(
            root_cfg,
            vp3d_cfg.get("model_path", "models/pretrained_h36m_detectron_coco.bin"),
        )
        if not os.path.exists(model_path):
            print(
                f"[2b] VP3D skipped — model not found at {model_path}\n"
                "     Download with:\n"
                "       mkdir -p models\n"
                "       curl -L -o models/pretrained_h36m_detectron_coco.bin \\\n"
                "         https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
            )
        else:
            print("[2b] VP3D: lifting 2D → 3D via temporal TCN ...")
            from src.cv.vp3d_lifter import VP3DLifter, compute_processed_frame_size

            proc_w, proc_h = compute_processed_frame_size(
                meta.width, meta.height, cfg["pipeline"]["max_height"]
            )
            lifter = VP3DLifter(
                model_path=model_path,
                filter_widths=vp3d_cfg.get("filter_widths", [3, 3, 3, 3, 3]),
                channels=vp3d_cfg.get("channels", 1024),
                dropout=vp3d_cfg.get("dropout", 0.25),
                device="cuda" if _has_cuda() else "cpu",
            )
            poses = lifter.lift_poses(poses, proc_w, proc_h)
            print(f"      VP3D world_landmarks updated for {len(poses)} frames")

    bar_cfg = cfg["bar"]
    bar_detector = BarDetector(
        backend=bar_cfg["backend"],
        wrist_span_scale=bar_cfg["wrist_span_scale"],
        min_confidence=bar_cfg["min_confidence"],
    )
    bars_pre, bar_filter_stats, bar_long_gap_mask = _clean_bar_sequence(
        bar_detector.detect_sequence(frames, poses_raw),
        poses_raw,
        min_confidence=float(tracking_cfg.get("bar_confidence_threshold", 0.2)),
        jump_ratio_threshold=float(tracking_cfg.get("bar_jump_ratio_threshold", 0.25)),
        min_jump_px=float(tracking_cfg.get("bar_min_jump_px", 6.0)),
        max_interp_gap=int(tracking_cfg.get("max_interp_gap", 5)),
    )
    bars_cam = _apply_camera_compensation_to_bars(bars_pre, camera_motion)
    bars = smooth_bar_detections(
        bars_cam, window=pose_cfg["smooth_window"], poly=pose_cfg["smooth_poly"]
    )
    if pose_cfg.get("smooth_method", "savgol") == "savgol":
        bars = _restore_bar_long_gaps(bars, bars_cam, bar_long_gap_mask)

    artifact["tracking_filter"] = {
        "pose": pose_filter_stats,
        "bar": bar_filter_stats,
    }

    jitter_cfg = cfg.get("jitter_filter", {})
    bad_frame_mask = _detect_jitter_frames(
        poses,
        max_jump_ratio=float(jitter_cfg.get("max_jump_ratio", 0.32)),
        min_valid_keypoints=int(jitter_cfg.get("min_valid_keypoints", 8)),
        min_joint_jump_px=float(jitter_cfg.get("min_joint_jump_px", 6.0)),
    )
    poses = _repair_bad_pose_frames(
        poses, bad_frame_mask, max_interp_gap=int(jitter_cfg.get("max_interp_gap", 4))
    )
    bars = _repair_bad_bar_frames(
        bars, bad_frame_mask, max_interp_gap=int(jitter_cfg.get("max_interp_gap", 4))
    )
    artifact["n_frames"] = int(n_frames)
    artifact["duration_sec"] = round(float(n_frames / max(meta.fps, 1e-6)), 3)
    jitter_info = {
        "enabled": True,
        "original_count": int(n_frames),
        "bad_count": int(np.count_nonzero(bad_frame_mask)),
        "bad_frame_indices": [int(i) for i in np.where(bad_frame_mask)[0].tolist()],
        "max_jump_ratio": float(jitter_cfg.get("max_jump_ratio", 0.32)),
        "min_valid_keypoints": int(jitter_cfg.get("min_valid_keypoints", 8)),
        "min_joint_jump_px": float(jitter_cfg.get("min_joint_jump_px", 6.0)),
        "max_interp_gap": int(jitter_cfg.get("max_interp_gap", 4)),
    }
    artifact["jitter_filter"] = jitter_info
    if jitter_info["bad_count"] > 0:
        print(
            f"      Jitter filter flagged {jitter_info['bad_count']} frame(s) out of {jitter_info['original_count']}"
        )
    with open(
        os.path.join(processed_root, f"{vid_id}_jitter_filter.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(jitter_info, fh, indent=2)

    pose_overlay_path = os.path.join(processed_root, f"{vid_id}_pose_overlay.mp4")
    _write_pose_overlay_video(
        frames,
        poses,
        bars,
        pose_overlay_path,
        meta.fps,
        max_hold=int(cfg.get("viz", {}).get("overlay_max_hold", 3)),
    )
    _write_mediapipe_keypoints(
        poses, os.path.join(processed_root, f"{vid_id}_mediapipe_keypoints")
    )
    del frames  # free ~500 MB — reloaded on-demand for Stage 8 output only

    # ── Video validation ──────────────────────────────────────────────────────
    # Runs on pose/bar data from Stage 2 — no extra compute cost.
    # Raises VideoValidationError with a user-friendly message if the video is
    # unsuitable (no person, no movement, wrong camera angle).
    if pose_cfg.get("backend", "mediapipe") != "disabled":
        _validate_video(poses, bars, meta.fps)
    else:
        print("      Pose validation skipped — pose estimation is disabled.")

    # ── Stage 3: Depth ──
    depths = []
    depth_feats = {
        "depth_enabled": False,
        "bar_forward_drift_depth": 0.0,
        "bar_depth_asymmetry": 0.0,
        "torso_depth_shift": 0.0,
        "subject_depth_stability": 0.0,
    }
    if False:
        print("[3/8] Estimating depth ...")
        depth_cache = os.path.join(cfg["pipeline"]["cache_dir"], "depth")
        depth_est = DepthEstimator(
            backend=cfg["depth"]["backend"],
            model_size=cfg["depth"]["model_size"],
            model_id=cfg["depth"].get("model_id"),
            cache_dir=depth_cache,
            colorize_previews=cfg["depth"]["colorize_previews"],
            device="cuda" if _has_cuda() else "cpu",
        )
        depths = depth_est.process_video(frames, video_id=vid_id)
        _save_depth_previews(depths, os.path.join(processed_root, "depth_maps"))
        df_raw = depth_est.extract_depth_features(depths, poses, bars)
        depth_feats = {
            "depth_enabled": True,
            "bar_forward_drift_depth": df_raw.get("bar_forward_drift_depth", 0.0),
            "bar_depth_asymmetry": df_raw.get("bar_depth_asymmetry", 0.0),
            "torso_depth_shift": df_raw.get("torso_depth_shift", 0.0),
            "subject_depth_stability": df_raw.get("subject_depth_stability", 0.0),
        }
        artifact["raw_signals"]["bar_center_z_proxy"] = df_raw.get(
            "bar_center_z_proxy", []
        )
        artifact["raw_signals"]["left_right_wrist_depth_diff"] = df_raw.get(
            "left_right_wrist_depth_diff", []
        )
        artifact["raw_signals"]["bar_depth_relative_to_shoulder_plane"] = df_raw.get(
            "bar_depth_relative_to_shoulder_plane", []
        )
    else:
        print("[3/8] Depth removed — skipping.")
        _write_depth_disabled_note(os.path.join(processed_root, "depth_maps"))
    artifact["depth_features"] = depth_feats

    # ── Stage 4: Trajectories + Raw signals ──
    print("[4/8] Building trajectories and raw signals ...")
    scale = compute_scale(poses)
    midline_x = compute_midline_x(poses)
    resample_len = cfg["pipeline"]["max_frames"]

    traj_dict = build_all_trajectories(
        poses,
        bars,
        resample_len=resample_len,
        smooth_window=cfg["signals"]["smooth_window"],
    )
    artifact["trajectories"] = {
        k: v for k, v in traj_dict.items() if not k.startswith("_")
    }

    # Raw framewise signals
    def kp_y(kp_name):
        return np.array([p.keypoints[KP[kp_name], 1] for p in poses])

    def kp_x(kp_name):
        return np.array([p.keypoints[KP[kp_name], 0] for p in poses])

    bar_cx = np.array([b.center_x for b in bars])
    bar_cy = np.array([b.center_y for b in bars])
    bar_tilt = np.array([b.tilt_deg for b in bars])

    angle_seq = [extract_angles_from_pose(p) for p in poses]
    left_elbow_deg = np.array(
        [a.get("left_elbow_angle_deg", np.nan) for a in angle_seq]
    )
    right_elbow_deg = np.array(
        [a.get("right_elbow_angle_deg", np.nan) for a in angle_seq]
    )
    shoulder_tilt = np.array(
        [a.get("shoulder_line_tilt_deg", np.nan) for a in angle_seq]
    )
    hip_tilt_arr = np.array([a.get("hip_line_tilt_deg", np.nan) for a in angle_seq])

    ls_x = kp_x("left_shoulder")
    rs_x = kp_x("right_shoulder")
    lh_x = kp_x("left_hip")
    rh_x = kp_x("right_hip")
    trunk_cx_arr = np.nanmean(
        np.stack([(ls_x + rs_x) / 2.0, (lh_x + rh_x) / 2.0]), axis=0
    )
    hip_cx_arr = (lh_x + rh_x) / 2.0

    lk_deg = np.array([a.get("left_knee_angle_deg", np.nan) for a in angle_seq])
    rk_deg = np.array([a.get("right_knee_angle_deg", np.nan) for a in angle_seq])

    def clean(arr):
        return [round(float(v), 6) if not np.isnan(v) else None for v in arr]

    artifact["raw_signals"].update(
        {
            "bar_center_x": clean(bar_cx),
            "bar_center_y": clean(bar_cy),
            "bar_tilt_deg": clean(bar_tilt),
            "left_wrist_y": clean(kp_y("left_wrist")),
            "right_wrist_y": clean(kp_y("right_wrist")),
            "left_elbow_angle_deg": clean(left_elbow_deg),
            "right_elbow_angle_deg": clean(right_elbow_deg),
            "shoulder_line_tilt_deg": clean(shoulder_tilt),
            "hip_line_tilt_deg": clean(hip_tilt_arr),
            "trunk_center_x": clean(trunk_cx_arr),
            "left_knee_angle_deg": clean(lk_deg),
            "right_knee_angle_deg": clean(rk_deg),
        }
    )

    advanced = _extract_advanced_biomechanics(
        poses=poses,
        bars=bars,
        fps=meta.fps,
        window_size=int(cfg.get("advanced", {}).get("window_size_frames", 45)),
    )
    artifact["advanced_analysis"] = advanced["summary"]
    np.save(
        os.path.join(processed_root, f"{vid_id}_normalized_skeleton.npy"),
        advanced["normalized_skeleton"],
    )
    np.save(
        os.path.join(processed_root, f"{vid_id}_temporal_windows.npy"),
        advanced["temporal_windows"],
    )
    with open(
        os.path.join(processed_root, f"{vid_id}_advanced_analysis.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(advanced["summary"], fh, indent=2)

    # ── Stage 5: Segmentation ──
    print("[5/8] Segmenting rep phases ...")
    bar_cy_smooth = savgol_smooth(bar_cy, cfg["signals"]["smooth_window"])
    phases = segment_phases(bar_cy_smooth, meta.fps)
    artifact["phase_segments"] = phases_to_dicts(phases)

    # Phase label per original frame
    phase_per_frame = ["unknown"] * n_frames
    for ph in phases:
        for fi in range(ph.start_frame, min(ph.end_frame + 1, n_frames)):
            phase_per_frame[fi] = ph.phase_type

    # ── Stage 6: Feature engineering + wave analysis ──
    print("[6/8] Computing features and wave analysis ...")
    bar_cy_scaled = bar_cy / (scale + 1e-8)
    bar_feats = compute_bar_features(
        bar_cx, bar_cy_scaled, bar_tilt, meta.fps, scale, midline_x
    )
    bilateral_feats = compute_bilateral_features(
        kp_y("left_wrist"),
        kp_y("right_wrist"),
        left_elbow_deg,
        right_elbow_deg,
        meta.fps,
        scale,
    )
    trunk_feats = compute_trunk_features(
        shoulder_tilt, hip_tilt_arr, trunk_cx_arr, scale
    )
    hip_feats = compute_hip_features(hip_cx_arr, scale)
    leg_feats = compute_leg_features(lk_deg, rk_deg)

    all_features = {
        **bar_feats,
        **bilateral_feats,
        **trunk_feats,
        **hip_feats,
        **leg_feats,
    }
    if depth_feats["depth_enabled"]:
        all_features["bar_depth_drift_normalized"] = depth_feats[
            "bar_forward_drift_depth"
        ]

    # Normalize bar_cy by body scale so energy/power outputs are in
    # shoulder-width units (dimensionless) instead of raw pixels.
    # Without this, Work+/Work- are in px²/s² and reach 200,000+.
    bar_cy_norm = bar_cy_smooth / (scale + 1e-8)
    wave_feats = compute_wave_features(bar_cy_norm, meta.fps, phases, scale)
    # Inject symmetry into quality
    wave_feats["quality"]["symmetry"] = bilateral_feats.get("symmetry_score", 0.0)
    artifact["wave_features"] = wave_feats

    # ── Fault flags ──
    thresholds_path = os.path.join(root_cfg, "configs", "thresholds.yaml")
    with open(thresholds_path) as f:
        thresholds_cfg = yaml.safe_load(f)
    fault_flags = assign_clip_fault_flags(all_features, thresholds_cfg)
    artifact["fault_flags"] = fault_flags

    # ── Stage 7: Language ──
    print("[7/8] Generating coaching feedback ...")
    rules_path = os.path.join(root_cfg, "configs", "rules_ohp.yaml")
    rules = load_rules(rules_path)
    triggered = select_rules(fault_flags, rules)
    rule_feedback = format_coaching_feedback(
        triggered,
        "unknown",
        wave_feats["quality"],
        rules,
        uncertainty=depth_feats["depth_enabled"],
    )
    # VLM (if enabled) — only sample key frames when actually needed
    vlm_cfg = cfg.get("vlm", {})
    vlm_gen = VLMFeedbackGenerator(vlm_cfg)
    if vlm_cfg.get("enabled", False):
        _vlm_frames = [
            f
            for _, f in iter_frames(
                video_path,
                max_height=cfg["pipeline"]["max_height"],
                frame_step=cfg["pipeline"]["frame_step"],
            )
        ]
        key_frames = _sample_key_frames(_vlm_frames, n=vlm_cfg.get("num_key_frames", 4))
        del _vlm_frames
    else:
        key_frames = []
    language = vlm_gen.generate(
        artifact, key_frames=key_frames, rule_based_fallback=rule_feedback
    )
    artifact["language"] = language

    # ── Stage 8: Outputs ──
    print("[8/8] Saving outputs ...")
    # JSON
    json_path = write_clip_json(artifact, json_out, vid_id)
    print(f"  JSON: {json_path}")

    # Annotated video — reload frames from disk only when needed
    if cfg["viz"]["save_annotated_video"]:
        frames = [
            f
            for _, f in iter_frames(
                video_path,
                max_height=cfg["pipeline"]["max_height"],
                frame_step=cfg["pipeline"]["frame_step"],
            )
        ]
        vid_path = os.path.join(video_out, f"{vid_id}_annotated.mp4")
        write_annotated_video(
            frames, poses, bars, fault_flags, phase_per_frame, vid_path, meta.fps
        )
        print(f"  Annotated video: {vid_path}")
        del frames

    # Plots
    if cfg["viz"]["save_plots"]:
        from src.viz.signal_plots import (
            plot_trajectories,
            plot_signal_dashboard,
            plot_bilateral_symmetry,
            plot_harmonic_wave_patterns,
        )

        plot_trajectories(
            artifact["trajectories"],
            artifact["phase_segments"],
            os.path.join(plots_out, f"{vid_id}_trajectories.png"),
            meta.fps,
        )
        plot_signal_dashboard(
            bar_cy_smooth,
            meta.fps,
            artifact["phase_segments"],
            os.path.join(plots_out, f"{vid_id}_dashboard.png"),
        )
        plot_bilateral_symmetry(
            kp_y("left_wrist"),
            kp_y("right_wrist"),
            left_elbow_deg,
            right_elbow_deg,
            os.path.join(plots_out, f"{vid_id}_bilateral.png"),
        )
        if cfg.get("viz", {}).get("save_harmonic_plot", True):
            plot_harmonic_wave_patterns(
                bar_cy,
                meta.fps,
                wave_feats,
                os.path.join(plots_out, f"{vid_id}_harmonic_wave.png"),
            )

    # Text report
    generate_text_report(artifact, os.path.join(reports_out, f"{vid_id}_report.txt"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {vid_id}")
    print(
        f"  Grade: {wave_feats['quality']['grade']}  Cluster: unknown  Faults: {[k for k,v in fault_flags.items() if v]}"
    )
    return artifact


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _sample_key_frames(frames: list, n: int = 4) -> list:
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def _write_pose_overlay_video(
    frames: list[np.ndarray],
    poses,
    bars,
    out_path: str,
    fps: float,
    max_hold: int = 3,
) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    disp_poses = _make_visual_continuity_poses(poses, max_hold=max_hold)
    disp_bars = _make_visual_continuity_bars(bars, max_hold=max_hold)
    for frame, pose, bar in zip(frames, disp_poses, disp_bars):
        annotated = frame.copy()  # one copy; draw_skeleton/draw_bar work in-place
        draw_skeleton(annotated, pose, color=(0, 140, 255), radius=4, thickness=2)
        for idx, pt in enumerate(pose.keypoints):
            if np.isnan(pt).any() or not pose.visible[idx]:
                continue
            cv2.putText(
                annotated,
                str(idx),
                (int(pt[0]) + 3, int(pt[1]) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 140, 255),
                1,
                cv2.LINE_AA,
            )
        draw_bar(annotated, bar)
        writer.write(annotated)
    writer.release()


def _write_mediapipe_keypoints(poses, out_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    rows = []
    for p in poses:
        for idx, kp_name in enumerate(KEYPOINT_NAMES):
            x, y = p.keypoints[idx]
            rows.append(
                {
                    "frame_idx": int(p.frame_idx),
                    "keypoint_idx": int(idx),
                    "keypoint_name": kp_name,
                    "x": _safe_float(x),
                    "y": _safe_float(y),
                    "visibility": _safe_float(p.confidences[idx]),
                    "is_visible": bool(p.visible[idx]),
                }
            )

    json_path = f"{out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_idx",
                "keypoint_idx",
                "keypoint_name",
                "x",
                "y",
                "visibility",
                "is_visible",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(v: float):
    if np.isnan(v):
        return None
    return round(float(v), 6)


def _save_depth_previews(depths: list[np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, depth in enumerate(depths):
        np.save(os.path.join(out_dir, f"{i:06d}.npy"), depth)
        preview = cv2.applyColorMap(
            (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), preview)


def _write_depth_disabled_note(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("Depth estimation is disabled in the active config for this run.\n")


def _estimate_camera_motion(frames: list[np.ndarray], poses) -> np.ndarray:
    if len(frames) <= 1:
        return np.zeros((len(frames), 2), dtype=np.float64)

    cumulative = np.zeros((len(frames), 2), dtype=np.float64)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.full((h, w), 255, dtype=np.uint8)

        pts = poses[i - 1].keypoints
        valid = pts[~np.isnan(pts).any(axis=1)]
        if len(valid) > 0:
            x0, y0 = np.min(valid[:, 0]), np.min(valid[:, 1])
            x1, y1 = np.max(valid[:, 0]), np.max(valid[:, 1])
            pad = 30
            xa, ya = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            xb, yb = int(min(w - 1, x1 + pad)), int(min(h - 1, y1 + pad))
            mask[ya:yb, xa:xb] = 0

        p0 = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=250, qualityLevel=0.01, minDistance=8, mask=mask
        )
        flow = np.array([0.0, 0.0], dtype=np.float64)
        if p0 is not None and len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            if p1 is not None and st is not None:
                good = st.reshape(-1) == 1
                if np.any(good):
                    delta = (p1[good] - p0[good]).reshape(-1, 2)
                    flow = np.median(delta, axis=0)

        cumulative[i] = cumulative[i - 1] + flow
        prev_gray = gray

    return cumulative


def _write_camera_motion(motion: np.ndarray, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_idx",
                "camera_dx",
                "camera_dy",
                "camera_dx_cum",
                "camera_dy_cum",
            ],
        )
        writer.writeheader()
        prev = np.array([0.0, 0.0], dtype=np.float64)
        for i, cur in enumerate(motion):
            inc = cur - prev
            writer.writerow(
                {
                    "frame_idx": i,
                    "camera_dx": round(float(inc[0]), 6),
                    "camera_dy": round(float(inc[1]), 6),
                    "camera_dx_cum": round(float(cur[0]), 6),
                    "camera_dy_cum": round(float(cur[1]), 6),
                }
            )
            prev = cur


def _apply_camera_compensation_to_poses(poses, motion: np.ndarray):
    out = []
    for p in poses:
        idx = min(max(int(p.frame_idx), 0), len(motion) - 1)
        shift = motion[idx]
        kp = p.keypoints.copy().astype(np.float64)
        valid = ~np.isnan(kp).any(axis=1)
        kp[valid, 0] -= shift[0]
        kp[valid, 1] -= shift[1]
        out.append(
            type(p)(
                keypoints=kp,
                confidences=p.confidences.copy(),
                visible=p.visible.copy(),
                frame_idx=p.frame_idx,
                confidence_threshold=p.confidence_threshold,
            )
        )
    return out


def _apply_camera_compensation_to_bars(bars, motion: np.ndarray):
    from src.cv.bar_detector import BarDetection

    out = []
    for b in bars:
        idx = min(max(int(b.frame_idx), 0), len(motion) - 1)
        sx, sy = motion[idx]

        cx = float("nan") if np.isnan(b.center_x) else float(b.center_x - sx)
        cy = float("nan") if np.isnan(b.center_y) else float(b.center_y - sy)

        le = b.left_end
        re = b.right_end
        if any(np.isnan(v) for v in le):
            le2 = le
        else:
            le2 = (float(le[0] - sx), float(le[1] - sy))
        if any(np.isnan(v) for v in re):
            re2 = re
        else:
            re2 = (float(re[0] - sx), float(re[1] - sy))

        out.append(
            BarDetection(
                center_x=cx,
                center_y=cy,
                tilt_deg=b.tilt_deg,
                left_end=le2,
                right_end=re2,
                confidence=b.confidence,
                frame_idx=b.frame_idx,
                method=b.method,
            )
        )
    return out


def _clean_pose_sequence(
    poses,
    confidence_threshold: float,
    jump_ratio_threshold: float,
    min_jump_px: float,
    max_interp_gap: int,
):
    joint_names = [
        "left_wrist",
        "right_wrist",
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ]
    joint_idx = [KP[n] for n in joint_names]

    out = []
    for p in poses:
        kp = p.keypoints.copy().astype(np.float64)
        conf = p.confidences.copy().astype(np.float64)
        vis = p.visible.copy()
        low_conf = conf < confidence_threshold
        kp[low_conf] = np.nan
        vis[low_conf] = False
        out.append(
            PoseResult(
                keypoints=kp,
                confidences=conf,
                visible=vis,
                frame_idx=p.frame_idx,
                confidence_threshold=p.confidence_threshold,
            )
        )

    rejected = 0
    last_valid = {k: None for k in joint_idx}
    for i in range(len(out)):
        cur = out[i]
        scale = _estimate_torso_scale_px(cur)
        jump_thresh = max(min_jump_px, jump_ratio_threshold * scale)
        for k in joint_idx:
            b = cur.keypoints[k]
            if np.isnan(b).any():
                continue
            a = last_valid[k]
            if a is None:
                last_valid[k] = b.copy()
                continue
            d = float(np.linalg.norm(b - a))
            if d > jump_thresh:
                cur.keypoints[k] = np.nan
                cur.visible[k] = False
                cur.confidences[k] = 0.0
                rejected += 1
                continue
            last_valid[k] = b.copy()

    t = len(out)
    n_kp = len(KEYPOINT_NAMES)
    long_gap_mask = np.zeros((t, n_kp), dtype=bool)
    interpolated_points = 0
    for k in range(n_kp):
        x = np.array([p.keypoints[k, 0] for p in out], dtype=np.float64)
        y = np.array([p.keypoints[k, 1] for p in out], dtype=np.float64)
        x_filled, x_long, x_interp = _interp_short_gaps_1d(x, max_interp_gap)
        y_filled, y_long, y_interp = _interp_short_gaps_1d(y, max_interp_gap)
        long_gap_mask[:, k] = x_long | y_long
        interp_mask = x_interp | y_interp
        interpolated_points += int(np.count_nonzero(interp_mask))
        for i, p in enumerate(out):
            p.keypoints[k, 0] = x_filled[i]
            p.keypoints[k, 1] = y_filled[i]
            if interp_mask[i] and not (np.isnan(x_filled[i]) or np.isnan(y_filled[i])):
                p.visible[k] = True
                p.confidences[k] = max(p.confidences[k], confidence_threshold)

    stats = {
        "confidence_threshold": float(confidence_threshold),
        "jump_ratio_threshold": float(jump_ratio_threshold),
        "min_jump_px": float(min_jump_px),
        "max_interp_gap": int(max_interp_gap),
        "joint_jump_rejections": int(rejected),
        "short_gap_interpolated_points": int(interpolated_points),
    }
    return out, stats, long_gap_mask


def _restore_pose_long_gaps(poses_smooth, poses_base, long_gap_mask: np.ndarray):
    out = []
    for i, p in enumerate(poses_smooth):
        kp = p.keypoints.copy().astype(np.float64)
        conf = p.confidences.copy().astype(np.float64)
        vis = p.visible.copy()
        mask = long_gap_mask[i]
        kp[mask] = np.nan
        vis[mask] = False
        conf[mask] = 0.0
        out.append(
            PoseResult(
                keypoints=kp,
                confidences=conf,
                visible=vis,
                frame_idx=poses_base[i].frame_idx,
                confidence_threshold=poses_base[i].confidence_threshold,
            )
        )
    return out


def _clean_bar_sequence(
    bars,
    poses,
    min_confidence: float,
    jump_ratio_threshold: float,
    min_jump_px: float,
    max_interp_gap: int,
):
    cx = np.array([b.center_x for b in bars], dtype=np.float64)
    cy = np.array([b.center_y for b in bars], dtype=np.float64)
    tilt = np.array([b.tilt_deg for b in bars], dtype=np.float64)
    conf = np.array([b.confidence for b in bars], dtype=np.float64)

    low_conf = conf < min_confidence
    cx[low_conf] = np.nan
    cy[low_conf] = np.nan
    tilt[low_conf] = np.nan

    rejected = 0
    last_valid = None
    for i in range(len(cx)):
        if np.isnan(cx[i]) or np.isnan(cy[i]):
            continue
        scale = _estimate_torso_scale_px(poses[i])
        jump_thresh = max(min_jump_px, jump_ratio_threshold * scale)
        if last_valid is None:
            last_valid = np.array([cx[i], cy[i]], dtype=np.float64)
            continue
        disp = float(np.hypot(cx[i] - last_valid[0], cy[i] - last_valid[1]))
        if disp > jump_thresh:
            cx[i] = np.nan
            cy[i] = np.nan
            tilt[i] = np.nan
            rejected += 1
            continue
        last_valid = np.array([cx[i], cy[i]], dtype=np.float64)

    cx_filled, cx_long, cx_interp = _interp_short_gaps_1d(cx, max_interp_gap)
    cy_filled, cy_long, cy_interp = _interp_short_gaps_1d(cy, max_interp_gap)
    tilt_filled, tilt_long, tilt_interp = _interp_short_gaps_1d(tilt, max_interp_gap)
    long_gap_mask = cx_long | cy_long | tilt_long
    interp_mask = cx_interp | cy_interp | tilt_interp

    cleaned = []
    for i, b in enumerate(bars):
        c = float(conf[i]) if not np.isnan(conf[i]) else 0.0
        if interp_mask[i]:
            c = max(c, min_confidence)
        cleaned.append(
            BarDetection(
                center_x=(
                    float(cx_filled[i]) if not np.isnan(cx_filled[i]) else float("nan")
                ),
                center_y=(
                    float(cy_filled[i]) if not np.isnan(cy_filled[i]) else float("nan")
                ),
                tilt_deg=(
                    float(tilt_filled[i])
                    if not np.isnan(tilt_filled[i])
                    else float("nan")
                ),
                left_end=b.left_end,
                right_end=b.right_end,
                confidence=c,
                frame_idx=b.frame_idx,
                method=b.method,
            )
        )

    stats = {
        "confidence_threshold": float(min_confidence),
        "jump_ratio_threshold": float(jump_ratio_threshold),
        "min_jump_px": float(min_jump_px),
        "max_interp_gap": int(max_interp_gap),
        "bar_jump_rejections": int(rejected),
        "short_gap_interpolated_points": int(np.count_nonzero(interp_mask)),
    }
    return cleaned, stats, long_gap_mask


def _restore_bar_long_gaps(bars_smooth, bars_base, long_gap_mask: np.ndarray):
    out = []
    for i, b in enumerate(bars_smooth):
        if long_gap_mask[i]:
            out.append(
                BarDetection(
                    center_x=float("nan"),
                    center_y=float("nan"),
                    tilt_deg=float("nan"),
                    left_end=bars_base[i].left_end,
                    right_end=bars_base[i].right_end,
                    confidence=0.0,
                    frame_idx=bars_base[i].frame_idx,
                    method=bars_base[i].method,
                )
            )
        else:
            out.append(b)
    return out


def _interp_short_gaps_1d(arr: np.ndarray, max_gap: int):
    out = arr.copy().astype(np.float64)
    n = len(out)
    long_gap_mask = np.zeros(n, dtype=bool)
    interp_mask = np.zeros(n, dtype=bool)
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
        can_fill = (
            left >= 0
            and right < n
            and gap_len <= max_gap
            and not np.isnan(out[left])
            and not np.isnan(out[right])
        )
        if can_fill:
            out[start : end + 1] = np.interp(
                np.arange(start, end + 1), [left, right], [out[left], out[right]]
            )
            interp_mask[start : end + 1] = True
        else:
            long_gap_mask[start : end + 1] = True
    return out, long_gap_mask, interp_mask


def _detect_jitter_frames(
    poses,
    max_jump_ratio: float = 0.32,
    min_valid_keypoints: int = 8,
    min_joint_jump_px: float = 6.0,
):
    if len(poses) <= 2:
        return np.zeros(len(poses), dtype=bool)

    bad = np.zeros(len(poses), dtype=bool)

    for i in range(1, len(poses)):
        prev = poses[i - 1]
        cur = poses[i]

        valid = (
            prev.visible
            & cur.visible
            & (~np.isnan(prev.keypoints).any(axis=1))
            & (~np.isnan(cur.keypoints).any(axis=1))
        )
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < min_valid_keypoints:
            continue

        prev_xy = prev.keypoints[valid_idx, :2]
        cur_xy = cur.keypoints[valid_idx, :2]
        disp = np.linalg.norm(cur_xy - prev_xy, axis=1)
        large = disp[disp >= min_joint_jump_px]
        if len(large) == 0:
            continue

        torso_scale = _estimate_torso_scale_px(cur)
        jump_ratio = float(np.median(large) / max(torso_scale, 1e-6))
        if jump_ratio > max_jump_ratio:
            bad[i] = True

    return bad


def _repair_bad_pose_frames(poses, bad_mask: np.ndarray, max_interp_gap: int = 4):
    out = []
    for i, p in enumerate(poses):
        kp = p.keypoints.copy().astype(np.float64)
        conf = p.confidences.copy().astype(np.float64)
        vis = p.visible.copy()
        if bad_mask[i]:
            kp[:] = np.nan
            conf[:] = 0.0
            vis[:] = False
        out.append(
            PoseResult(
                keypoints=kp,
                confidences=conf,
                visible=vis,
                frame_idx=p.frame_idx,
                confidence_threshold=p.confidence_threshold,
            )
        )

    n_kp = len(KEYPOINT_NAMES)
    for k in range(n_kp):
        x = np.array([p.keypoints[k, 0] for p in out], dtype=np.float64)
        y = np.array([p.keypoints[k, 1] for p in out], dtype=np.float64)
        x_filled, _, x_interp = _interp_short_gaps_1d(x, max_interp_gap)
        y_filled, _, y_interp = _interp_short_gaps_1d(y, max_interp_gap)
        interp_mask = x_interp | y_interp
        for i, p in enumerate(out):
            p.keypoints[k, 0] = x_filled[i]
            p.keypoints[k, 1] = y_filled[i]
            if interp_mask[i] and not (np.isnan(x_filled[i]) or np.isnan(y_filled[i])):
                p.visible[k] = True
                p.confidences[k] = max(p.confidences[k], p.confidence_threshold)
    return out


def _repair_bad_bar_frames(bars, bad_mask: np.ndarray, max_interp_gap: int = 4):
    cx = np.array([b.center_x for b in bars], dtype=np.float64)
    cy = np.array([b.center_y for b in bars], dtype=np.float64)
    tilt = np.array([b.tilt_deg for b in bars], dtype=np.float64)
    conf = np.array([b.confidence for b in bars], dtype=np.float64)

    cx[bad_mask] = np.nan
    cy[bad_mask] = np.nan
    tilt[bad_mask] = np.nan
    conf[bad_mask] = 0.0

    cx_filled, _, cx_interp = _interp_short_gaps_1d(cx, max_interp_gap)
    cy_filled, _, cy_interp = _interp_short_gaps_1d(cy, max_interp_gap)
    tilt_filled, _, tilt_interp = _interp_short_gaps_1d(tilt, max_interp_gap)
    interp_mask = cx_interp | cy_interp | tilt_interp

    out = []
    for i, b in enumerate(bars):
        c = float(conf[i])
        if interp_mask[i]:
            c = max(c, 0.2)
        out.append(
            BarDetection(
                center_x=(
                    float(cx_filled[i]) if not np.isnan(cx_filled[i]) else float("nan")
                ),
                center_y=(
                    float(cy_filled[i]) if not np.isnan(cy_filled[i]) else float("nan")
                ),
                tilt_deg=(
                    float(tilt_filled[i])
                    if not np.isnan(tilt_filled[i])
                    else float("nan")
                ),
                left_end=b.left_end,
                right_end=b.right_end,
                confidence=c,
                frame_idx=b.frame_idx,
                method=b.method,
            )
        )
    return out


def _make_visual_continuity_poses(poses, max_hold: int = 3):
    if not poses:
        return poses
    n_kp = len(KEYPOINT_NAMES)
    last_xy = [None] * n_kp
    hold_age = [0] * n_kp
    out = []
    for p in poses:
        kp = p.keypoints.copy().astype(np.float64)
        conf = p.confidences.copy().astype(np.float64)
        vis = p.visible.copy()
        for k in range(n_kp):
            valid = (not np.isnan(kp[k]).any()) and bool(vis[k])
            if valid:
                last_xy[k] = kp[k].copy()
                hold_age[k] = 0
                continue
            if last_xy[k] is not None and hold_age[k] < max_hold:
                kp[k] = last_xy[k]
                vis[k] = True
                conf[k] = max(conf[k], 0.2)
                hold_age[k] += 1
            else:
                hold_age[k] = max_hold
        out.append(
            PoseResult(
                keypoints=kp,
                confidences=conf,
                visible=vis,
                frame_idx=p.frame_idx,
                confidence_threshold=p.confidence_threshold,
            )
        )
    return out


def _make_visual_continuity_bars(bars, max_hold: int = 3):
    if not bars:
        return bars
    last_bar = None
    hold_age = 0
    out = []
    for b in bars:
        valid = not (
            np.isnan(b.center_x) or np.isnan(b.center_y) or np.isnan(b.tilt_deg)
        )
        if valid:
            last_bar = b
            hold_age = 0
            out.append(b)
            continue
        if last_bar is not None and hold_age < max_hold:
            hold_age += 1
            out.append(
                BarDetection(
                    center_x=last_bar.center_x,
                    center_y=last_bar.center_y,
                    tilt_deg=last_bar.tilt_deg,
                    left_end=last_bar.left_end,
                    right_end=last_bar.right_end,
                    confidence=max(0.2, b.confidence),
                    frame_idx=b.frame_idx,
                    method=b.method,
                )
            )
        else:
            out.append(b)
    return out


def _smooth_poses_body_normalized(poses, window: int = 11, poly: int = 3):
    if not poses:
        return poses

    t = len(poses)
    n_kp = len(KEYPOINT_NAMES)

    centers = np.full((t, 2), np.nan, dtype=np.float64)
    scales = np.full((t,), np.nan, dtype=np.float64)

    for i, p in enumerate(poses):
        ls = p.keypoints[KP["left_shoulder"]]
        rs = p.keypoints[KP["right_shoulder"]]
        lh = p.keypoints[KP["left_hip"]]
        rh = p.keypoints[KP["right_hip"]]

        hips = np.stack([lh, rh], axis=0)
        shld = np.stack([ls, rs], axis=0)

        hip_ok = ~np.isnan(hips).any(axis=1)
        sh_ok = ~np.isnan(shld).any(axis=1)
        if np.any(hip_ok):
            centers[i] = np.nanmean(hips[hip_ok], axis=0)
        elif np.any(sh_ok):
            centers[i] = np.nanmean(shld[sh_ok], axis=0)

        shoulder_w = (
            np.linalg.norm(rs - ls)
            if not (np.isnan(ls).any() or np.isnan(rs).any())
            else np.nan
        )
        hip_w = (
            np.linalg.norm(rh - lh)
            if not (np.isnan(lh).any() or np.isnan(rh).any())
            else np.nan
        )
        vals = np.array([shoulder_w, hip_w], dtype=np.float64)
        if not np.isnan(vals).all():
            scales[i] = np.nanmedian(vals)

    centers[:, 0] = _nan_interp(centers[:, 0])
    centers[:, 1] = _nan_interp(centers[:, 1])
    scales = _nan_interp(scales)
    scales[scales <= 1e-6] = 40.0

    out = [
        PoseResult(
            keypoints=p.keypoints.copy().astype(np.float64),
            confidences=p.confidences.copy().astype(np.float64),
            visible=p.visible.copy(),
            frame_idx=p.frame_idx,
            confidence_threshold=p.confidence_threshold,
        )
        for p in poses
    ]

    for k in range(n_kp):
        x = np.array([p.keypoints[k, 0] for p in poses], dtype=np.float64)
        y = np.array([p.keypoints[k, 1] for p in poses], dtype=np.float64)

        x_norm = (x - centers[:, 0]) / scales
        y_norm = (y - centers[:, 1]) / scales

        x_s = smooth_1d(x_norm, window=window, poly=poly)
        y_s = smooth_1d(y_norm, window=window, poly=poly)

        x_out = x_s * scales + centers[:, 0]
        y_out = y_s * scales + centers[:, 1]

        for i in range(t):
            if np.isnan(x[i]) or np.isnan(y[i]):
                continue
            out[i].keypoints[k, 0] = float(x_out[i])
            out[i].keypoints[k, 1] = float(y_out[i])

    return out


def _estimate_torso_scale_px(pose) -> float:
    ls = pose.keypoints[KP["left_shoulder"]]
    rs = pose.keypoints[KP["right_shoulder"]]
    lh = pose.keypoints[KP["left_hip"]]
    rh = pose.keypoints[KP["right_hip"]]

    shoulder_w = (
        np.linalg.norm(rs - ls)
        if not (np.isnan(ls).any() or np.isnan(rs).any())
        else np.nan
    )
    hip_w = (
        np.linalg.norm(rh - lh)
        if not (np.isnan(lh).any() or np.isnan(rh).any())
        else np.nan
    )

    vals = np.array([shoulder_w, hip_w], dtype=np.float64)
    if np.isnan(vals).all():
        return 40.0
    scale = np.nanmedian(vals)
    if np.isnan(scale) or scale <= 0:
        return 40.0
    return float(scale)


def _extract_advanced_biomechanics(
    poses, bars, fps: float, window_size: int = 45
) -> dict:
    n = len(poses)
    n_kp = len(KEYPOINT_NAMES)
    skeleton = np.full((n, n_kp, 2), np.nan, dtype=np.float32)
    for i, p in enumerate(poses):
        skeleton[i] = p.keypoints.astype(np.float32)

    lh = skeleton[:, KP["left_hip"], :]
    rh = skeleton[:, KP["right_hip"], :]
    mid_hip = np.nanmean(np.stack([lh, rh], axis=0), axis=0)
    normalized = skeleton - mid_hip[:, None, :]

    hip_y = _nan_interp(mid_hip[:, 1].astype(np.float64))
    hip_vel = np.gradient(hip_y) * fps
    zero_v = np.where(np.diff(np.sign(hip_vel)) != 0)[0].astype(int)

    bar_x = _nan_interp(np.array([b.center_x for b in bars], dtype=np.float64))
    bar_y = _nan_interp(np.array([b.center_y for b in bars], dtype=np.float64))
    vx = np.gradient(bar_x) * fps
    vy = np.gradient(bar_y) * fps
    ax = np.gradient(vx) * fps
    ay = np.gradient(vy) * fps
    jx = np.gradient(ax) * fps
    jy = np.gradient(ay) * fps
    jerk_mag = np.sqrt(jx * jx + jy * jy)

    curv_num = np.abs(vx * ay - vy * ax)
    curv_den = np.power(vx * vx + vy * vy + 1e-8, 1.5)
    curvature = curv_num / curv_den

    lhip = skeleton[:, KP["left_hip"], :]
    lknee = skeleton[:, KP["left_knee"], :]
    lank = skeleton[:, KP["left_ankle"], :]
    rhip = skeleton[:, KP["right_hip"], :]
    rknee = skeleton[:, KP["right_knee"], :]
    rank = skeleton[:, KP["right_ankle"], :]
    l_cos = _angle_cosine(lhip, lknee, lank)
    r_cos = _angle_cosine(rhip, rknee, rank)

    ls = skeleton[:, KP["left_shoulder"], :]
    rs = skeleton[:, KP["right_shoulder"], :]
    torso_twist = _planar_twist(ls, rs, lh, rh)

    windows = _sliding_windows(normalized, window_size=window_size)

    summary = {
        "window_size_frames": int(window_size),
        "num_windows": int(windows.shape[0]),
        "harmonic_signal": {
            "hip_y_mean": round(float(np.nanmean(hip_y)), 6),
            "hip_y_std": round(float(np.nanstd(hip_y)), 6),
            "zero_velocity_points": [int(v) for v in zero_v.tolist()],
            "zero_velocity_count": int(len(zero_v)),
        },
        "smoothness": {
            "jerk_rms": round(float(np.sqrt(np.nanmean(jerk_mag * jerk_mag))), 6),
            "jerk_p95": round(float(np.nanpercentile(jerk_mag, 95)), 6),
            "path_curvature_mean": round(float(np.nanmean(curvature)), 6),
            "path_curvature_p95": round(float(np.nanpercentile(curvature, 95)), 6),
        },
        "biomechanics": {
            "left_knee_cos_theta_mean": round(float(np.nanmean(l_cos)), 6),
            "right_knee_cos_theta_mean": round(float(np.nanmean(r_cos)), 6),
            "torso_twist_mean": round(float(np.nanmean(torso_twist)), 6),
            "torso_twist_p95": round(float(np.nanpercentile(torso_twist, 95)), 6),
        },
    }

    return {
        "summary": summary,
        "normalized_skeleton": normalized,
        "temporal_windows": windows,
    }


def _nan_interp(arr: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(np.float64)
    nans = np.isnan(out)
    if nans.all():
        return np.zeros_like(out)
    if nans.any():
        idx = np.where(~nans)[0]
        out[nans] = np.interp(np.where(nans)[0], idx, out[idx])
    return out


def _sliding_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0:
        window_size = 1
    if len(arr) < window_size:
        return arr[None, ...]
    return np.stack(
        [arr[i : i + window_size] for i in range(0, len(arr) - window_size + 1)], axis=0
    )


def _angle_cosine(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    num = np.sum(ba * bc, axis=1)
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    return num / den


def _planar_twist(
    ls: np.ndarray, rs: np.ndarray, lh: np.ndarray, rh: np.ndarray
) -> np.ndarray:
    sh = rs - ls
    hp = rh - lh
    sh3 = np.concatenate([sh, np.zeros((len(sh), 1), dtype=sh.dtype)], axis=1)
    hp3 = np.concatenate([hp, np.zeros((len(hp), 1), dtype=hp.dtype)], axis=1)
    cross = np.cross(sh3, hp3)
    return np.linalg.norm(cross, axis=1)


class VideoValidationError(ValueError):
    """Raised when a video fails pre-analysis validation."""

    pass


def _validate_video(
    poses: list,
    bars: list,
    fps: float,
) -> None:
    """
    Three lightweight checks on data already computed in Stage 2.
    Raises VideoValidationError with a user-friendly message if any check fails.
    No new models — pure logic on pose keypoints and bar positions.
    """
    n = len(poses)
    if n == 0:
        raise VideoValidationError("No frames could be decoded from this video.")

    # ── Critical keypoints used for OHP analysis (back-view) ──────────────────
    _CRITICAL = [
        KP["left_shoulder"],
        KP["right_shoulder"],
        KP["left_elbow"],
        KP["right_elbow"],
        KP["left_wrist"],
        KP["right_wrist"],
        KP["left_hip"],
        KP["right_hip"],
    ]

    # ── Check 1: Is a person visible? ─────────────────────────────────────────
    # Count frames where at least 4 of the 8 critical keypoints are visible.
    # Threshold: 30% of frames must have a detectable person.
    frames_with_person = sum(
        1 for p in poses if sum(1 for idx in _CRITICAL if p.visible[idx]) >= 4
    )
    person_ratio = frames_with_person / n
    if person_ratio < 0.30:
        raise VideoValidationError(
            f"No person detected in {100 * (1 - person_ratio):.0f}% of frames "
            f"({frames_with_person}/{n} frames had a visible person). "
            "Make sure the video shows a person from the back and the full body is in frame."
        )

    # ── Check 2: Is there overhead press movement? ────────────────────────────
    # The bar (inferred from wrists) must travel vertically by at least 15% of
    # the frame height to constitute a meaningful press. A static video or a
    # squat/deadlift will have near-zero vertical bar travel.
    bar_cy = np.array([b.center_y for b in bars])
    valid_bar = bar_cy[~np.isnan(bar_cy)]
    if len(valid_bar) < 6:
        raise VideoValidationError(
            "Bar position could not be tracked in enough frames. "
            "Make sure both wrists are visible throughout the lift."
        )
    bar_travel_ratio = (np.max(valid_bar) - np.min(valid_bar)) / max(
        np.nanmax(
            [
                p.keypoints[:, 1].max()
                for p in poses
                if not np.all(np.isnan(p.keypoints))
            ]
        ),
        1.0,
    )
    if bar_travel_ratio < 0.08:
        raise VideoValidationError(
            f"No overhead press movement detected (bar vertical travel = "
            f"{bar_travel_ratio * 100:.1f}% of frame height, expected ≥ 8%). "
            "Upload a video of an overhead press, not a static hold or different exercise."
        )

    # ── Check 3: Is this a back-view camera angle? ────────────────────────────
    # In a back-view video the subject faces away from the camera:
    #   - Both shoulders should be visible most of the time.
    #   - Nose should be invisible most of the time (face away from camera).
    # If nose is consistently MORE visible than shoulders, it's likely front-view.
    nose_vis = np.mean([float(p.visible[KP["nose"]]) for p in poses])
    ls_vis = np.mean([float(p.visible[KP["left_shoulder"]]) for p in poses])
    rs_vis = np.mean([float(p.visible[KP["right_shoulder"]]) for p in poses])
    shoulder_vis = (ls_vis + rs_vis) / 2.0

    if nose_vis > 0.5 and nose_vis > shoulder_vis * 1.5:
        raise VideoValidationError(
            f"This appears to be a front-view video (nose visible in "
            f"{nose_vis * 100:.0f}% of frames). "
            "LiftLens requires a back-view camera angle — position the camera "
            "behind the lifter."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single OHP video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    # Change to pipeline root
    script_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(script_dir)

    run(args.video, args.config, args.output_dir)
