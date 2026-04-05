"""Depth Anything V2 integration — framewise depth with caching."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


class DepthEstimator:
    """
    Wraps Depth Anything V2 (or V1/MiDaS fallback) for per-frame depth estimation.
    Caches depth maps as .npy files.
    """

    def __init__(
        self,
        backend: str = "depth_anything_v2",
        model_size: str = "small",
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        colorize_previews: bool = True,
        device: str = "cpu",
    ):
        self.backend = backend
        self.model_size = model_size
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.colorize = colorize_previews
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.backend == "depth_anything_v2":
            self._load_depth_anything_v2()
        elif self.backend == "depth_anything_v1":
            self._load_depth_anything_v1()
        elif self.backend == "midas":
            self._load_midas()
        else:
            raise ValueError(f"Unknown depth backend: {self.backend}")

    def _load_depth_anything_v2(self):
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            size_map = {
                "small": "depth-anything/Depth-Anything-V2-Small-hf",
                "base": "depth-anything/Depth-Anything-V2-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Large-hf",
            }
            model_id = self.model_id or size_map.get(self.model_size, size_map["small"])
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=model_id,
                device=(
                    0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
                ),
            )
            self._model = "hf_pipeline"
            print(f"[DepthEstimator] Loaded Depth Anything V2 model: {model_id}")
        except Exception as e:
            print(
                f"[DepthEstimator] Depth Anything V2 failed ({e}), falling back to MiDaS."
            )
            self._load_midas()

    def _load_depth_anything_v1(self):
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device=(
                    0
                    if (
                        self.device == "cuda"
                        and __import__("torch").cuda.is_available()
                    )
                    else -1
                ),
            )
            self._model = "hf_pipeline"
            print("[DepthEstimator] Loaded Depth Anything V1 via HuggingFace.")
        except Exception as e:
            print(
                f"[DepthEstimator] Depth Anything V1 failed ({e}), falling back to MiDaS."
            )
            self._load_midas()

    def _load_midas(self):
        import torch

        self._midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self._midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._midas_transform = midas_transforms.small_transform
        dev = torch.device(
            "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self._midas.to(dev)
        self._midas_device = dev
        self._model = "midas"
        print("[DepthEstimator] Loaded MiDaS small.")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return normalized depth map (H, W) float32 in [0,1], closer = higher value."""
        self._load_model()
        if self._model == "hf_pipeline":
            from PIL import Image as PILImage

            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            result = self._pipe(pil_img)
            depth = np.array(result["depth"], dtype=np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth
        elif self._model == "midas":
            import torch

            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            input_batch = self._midas_transform(rgb).to(self._midas_device)
            with torch.no_grad():
                prediction = self._midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=bgr_frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.cpu().numpy().astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth
        else:
            raise RuntimeError("Model not loaded.")

    def estimate_and_cache(
        self, bgr_frame: np.ndarray, video_id: str, frame_idx: int
    ) -> np.ndarray:
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, video_id, f"{frame_idx:06d}.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)

        depth = self.estimate(bgr_frame)

        if self.cache_dir:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, depth)
            if self.colorize:
                preview = cv2.applyColorMap(
                    (depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                prev_path = cache_path.replace(".npy", "_preview.jpg")
                cv2.imwrite(prev_path, preview)

        return depth

    def process_video(
        self, frames: list[np.ndarray], video_id: str = "unknown"
    ) -> list[np.ndarray]:
        depths = []
        for i, frame in enumerate(frames):
            d = self.estimate_and_cache(frame, video_id, i)
            depths.append(d)
        return depths

    def extract_depth_features(
        self,
        depths: list[np.ndarray],
        poses,  # list[PoseResult]
        bar_detections,  # list[BarDetection]
    ) -> dict:
        """Derive depth-based biomechanical features."""
        from .pose_estimator import KP

        bar_depth_vals, lw_depths, rw_depths, torso_depths = [], [], [], []

        for depth, pose, bar in zip(depths, poses, bar_detections):
            h, w = depth.shape

            def sample(x, y):
                xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
                return float(depth[yi, xi])

            # Bar depth
            if not np.isnan(bar.center_x):
                bar_depth_vals.append(sample(bar.center_x, bar.center_y))
            else:
                bar_depth_vals.append(float("nan"))

            # Wrist depths
            lw = pose.keypoints[KP["left_wrist"]]
            rw = pose.keypoints[KP["right_wrist"]]
            lw_depths.append(
                sample(lw[0], lw[1]) if not np.isnan(lw).any() else float("nan")
            )
            rw_depths.append(
                sample(rw[0], rw[1]) if not np.isnan(rw).any() else float("nan")
            )

            # Torso center depth
            ls = pose.keypoints[KP["left_shoulder"]]
            rs = pose.keypoints[KP["right_shoulder"]]
            lh = pose.keypoints[KP["left_hip"]]
            rh = pose.keypoints[KP["right_hip"]]
            valid = [p for p in [ls, rs, lh, rh] if not np.isnan(p).any()]
            if valid:
                cx = np.mean([p[0] for p in valid])
                cy = np.mean([p[1] for p in valid])
                torso_depths.append(sample(cx, cy))
            else:
                torso_depths.append(float("nan"))

        def nanmean(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return 0.0
            val = float(np.nanmean(arr))
            return 0.0 if np.isnan(val) else val

        def nanstd(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return 0.0
            val = float(np.nanstd(arr))
            return 0.0 if np.isnan(val) else val

        bar_arr = np.array(bar_depth_vals, dtype=np.float64)
        lw_arr = np.array(lw_depths, dtype=np.float64)
        rw_arr = np.array(rw_depths, dtype=np.float64)
        torso_arr = np.array(torso_depths, dtype=np.float64)

        wrist_diff = lw_arr - rw_arr
        bar_torso_diff = bar_arr - torso_arr

        return {
            "bar_center_z_proxy": [round(v, 6) for v in bar_arr.tolist()],
            "left_right_wrist_depth_diff": [round(v, 6) for v in wrist_diff.tolist()],
            "bar_depth_relative_to_shoulder_plane": [
                round(v, 6) for v in bar_torso_diff.tolist()
            ],
            "bar_forward_drift_depth": nanmean(
                np.abs(np.diff(bar_torso_diff[~np.isnan(bar_torso_diff)]))
            ),
            "bar_depth_asymmetry": nanmean(np.abs(wrist_diff[~np.isnan(wrist_diff)])),
            "torso_depth_shift": nanstd(torso_arr[~np.isnan(torso_arr)]),
            "subject_depth_stability": 1.0
            - min(nanstd(torso_arr[~np.isnan(torso_arr)]), 1.0),
        }
