"""
VP3D Temporal 3D Lifter
=======================
Lifts MediaPipe 2D keypoints to 3D using the VideoPose3D pre-trained TCN,
AND writes cleaned 2D back into pose.keypoints so the entire downstream
pipeline (bar detector, feature engineering, quality score) benefits.

Pipeline:
  MediaPipe 33 kp (pixel)
    → select COCO-17 subset
    → linear interpolation over NaN frames      (VP3D prepare_data_2d_custom style)
    → normalize_screen_coordinates              (VP3D camera.py style)
    → Savitzky-Golay smooth along time axis
    → [A] denormalize → write pose.keypoints    ← fixes bar detector / quality
    → edge-pad to receptive field
    → TemporalModel (pre-trained: pretrained_h36m_detectron_coco.bin)
    → H36M-17 3D output (camera space, root-relative)
    → [B] write pose.world_landmarks            ← depth features

Pre-trained model download:
  mkdir -p ml/ohp_form_pipeline/models
  curl -L -o ml/ohp_form_pipeline/models/pretrained_h36m_detectron_coco.bin \\
    https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .pose_estimator import PoseResult

# ---------------------------------------------------------------------------
# Joint-index mappings
# ---------------------------------------------------------------------------

# MediaPipe 33 indices for the 17 COCO joints (in COCO order)
# COCO: nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el,
#       l_wr, r_wr, l_hip, r_hip, l_kn, r_kn, l_ank, r_ank
_MP_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# H36M-17 output joints → MediaPipe 33 indices (None = virtual joint, skip)
# H36M order: hip_c, r_hip, r_kn, r_ank, l_hip, l_kn, l_ank,
#             spine, thorax, neck, head, l_sh, l_el, l_wr, r_sh, r_el, r_wr
_H36M_TO_MP33: dict[int, Optional[int]] = {
    0:  None,   # hip_center  — virtual
    1:  24,     # right_hip   → MP right_hip
    2:  26,     # right_knee  → MP right_knee
    3:  28,     # right_ankle → MP right_ankle
    4:  23,     # left_hip    → MP left_hip
    5:  25,     # left_knee   → MP left_knee
    6:  27,     # left_ankle  → MP left_ankle
    7:  None,   # spine       — virtual
    8:  None,   # thorax      — virtual
    9:  0,      # neck/nose   → MP nose
    10: None,   # head        — virtual
    11: 11,     # left_shoulder  → MP left_shoulder
    12: 13,     # left_elbow     → MP left_elbow
    13: 15,     # left_wrist     → MP left_wrist
    14: 12,     # right_shoulder → MP right_shoulder
    15: 14,     # right_elbow    → MP right_elbow
    16: 16,     # right_wrist    → MP right_wrist
}

# ---------------------------------------------------------------------------
# Coordinate helpers (from VP3D common/camera.py)
# ---------------------------------------------------------------------------

def _normalize_screen(X: np.ndarray, w: int, h: int) -> np.ndarray:
    """Map pixel coords [0,w] × [0,h] → aspect-ratio-preserving [-1,1] × [-h/w, h/w]."""
    assert X.shape[-1] == 2
    return X / w * 2.0 - np.array([1.0, h / w], dtype=np.float32)


def _image_coordinates(norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """Inverse of _normalize_screen: normalized → pixel coords (VP3D camera.py style)."""
    assert norm.shape[-1] == 2
    return (norm + np.array([1.0, h / w], dtype=np.float32)) * (w / 2.0)


# ---------------------------------------------------------------------------
# TemporalModel (self-contained copy from VP3D common/model.py)
# — torch/nn are imported lazily inside VP3DLifter.__init__ so the module
#   can be imported without torch installed (VP3D disabled = torch never loaded)
# ---------------------------------------------------------------------------

def _build_temporal_model(nn, num_joints_in, in_features, num_joints_out,
                          filter_widths, causal=False, dropout=0.25, channels=1024):
    """Factory that builds the TemporalModel using a pre-imported nn module."""

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            for fw in filter_widths:
                assert fw % 2 != 0, "Only odd filter widths supported"
            self.num_joints_in = num_joints_in
            self.in_features = in_features
            self.num_joints_out = num_joints_out
            self.filter_widths = filter_widths
            self.drop = nn.Dropout(dropout)
            self.relu = nn.ReLU(inplace=True)
            self.pad = [filter_widths[0] // 2]
            self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
            self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

        def receptive_field(self) -> int:
            return 1 + 2 * sum(self.pad)

        def forward(self, x):
            assert len(x.shape) == 4
            sz = x.shape[:3]
            x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = self._forward_blocks(x)
            x = x.permute(0, 2, 1).view(sz[0], -1, self.num_joints_out, 3)
            return x

    class _TemporalModel(_Base):
        def __init__(self):
            super().__init__()
            self.expand_conv = nn.Conv1d(
                num_joints_in * in_features, channels, filter_widths[0], bias=False)
            layers_conv, layers_bn = [], []
            self.causal_shift = [filter_widths[0] // 2 if causal else 0]
            next_dilation = filter_widths[0]
            for i in range(1, len(filter_widths)):
                self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
                self.causal_shift.append(
                    (filter_widths[i] // 2 * next_dilation) if causal else 0)
                layers_conv.append(nn.Conv1d(
                    channels, channels, filter_widths[i],
                    dilation=next_dilation, bias=False))
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
                layers_conv.append(nn.Conv1d(channels, channels, 1, bias=False))
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
                next_dilation *= filter_widths[i]
            self.layers_conv = nn.ModuleList(layers_conv)
            self.layers_bn = nn.ModuleList(layers_bn)

        def _forward_blocks(self, x):
            x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
            for i in range(len(self.pad) - 1):
                pad = self.pad[i + 1]
                shift = self.causal_shift[i + 1]
                res = x[:, :, pad + shift: x.shape[2] - pad + shift]
                x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
                x = res + self.drop(
                    self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
            return self.shrink(x)

    return _TemporalModel()


# ---------------------------------------------------------------------------
# VP3DLifter
# ---------------------------------------------------------------------------

class VP3DLifter:
    """
    Loads the pre-trained VideoPose3D TCN and lifts a full video's worth of
    MediaPipe PoseResults from 2D pixel space to 3D camera space.

    Parameters
    ----------
    model_path : str
        Path to ``pretrained_h36m_detectron_coco.bin``.
    filter_widths : list[int]
        TCN architecture — must match the checkpoint (default [3,3,3,3,3]).
    channels : int
        Number of conv channels — must match the checkpoint (default 1024).
    dropout : float
        Dropout probability (default 0.25).
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    NUM_IN_JOINTS = 17   # COCO-17 input
    NUM_OUT_JOINTS = 17  # H36M-17 output
    IN_FEATURES = 2      # (x, y) normalized

    def __init__(
        self,
        model_path: str,
        filter_widths: list[int] = None,
        channels: int = 1024,
        dropout: float = 0.25,
        device: str = "cpu",
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for VP3D lifting. Install it:\n"
                "  pip install -r backend/requirements-extras.txt\n"
                "Or inside the container:\n"
                "  docker exec liftlens-api pip install torch==2.3.0+cpu "
                "--extra-index-url https://download.pytorch.org/whl/cpu"
            )

        if filter_widths is None:
            filter_widths = [3, 3, 3, 3, 3]

        self._torch = torch
        self.device = torch.device(device)
        self._model = _build_temporal_model(
            nn,
            self.NUM_IN_JOINTS, self.IN_FEATURES, self.NUM_OUT_JOINTS,
            filter_widths, causal=False, dropout=dropout, channels=channels,
        ).to(self.device)
        self._model.eval()

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        # Checkpoint stores weights under 'model_pos'
        self._model.load_state_dict(ckpt["model_pos"])

        self._pad = (self._model.receptive_field() - 1) // 2
        print(
            f"[VP3DLifter] Loaded from {model_path} | "
            f"receptive field: {self._model.receptive_field()} frames | "
            f"pad: {self._pad}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lift_poses(
        self,
        poses: list[PoseResult],
        frame_width: int,
        frame_height: int,
    ) -> list[PoseResult]:
        """
        Run VP3D lifting on a full video sequence.

        Two things are updated in-place on every PoseResult:
          A. ``keypoints``       — cleaned 2D pixel coords (NaN-filled + smoothed)
          B. ``world_landmarks`` — VP3D 3D camera-space coords

        Step A is what actually improves bar detection, smoothness, and quality
        scoring — all downstream code reads ``pose.keypoints``.

        Parameters
        ----------
        poses : list[PoseResult]
            Output of ``PoseEstimator.process_video()`` (after smoothing).
        frame_width, frame_height : int
            Pixel dimensions of the processed frames (after any resizing).
        """
        if not poses:
            return poses

        # 1. Extract COCO-17 2D keypoints  (T, 17, 2) in pixel space
        kp2d = self._extract_coco17(poses)

        # 2. Fill NaN frames by linear interpolation (VP3D prepare_data_2d_custom style)
        kp2d = self._interpolate_missing(kp2d)

        # 3. Normalize to VP3D screen space  [-1,1] × [-h/w, h/w]
        kp2d_norm = _normalize_screen(kp2d, frame_width, frame_height)

        # 4. Smooth normalized 2D along time (Savitzky-Golay)
        #    This is the key step — removes MediaPipe jitter before writing back
        kp2d_smooth = self._smooth_2d_norm(kp2d_norm)

        # 5A. Denormalize and overwrite pose.keypoints with cleaned 2D
        #     Bar detector, feature engineering, and quality scoring all read
        #     pose.keypoints — this is what makes VP3D actually improve quality.
        self._write_2d_keypoints(poses, kp2d_smooth, frame_width, frame_height)

        # 5B. Run TCN → 3D and write into world_landmarks
        poses_3d = self._run_tcn(kp2d_norm)
        self._write_back(poses, poses_3d)

        return poses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_coco17(self, poses: list[PoseResult]) -> np.ndarray:
        """Stack COCO-17 pixel coords from MediaPipe 33-kp results → (T, 17, 2)."""
        T = len(poses)
        out = np.full((T, 17, 2), np.nan, dtype=np.float32)
        for t, pose in enumerate(poses):
            for coco_i, mp_i in enumerate(_MP_TO_COCO17):
                kp = pose.keypoints[mp_i]   # (2,) or NaN
                if not np.isnan(kp).any():
                    out[t, coco_i] = kp
        return out

    @staticmethod
    def _interpolate_missing(kp2d: np.ndarray) -> np.ndarray:
        """
        Linear interpolation over NaN frames per joint per axis.
        Matches VP3D's prepare_data_2d_custom.py lines 44-50.
        Any leading/trailing NaN is filled by nearest-valid edge value.
        """
        T, J, C = kp2d.shape
        indices = np.arange(T, dtype=np.float32)
        out = kp2d.copy()
        for j in range(J):
            for c in range(C):
                col = out[:, j, c]
                mask = ~np.isnan(col)
                if mask.sum() == 0:
                    out[:, j, c] = 0.0  # fully missing joint → zero
                elif mask.sum() < T:
                    out[:, j, c] = np.interp(indices, indices[mask], col[mask])
        return out

    def _run_tcn(self, kp2d_norm: np.ndarray) -> np.ndarray:
        """
        Pad → torch → model → numpy.

        Parameters
        ----------
        kp2d_norm : (T, 17, 2) float32

        Returns
        -------
        (T, 17, 3) float32  — H36M 3D, root-relative, metres
        """
        T = kp2d_norm.shape[0]
        pad = self._pad

        # Edge-pad each side (same as VP3D UnchunkedGenerator)
        padded = np.pad(kp2d_norm, ((pad, pad), (0, 0), (0, 0)), mode="edge")  # (T+2p, 17, 2)

        x = self._torch.from_numpy(padded).float().unsqueeze(0).to(self.device)  # (1, T+2p, 17, 2)

        with self._torch.no_grad():
            out = self._model(x)  # (1, T, 17, 3)

        return out.squeeze(0).cpu().numpy()  # (T, 17, 3)

    @staticmethod
    def _smooth_2d_norm(kp2d_norm: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
        """
        Savitzky-Golay smoothing along the time axis per joint per axis.
        Operates on normalized-screen-space coordinates.
        """
        from scipy.signal import savgol_filter
        T, J, C = kp2d_norm.shape
        if T < 4:
            return kp2d_norm.copy()
        # window must be odd and ≤ T
        w = min(window, T if T % 2 == 1 else T - 1)
        if w < poly + 2:
            return kp2d_norm.copy()
        out = kp2d_norm.copy()
        for j in range(J):
            for c in range(C):
                out[:, j, c] = savgol_filter(out[:, j, c], w, poly)
        return out

    def _write_2d_keypoints(
        self,
        poses: list[PoseResult],
        kp2d_smooth: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """
        Denormalize VP3D-cleaned 2D coords and overwrite pose.keypoints.

        Converts from VP3D normalized screen space back to pixel coords using
        the inverse of _normalize_screen (VP3D camera.py image_coordinates).

        Only the 17 COCO joints (mapped to their MediaPipe indices) are updated.
        The remaining 16 MediaPipe-only joints (e.g. face mesh) are left as-is.
        """
        for t, pose in enumerate(poses):
            for coco_i, mp_i in enumerate(_MP_TO_COCO17):
                norm_xy = kp2d_smooth[t, coco_i]                          # (2,)
                pixel_xy = _image_coordinates(norm_xy, frame_width, frame_height)
                pose.keypoints[mp_i] = pixel_xy.astype(np.float32)
                pose.visible[mp_i] = True
                # Ensure confidence is at least 0.5 — VP3D filled this slot
                pose.confidences[mp_i] = max(float(pose.confidences[mp_i]), 0.5)

    @staticmethod
    def _write_back(poses: list[PoseResult], poses_3d: np.ndarray) -> None:
        """
        Put H36M-17 3D coordinates into each PoseResult.world_landmarks
        at the corresponding MediaPipe-33 joint index.

        Joints without a direct MP equivalent (virtual hip_center, spine, thorax,
        head) are skipped — those slots keep whatever MediaPipe set.
        """
        n_mp = poses[0].keypoints.shape[0]  # 33
        for t, (pose, kp3) in enumerate(zip(poses, poses_3d)):
            if pose.world_landmarks is None:
                pose.world_landmarks = np.full((n_mp, 3), np.nan, dtype=np.float32)
            for h36m_i, mp_i in _H36M_TO_MP33.items():
                if mp_i is not None:
                    pose.world_landmarks[mp_i] = kp3[h36m_i]


# ---------------------------------------------------------------------------
# Utility: compute actual processed frame dimensions
# ---------------------------------------------------------------------------

def compute_processed_frame_size(
    original_width: int,
    original_height: int,
    max_height: int,
) -> tuple[int, int]:
    """
    Return (width, height) of frames after video_loader resize.
    Matches iter_frames() logic in src/io/video_loader.py.
    """
    if original_height > max_height:
        scale = max_height / original_height
        return int(original_width * scale), max_height
    return original_width, original_height
