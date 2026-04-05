"""
ExerciseAware Vision Adapter — trainable layer injected on top of Qwen2.5-VL-0.5B
vision encoder that fuses biomechanical pose features with visual representations.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Qwen2.5-VL-0.5B Vision Encoder (frozen)               │
  │  → visual_hidden  (B, N_patches, D_vision)              │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │  ExerciseAwareAdapter (trainable)                       │
  │                                                         │
  │  1. PoseTokenProjector: pose features → D_vision        │
  │     (keypoints, angles, velocities, phase)              │
  │                                                         │
  │  2. HarmonicEncoder: FFT features of movement signal    │
  │     (captures rhythmic/harmonic nature of exercise)     │
  │                                                         │
  │  3. CrossAttention: visual tokens attend to pose tokens │
  │     → exercise-aware visual features                    │
  │                                                         │
  │  4. Gated fusion: α·original + (1-α)·adapted           │
  └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoseTokenProjector(nn.Module):
    """Projects per-frame pose features into the vision hidden dimension.

    Input features per frame:
      - 33 keypoints × 2 coords = 66
      - 6 joint angles (elbow L/R, knee L/R, shoulder tilt, hip tilt) = 6
      - 6 velocities (wrist L/R y-vel, bar x/y vel, elbow L/R angular vel) = 6
      - 4 phase features (one-hot: descent, ascent, lockout, unknown) = 4
      - 3 bar features (center_x, center_y, tilt_deg) = 3
      Total = 85 features per frame
    """

    POSE_DIM = 85

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self.POSE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        """pose_features: (B, T, 85) → (B, T, D)"""
        return self.proj(pose_features)


class HarmonicEncoder(nn.Module):
    """Encodes the harmonic/rhythmic nature of exercise movement via FFT features.

    Captures the periodic structure of rep cadence, bar path oscillation,
    and bilateral symmetry patterns.
    """

    def __init__(self, n_signals: int = 4, n_harmonics: int = 16, d_model: int = 896,
                 dropout: float = 0.1):
        super().__init__()
        self.n_harmonics = n_harmonics
        # Each signal produces: n_harmonics magnitudes + n_harmonics phases + 3 summary stats
        feat_per_signal = n_harmonics * 2 + 3  # mag, phase, dominant_freq, spectral_centroid, bandwidth
        total_feat = n_signals * feat_per_signal
        self.proj = nn.Sequential(
            nn.Linear(total_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def extract_harmonic_features(self, signal: torch.Tensor) -> torch.Tensor:
        """Extract FFT-based harmonic features from a 1D signal.
        signal: (B, T) → (B, n_harmonics*2 + 3)
        """
        B, T = signal.shape
        # Zero-pad to power of 2
        nfft = max(64, 2 ** int(math.ceil(math.log2(T))))
        fft = torch.fft.rfft(signal, n=nfft, dim=-1)
        magnitudes = torch.abs(fft)[:, :self.n_harmonics]  # (B, n_harmonics)
        phases = torch.angle(fft)[:, :self.n_harmonics]      # (B, n_harmonics)

        # Normalize magnitudes
        mag_sum = magnitudes.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        magnitudes = magnitudes / mag_sum

        # Summary stats
        freq_bins = torch.arange(self.n_harmonics, device=signal.device, dtype=signal.dtype)
        dominant_freq = torch.argmax(magnitudes, dim=-1, keepdim=True).float() / self.n_harmonics
        spectral_centroid = (magnitudes * freq_bins).sum(dim=-1, keepdim=True) / mag_sum.squeeze(-1).unsqueeze(-1).clamp(min=1e-8)
        bandwidth = torch.sqrt(
            (magnitudes * (freq_bins - spectral_centroid) ** 2).sum(dim=-1, keepdim=True)
        )

        return torch.cat([magnitudes, phases, dominant_freq, spectral_centroid, bandwidth], dim=-1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """signals: (B, n_signals, T) → (B, 1, D) single harmonic token"""
        B, S, T = signals.shape
        feats = []
        for i in range(S):
            feats.append(self.extract_harmonic_features(signals[:, i, :]))
        harmonic = torch.cat(feats, dim=-1)  # (B, total_feat)
        return self.proj(harmonic).unsqueeze(1)  # (B, 1, D)


class ExerciseCrossAttention(nn.Module):
    """Visual tokens cross-attend to pose+harmonic tokens."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, visual_tokens: torch.Tensor,
                exercise_tokens: torch.Tensor) -> torch.Tensor:
        """
        visual_tokens:   (B, N_vis, D)
        exercise_tokens: (B, N_ex, D)  — pose tokens + harmonic token
        Returns:         (B, N_vis, D)
        """
        attended, _ = self.attn(
            query=visual_tokens,
            key=exercise_tokens,
            value=exercise_tokens,
        )
        x = self.norm1(visual_tokens + attended)
        x = self.norm2(x + self.ffn(x))
        return x


class ExerciseAwareAdapter(nn.Module):
    """Full adapter module that sits between vision encoder and LLM.

    Fuses biomechanical understanding into visual representations via:
    1. Pose feature projection (spatial body awareness)
    2. Harmonic encoding (temporal movement rhythm)
    3. Cross-attention fusion (visual ↔ exercise feature alignment)
    4. Gated residual connection (smooth training start)
    """

    def __init__(
        self,
        d_model: int = 896,  # Qwen2.5-VL-0.5B hidden dim
        n_cross_attn_layers: int = 2,
        n_heads: int = 8,
        n_harmonics: int = 16,
        n_harmonic_signals: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pose_proj = PoseTokenProjector(d_model, dropout)
        self.harmonic_enc = HarmonicEncoder(
            n_signals=n_harmonic_signals,
            n_harmonics=n_harmonics,
            d_model=d_model,
            dropout=dropout,
        )
        self.cross_attn_layers = nn.ModuleList([
            ExerciseCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_attn_layers)
        ])
        # Gated fusion: starts near 0 so adapter is ~identity at init
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        visual_hidden: torch.Tensor,        # (B, N_patches, D)
        pose_features: torch.Tensor,         # (B, T_frames, 85)
        harmonic_signals: torch.Tensor,      # (B, 4, T_signal)
    ) -> torch.Tensor:
        """Inject exercise understanding into visual representations."""
        # Project pose features into model dimension
        pose_tokens = self.pose_proj(pose_features)      # (B, T, D)
        # Encode harmonic features
        harmonic_token = self.harmonic_enc(harmonic_signals)  # (B, 1, D)
        # Combine exercise context tokens
        exercise_tokens = torch.cat([pose_tokens, harmonic_token], dim=1)  # (B, T+1, D)

        # Multi-layer cross attention
        adapted = visual_hidden
        for layer in self.cross_attn_layers:
            adapted = layer(adapted, exercise_tokens)

        # Gated residual: sigmoid(gate) controls adapter contribution
        alpha = torch.sigmoid(self.gate)
        fused = (1 - alpha) * visual_hidden + alpha * adapted
        return fused

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_pose_feature_vector(
    keypoints: np.ndarray,     # (N_KP, 2) per frame
    angles: dict,              # joint angles dict per frame
    bar_cx: float,
    bar_cy: float,
    bar_tilt: float,
    phase: str,
    prev_keypoints: np.ndarray | None = None,
    prev_bar: tuple | None = None,
    fps: float = 30.0,
) -> np.ndarray:
    """Build the 85-dim feature vector for a single frame.

    Used during data preparation to create training features from
    the batch processing outputs.
    """
    from ..cv.pose_estimator import KEYPOINT_NAMES

    feats = []

    # 1. Keypoints (66 dims) — normalize to [0, 1] roughly
    kp_flat = keypoints.flatten()  # 33*2 = 66
    kp_flat = np.nan_to_num(kp_flat, nan=0.0)
    feats.extend(kp_flat.tolist())

    # 2. Joint angles (6 dims) — normalize to [0, 1] via /180
    angle_names = [
        "left_elbow_angle_deg", "right_elbow_angle_deg",
        "left_knee_angle_deg", "right_knee_angle_deg",
        "shoulder_line_tilt_deg", "hip_line_tilt_deg",
    ]
    for name in angle_names:
        val = angles.get(name, 0.0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        feats.append(val / 180.0)

    # 3. Velocities (6 dims)
    if prev_keypoints is not None:
        from ..cv.pose_estimator import KP
        dt = 1.0 / fps
        lw_vel = (keypoints[KP["left_wrist"], 1] - prev_keypoints[KP["left_wrist"], 1]) / dt
        rw_vel = (keypoints[KP["right_wrist"], 1] - prev_keypoints[KP["right_wrist"], 1]) / dt
        le_vel = (angles.get("left_elbow_angle_deg", 0) - 0) / dt  # approx
        re_vel = (angles.get("right_elbow_angle_deg", 0) - 0) / dt
        if prev_bar is not None:
            bx_vel = (bar_cx - prev_bar[0]) / dt if not np.isnan(bar_cx) else 0.0
            by_vel = (bar_cy - prev_bar[1]) / dt if not np.isnan(bar_cy) else 0.0
        else:
            bx_vel, by_vel = 0.0, 0.0
        for v in [lw_vel, rw_vel, bx_vel, by_vel, le_vel, re_vel]:
            feats.append(float(np.nan_to_num(v, nan=0.0)) / 1000.0)  # scale
    else:
        feats.extend([0.0] * 6)

    # 4. Phase one-hot (4 dims)
    phase_map = {"descent": 0, "ascent": 1, "lockout": 2, "unknown": 3}
    phase_vec = [0.0] * 4
    phase_vec[phase_map.get(phase, 3)] = 1.0
    feats.extend(phase_vec)

    # 5. Bar features (3 dims)
    feats.append(float(np.nan_to_num(bar_cx, nan=0.0)) / 720.0)
    feats.append(float(np.nan_to_num(bar_cy, nan=0.0)) / 720.0)
    feats.append(float(np.nan_to_num(bar_tilt, nan=0.0)) / 45.0)

    assert len(feats) == 85, f"Expected 85 features, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


def build_harmonic_signals_from_artifact(artifact: dict, target_len: int = 128) -> np.ndarray:
    """Extract 4 trajectory signals from a processed artifact dict.

    Returns: (4, target_len) array of [arm, legs, core, bar_path] trajectories.
    """
    signal_names = ["arm_trajectory", "legs_trajectory", "core_trajectory", "bar_path_trajectory"]
    signals = []
    for name in signal_names:
        raw = artifact.get("trajectories", {}).get(name, [])
        if not raw:
            raw = [0.0] * target_len
        arr = np.array(raw, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        # Resample to target_len
        if len(arr) != target_len:
            from scipy.signal import resample
            arr = resample(arr, target_len).astype(np.float32)
        # Normalize to [-1, 1]
        rng = arr.max() - arr.min()
        if rng > 1e-6:
            arr = 2.0 * (arr - arr.min()) / rng - 1.0
        signals.append(arr)
    return np.stack(signals)  # (4, target_len)
