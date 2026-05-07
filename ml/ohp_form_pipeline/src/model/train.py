"""
Training pipeline — Qwen3-VL + ExerciseAwareAdapter with plain-Qwen comparison.

Model A  — Qwen-VL + Adapter:
    Stage 1: ExerciseAwareAdapter (pose+harmonic cross-attention) with quality,
                     fault, and expert prediction heads.
    Stage 2: LoRA fine-tune of Qwen3-VL language model for coaching text.

Model B  — Qwen3-VL Plain:
    Qwen/Qwen3-VL-2B-Thinking without the ExerciseAware adapter.

Language LoRA training always runs and is augmented with a strength and
conditioning manual PDF.

Usage:
        cd ml/ohp_form_pipeline
        python -m src.model.train --stage all --data_dir <artifact_dir> --epochs 15
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm

PIPELINE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PIPELINE_ROOT)

weave = None

try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv(*args, **kwargs):
        return False


# Walk up to find the .env at project root (Capstone_new/)
_env_path = os.path.join(PIPELINE_ROOT, ".env")
if not os.path.exists(_env_path):
    _env_path = os.path.join(os.path.dirname(os.path.dirname(PIPELINE_ROOT)), ".env")
load_dotenv(_env_path)

# HuggingFace auth
_hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HF_TPOKEN", "")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    from huggingface_hub import login as hf_login

    hf_login(token=_hf_token, add_to_git_credential=False)

# Initialize Weave for trace logging only when explicitly enabled.
if os.environ.get("ENABLE_WEAVE", "0") == "1":
    try:
        import weave as _weave

        _weave.init("jnolas77-arizona-state-university/capstone")
        weave = _weave
    except Exception as e:
        print(f"[warn] Weave init failed: {e}")

FAULT_NAMES = [
    "left_right_lockout_asymmetry",
    "uneven_press_timing",
    "compensatory_lateral_shift",
    "trunk_shift_under_load",
    "hip_shift_compensation",
    "unstable_lockout",
]
NUM_FAULTS = len(FAULT_NAMES)

# Quality sub-score keys used as extra scalar features
QUALITY_KEYS = ["smoothness", "control", "efficiency", "consistency", "symmetry"]
DEPTH_KEYS = [
    "torso_depth_shift",
    "subject_depth_stability",
]

GRADE_TO_TIER = {
    "A": "good",
    "B": "good",
    "C": "okay",
    "D": "bad",
    "F": "bad",
}


def _init_run_log(run_name: str, config: dict, tags: list[str] | None = None):
    print(f"[log] run={run_name} tags={tags or []}")
    print(f"[log] config={json.dumps(config, sort_keys=True)}")


class _StreamTee:
    """Mirror stdout/stderr to console and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        try:
            return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)
        except Exception:
            return False


def _enable_txt_logging(log_path: str):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _StreamTee(sys.__stdout__, fh)
    sys.stderr = _StreamTee(sys.__stderr__, fh)
    return fh


def _extract_pdf_text(pdf_path: str) -> str:
    """Read text from a PDF file using pypdf/PyPDF2 if available."""
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[warn] Manual PDF not found: {pdf_path}")
        return ""

    text_chunks = []

    # Try pypdf first.
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text_chunks.append(page.extract_text() or "")
        text = "\n".join(text_chunks)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback to PyPDF2.
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text_chunks.append(page.extract_text() or "")
        text = "\n".join(text_chunks)
        if text.strip():
            return text
    except Exception as e:
        print(f"[warn] Could not parse manual PDF {pdf_path}: {e}")

    # Fallback to system pdftotext if available.
    try:
        text = subprocess.check_output(
            ["pdftotext", "-layout", pdf_path, "-"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        if text.strip():
            return text
    except Exception:
        pass

    return ""


def _chunk_text(text: str, chunk_chars: int = 500) -> list[str]:
    """Chunk text into sentence-aware windows for instruction tuning."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    chunks = []
    cur = ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 > chunk_chars and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks


def _to_json_safe(obj):
    """Convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


# ── Dataset ──────────────────────────────────────────────────────────────────


class OHPDataset(Dataset):
    """Unified dataset that loads every useful signal from analysis artifacts.

    Each sample returns:
        pose_features   : (max_frames, 85)   — per-frame keypoint + angle + bar
        harmonic_signals: (4, signal_len)     — bar/arm/legs/core trajectories
        scalar_features : (n_scalar,)         — quality sub-scores + depth feats
        quality_label   : scalar              — overall quality 0-1
        fault_labels    : (9,)                — binary fault flags
        expert_label    : scalar              — 1 if expert (overall >= 0.75)
    """

    def __init__(self, artifact_dir: str, max_frames: int = 8, signal_len: int = 128):
        self.artifact_dir = artifact_dir
        self.max_frames = max_frames
        self.signal_len = signal_len
        self.samples: list[dict] = []
        self._scan()

    # -- scanning --
    def _scan(self):
        root = Path(self.artifact_dir)
        for jp in sorted(root.rglob("analysis.json")):
            kp = jp.parent / "keypoints.json"
            masks_dir = jp.parent / "masks"
            self.samples.append(
                {
                    "artifact": str(jp),
                    "keypoints": str(kp) if kp.exists() else "",
                    "masks_dir": str(masks_dir) if masks_dir.exists() else "",
                }
            )
        print(f"OHPDataset: {len(self.samples)} samples from {self.artifact_dir}")

    @staticmethod
    def _mask_frame_features(mask: np.ndarray) -> list[float]:
        """Compact per-frame mask descriptors used alongside keypoints."""
        h, w = mask.shape[:2]
        ys, xs = np.where(mask.astype(bool))
        if xs.size == 0:
            return [0.0] * 8

        x_c = float(np.mean(xs) / max(w, 1))
        y_c = float(np.mean(ys) / max(h, 1))
        x_min = float(np.min(xs) / max(w, 1))
        x_max = float(np.max(xs) / max(w, 1))
        y_min = float(np.min(ys) / max(h, 1))
        y_max = float(np.max(ys) / max(h, 1))
        area = float(xs.size / max(h * w, 1))
        spread_y = float(np.std(ys) / max(h, 1))
        return [x_c, y_c, x_max - x_min, y_max - y_min, area, spread_y, y_min, y_max]

    # -- helpers for stratified split --
    def expert_labels_array(self) -> np.ndarray:
        """Return (N,) int array of expert labels for stratification."""
        labels = []
        for s in self.samples:
            with open(s["artifact"]) as f:
                d = json.load(f)
            expert = d.get("expert", None)
            if expert is None:
                grade = (
                    d.get("wave_features", {})
                    .get("quality", {})
                    .get("grade", "")
                )
                expert = str(grade).upper() == "A"
            labels.append(int(bool(expert)))
        return np.array(labels)

    def subject_ids_array(self) -> np.ndarray:
        """Return inferred subject IDs for group-aware splitting."""
        ids = []
        for s in self.samples:
            with open(s["artifact"], "r", encoding="utf-8") as f:
                d = json.load(f)
            sid = (
                d.get("subject_id")
                or d.get("subject")
                or d.get("user_id")
                or d.get("lifter_id")
            )
            if sid is None:
                v = str(d.get("video", ""))
                stem = Path(v).stem if v else Path(s["artifact"]).parent.name
                m = re.match(r"(.+)_\d+$", stem)
                sid = m.group(1) if m else stem
            ids.append(str(sid))
        return np.array(ids, dtype=object)

    def label_stats(self, indices: list[int] | None = None) -> dict:
        """Compute class distribution stats for imbalance diagnostics."""
        idxs = indices if indices is not None else list(range(len(self.samples)))
        n = len(idxs)
        fault_pos = np.zeros(NUM_FAULTS, dtype=np.float64)
        expert_pos = 0.0
        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0, "UNK": 0}

        for i in idxs:
            with open(self.samples[i]["artifact"], "r", encoding="utf-8") as f:
                art = json.load(f)
            ff = art.get("fault_flags", {})
            for j, name in enumerate(FAULT_NAMES):
                fault_pos[j] += float(bool(ff.get(name, False)))

            expert = art.get("expert", None)
            if expert is None:
                grade = str(
                    art.get("wave_features", {}).get("quality", {}).get("grade", "")
                ).upper()
                expert = grade == "A"
            expert_pos += float(bool(expert))

            g = str(art.get("wave_features", {}).get("quality", {}).get("grade", "")).upper()
            if g in grades:
                grades[g] += 1
            else:
                grades["UNK"] += 1

        fault_rate = fault_pos / max(n, 1)
        return {
            "n": int(n),
            "fault_pos": fault_pos,
            "fault_rate": fault_rate,
            "expert_pos": float(expert_pos),
            "expert_rate": float(expert_pos / max(n, 1)),
            "grades": grades,
        }

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _grade_to_targets(grade: str) -> tuple[float, float, dict[str, bool]]:
        """Map grade bands to expert target, sample weight, and weak fault targets.

        A/B -> good, C -> okay, D/F -> bad.
        """
        tier = GRADE_TO_TIER.get(str(grade).upper(), "okay")
        if tier == "good":
            return 1.0, 1.25, {name: False for name in FAULT_NAMES}
        if tier == "okay":
            weak = {name: False for name in FAULT_NAMES}
            if "unstable_lockout" in weak:
                weak["unstable_lockout"] = True
            return 0.0, 0.5, weak

        weak = {name: False for name in FAULT_NAMES}
        for k in ("trunk_shift_under_load", "hip_shift_compensation", "unstable_lockout"):
            if k in weak:
                weak[k] = True
        return 0.0, 1.5, weak

    def __getitem__(self, idx):
        s = self.samples[idx]

        with open(s["artifact"]) as f:
            artifact = json.load(f)

        masks_dir = s.get("masks_dir", "")
        n_mask_frames = int(artifact.get("n_frames", 0)) if masks_dir else 0

        kp_path = s.get("keypoints", "")
        if kp_path and os.path.exists(kp_path):
            with open(kp_path) as f:
                kp_data = json.load(f)
        else:
            # Mask-only artifacts may not include keypoints.json.
            kp_data = [
                {"frame_idx": i, "keypoints": {}}
                for i in range(int(artifact.get("n_frames", 0)))
            ]

        # ── pose features (B, T, 85) ──
        from src.model.exercise_vision_adapter import (
            build_harmonic_signals_from_artifact,
        )
        try:
            from src.cv.pose_estimator import KEYPOINT_NAMES
        except Exception:
            KEYPOINT_NAMES = [
                "nose",
                "left_eye_inner",
                "left_eye",
                "left_eye_outer",
                "right_eye_inner",
                "right_eye",
                "right_eye_outer",
                "left_ear",
                "right_ear",
                "mouth_left",
                "mouth_right",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_pinky",
                "right_pinky",
                "left_index",
                "right_index",
                "left_thumb",
                "right_thumb",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_heel",
                "right_heel",
                "left_foot_index",
                "right_foot_index",
            ]

        n_frames = max(len(kp_data), n_mask_frames)
        pose_features = np.zeros((self.max_frames, 85), dtype=np.float32)
        if n_frames > 0:
            indices = np.linspace(
                0, n_frames - 1, min(self.max_frames, n_frames), dtype=int
            )
            for i, fi in enumerate(indices):
                kp_idx = min(fi, max(len(kp_data) - 1, 0))
                kps = kp_data[kp_idx]["keypoints"] if len(kp_data) > 0 else {}
                kp_arr = []
                for name in KEYPOINT_NAMES:
                    kp = kps.get(name, {"x": None, "y": None})
                    x = kp["x"] if kp["x"] is not None else 0.0
                    y = kp["y"] if kp["y"] is not None else 0.0
                    kp_arr.extend([x / 720.0, y / 720.0])

                mask_feats = [0.0] * 8
                if masks_dir:
                    mp = os.path.join(masks_dir, f"frame_{fi:06d}.npy")
                    if os.path.exists(mp):
                        try:
                            mask = np.load(mp)
                            mask_feats = self._mask_frame_features(mask)
                        except Exception:
                            mask_feats = [0.0] * 8

                feat = (
                    kp_arr
                    + mask_feats[:6]
                    + [0.0] * 6
                    + [mask_feats[6], mask_feats[7], 0.0, 1.0]
                    + [0.0, 0.0, 0.0]
                )
                pose_features[i, : len(feat)] = feat[:85]

        # ── harmonic signals (4, signal_len) ──
        harmonic = build_harmonic_signals_from_artifact(artifact, self.signal_len)

        # ── scalar features ──
        quality = artifact.get("wave_features", {}).get("quality", {})
        depth = artifact.get("depth_features", {})
        scalars = []
        for k in QUALITY_KEYS:
            scalars.append(float(quality.get(k, 0.0)))
        for k in DEPTH_KEYS:
            scalars.append(float(depth.get(k, 0.0)))
        scalar_features = np.array(scalars, dtype=np.float32)

        # ── labels ──
        quality_score = float(quality.get("overall", 0.0))

        fault_flags = artifact.get("fault_flags", {})
        grade = (
            artifact.get("wave_features", {})
            .get("quality", {})
            .get("grade", "")
        )
        inferred_expert, expert_weight, weak_faults = self._grade_to_targets(str(grade).upper())
        if not any(bool(fault_flags.get(n, False)) for n in FAULT_NAMES):
            # Artifacts currently have near-all-zero fault flags; use weak grade-based supervision.
            fault_flags = weak_faults
        fault_labels = np.array(
            [float(fault_flags.get(n, False)) for n in FAULT_NAMES],
            dtype=np.float32,
        )

        expert_raw = artifact.get("expert", None)
        if expert_raw is None:
            expert_raw = inferred_expert
        expert = float(bool(expert_raw))

        return {
            "pose_features": torch.from_numpy(pose_features),
            "harmonic_signals": torch.from_numpy(harmonic),
            "scalar_features": torch.from_numpy(scalar_features),
            "quality_label": torch.tensor(quality_score, dtype=torch.float32),
            "fault_labels": torch.from_numpy(fault_labels),
            "expert_label": torch.tensor(expert, dtype=torch.float32),
            "expert_weight": torch.tensor(float(expert_weight), dtype=torch.float32),
        }


def make_splits(dataset: OHPDataset, val_ratio: float = 0.2, seed: int = 42):
    """Group-aware train/val split to reduce subject leakage."""
    labels = dataset.expert_labels_array()
    groups = dataset.subject_ids_array()
    n = len(labels)
    if n < 2:
        return Subset(dataset, np.arange(n)), Subset(dataset, np.arange(n))

    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        n_splits = 5 if unique_groups.size >= 5 else int(unique_groups.size)
        gkf = GroupKFold(n_splits=max(2, n_splits))
        train_idx, val_idx = next(gkf.split(np.zeros(n), labels, groups=groups))
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    print("[warn] Subject grouping unavailable (single group); falling back to stratified split.")

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ── Evaluation helpers ───────────────────────────────────────────────────────


def evaluate_predictions(
    q_preds: np.ndarray,
    q_labels: np.ndarray,
    f_preds: np.ndarray,
    f_labels: np.ndarray,
    e_preds: np.ndarray,
    e_labels: np.ndarray,
) -> dict:
    """Compute all comparison metrics for console/txt logging."""
    m = {}

    # Quality regression
    m["quality/mae"] = mean_absolute_error(q_labels, q_preds)
    m["quality/rmse"] = float(np.sqrt(mean_squared_error(q_labels, q_preds)))
    m["quality/r2"] = r2_score(q_labels, q_preds) if len(set(q_labels)) > 1 else 0.0

    # Fault detection (per-fault + macro)
    f_pred_bin = (f_preds > 0.5).astype(float)
    m["fault/accuracy"] = accuracy_score(f_labels.ravel(), f_pred_bin.ravel())
    for avg in ("macro", "weighted"):
        m[f"fault/precision_{avg}"] = precision_score(
            f_labels, f_pred_bin, average=avg, zero_division=0
        )
        m[f"fault/recall_{avg}"] = recall_score(
            f_labels, f_pred_bin, average=avg, zero_division=0
        )
        m[f"fault/f1_{avg}"] = f1_score(
            f_labels, f_pred_bin, average=avg, zero_division=0
        )

    # Per-fault F1
    per_fault_f1 = f1_score(f_labels, f_pred_bin, average=None, zero_division=0)
    for i, name in enumerate(FAULT_NAMES):
        m[f"fault_f1/{name}"] = float(per_fault_f1[i])

    # Expert classification
    e_pred_bin = (e_preds > 0.5).astype(float)
    m["expert/accuracy"] = accuracy_score(e_labels, e_pred_bin)
    m["expert/precision"] = precision_score(e_labels, e_pred_bin, zero_division=0)
    m["expert/recall"] = recall_score(e_labels, e_pred_bin, zero_division=0)
    m["expert/f1"] = f1_score(e_labels, e_pred_bin, zero_division=0)
    if len(set(e_labels)) > 1:
        m["expert/auc_roc"] = roc_auc_score(e_labels, e_preds)
    else:
        m["expert/auc_roc"] = 0.0

    return m


# ═════════════════════════════════════════════════════════════════════════════
# MODEL A — Qwen-VL + ExerciseAwareAdapter
# ═════════════════════════════════════════════════════════════════════════════


class VisionAdapterTrainer:
    """Train the ExerciseAwareAdapter + quality / fault / expert heads."""

    def __init__(
        self,
        d_model: int = 896,
        lr: float = 1e-4,
        device: str = "auto",
        fault_pos_weight: torch.Tensor | None = None,
        expert_pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        adapter_dropout: float = 0.5,
        max_epochs: int = 30,
    ):
        from src.model.exercise_vision_adapter import ExerciseAwareAdapter

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.d_model = d_model
        self.adapter = ExerciseAwareAdapter(d_model=d_model, dropout=adapter_dropout).to(self.device)

        self.quality_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)
        self.fault_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_FAULTS),
        ).to(self.device)
        self.expert_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        ).to(self.device)

        params = (
            list(self.adapter.parameters())
            + list(self.quality_head.parameters())
            + list(self.fault_head.parameters())
            + list(self.expert_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, int(max_epochs)), eta_min=max(lr * 0.05, 1e-6)
        )
        self.quality_loss_fn = nn.MSELoss()
        if fault_pos_weight is None:
            fault_pos_weight = torch.ones(NUM_FAULTS, dtype=torch.float32)
        self.fault_pos_weight = fault_pos_weight.to(self.device)
        self.expert_pos_weight = torch.tensor([expert_pos_weight], dtype=torch.float32, device=self.device)
        self.focal_gamma = float(focal_gamma)
        self.fault_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.fault_pos_weight)
        self.expert_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.expert_pos_weight)

        n_params = self.adapter.num_trainable_params
        n_heads = (
            sum(p.numel() for p in self.quality_head.parameters())
            + sum(p.numel() for p in self.fault_head.parameters())
            + sum(p.numel() for p in self.expert_head.parameters())
        )
        print(
            f"[Model A] Adapter: {n_params:,} params  |  Heads: {n_heads:,} params  |  Device: {self.device}"
        )
        print(
            f"[Model A] imbalance handling: focal_gamma={self.focal_gamma:.2f} "
            f"expert_pos_weight={float(self.expert_pos_weight.item()):.3f}"
        )

    def _focal_factor(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        return (1.0 - pt).clamp_min(1e-6).pow(self.focal_gamma)

    def _forward(self, batch):
        pose = batch["pose_features"].to(self.device)
        harm = batch["harmonic_signals"].to(self.device)
        B = pose.shape[0]
        dummy_visual = torch.randn(B, 16, self.d_model, device=self.device)
        fused = self.adapter(dummy_visual, pose, harm)
        pooled = fused.mean(dim=1)  # (B, D)
        q_pred = self.quality_head(pooled).squeeze(-1)  # (B,)
        f_pred = self.fault_head(pooled)  # (B, 9)
        e_pred = self.expert_head(pooled).squeeze(-1)  # (B,)
        return q_pred, f_pred, e_pred

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        self.adapter.train(train)
        self.quality_head.train(train)
        self.fault_head.train(train)
        self.expert_head.train(train)

        total_loss = 0.0
        n = 0
        q_all, ql_all, f_all, fl_all, e_all, el_all = [], [], [], [], [], []

        ctx = torch.no_grad() if not train else torch.enable_grad()
        with ctx:
            for batch in loader:
                q_pred, f_pred, e_pred = self._forward(batch)
                q_lab = batch["quality_label"].to(self.device)
                f_lab = batch["fault_labels"].to(self.device)
                e_lab = batch["expert_label"].to(self.device)
                e_w = batch.get("expert_weight", torch.ones_like(e_lab)).to(self.device)

                q_loss = self.quality_loss_fn(q_pred, q_lab)
                f_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    f_pred,
                    f_lab,
                    reduction="none",
                    pos_weight=self.fault_pos_weight,
                )
                e_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    e_pred,
                    e_lab,
                    reduction="none",
                    pos_weight=self.expert_pos_weight,
                )
                f_loss = (f_bce * self._focal_factor(f_pred, f_lab)).mean()
                e_loss = (e_bce * self._focal_factor(e_pred, e_lab) * e_w).mean()
                reg_loss = torch.tensor(0.0, device=self.device)
                for p in self.adapter.parameters():
                    reg_loss = reg_loss + p.norm(2)
                loss = (q_loss * 20.0) + f_loss + (e_loss * 2.0) + (1e-4 * reg_loss)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                    self.optimizer.step()

                B = q_pred.size(0)
                total_loss += loss.item() * B
                n += B
                q_all.append(q_pred.detach().cpu().numpy())
                ql_all.append(q_lab.detach().cpu().numpy())
                f_all.append(torch.sigmoid(f_pred).detach().cpu().numpy())
                fl_all.append(f_lab.detach().cpu().numpy())
                e_all.append(torch.sigmoid(e_pred).detach().cpu().numpy())
                el_all.append(e_lab.detach().cpu().numpy())

        q_preds = np.concatenate(q_all)
        q_labels = np.concatenate(ql_all)
        f_preds = np.concatenate(f_all)
        f_labels = np.concatenate(fl_all)
        e_preds = np.concatenate(e_all)
        e_labels = np.concatenate(el_all)

        metrics = evaluate_predictions(
            q_preds, q_labels, f_preds, f_labels, e_preds, e_labels
        )
        metrics["loss"] = total_loss / max(n, 1)
        metrics["adapter_gate"] = float(torch.sigmoid(self.adapter.gate).item())
        return metrics, (q_preds, q_labels, f_preds, f_labels, e_preds, e_labels)

    def train_epoch(self, loader):
        return self._run_epoch(loader, train=True)

    def eval_epoch(self, loader):
        return self._run_epoch(loader, train=False)

    def step_scheduler(self):
        self.scheduler.step()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "quality_head": self.quality_head.state_dict(),
                "fault_head": self.fault_head.state_dict(),
                "expert_head": self.expert_head.state_dict(),
            },
            path,
        )
        print(f"Saved adapter checkpoint -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Language LoRA fine-tuning (Stage 2 for Model A)
# ═════════════════════════════════════════════════════════════════════════════


class InstructionDataset(Dataset):
    """Build instruction-tuning pairs on the fly from analysis.json artifacts."""

    def __init__(
        self,
        artifact_dir: str,
        tokenizer,
        max_length: int = 512,
        manual_pdf: str = "",
        include_artifacts: bool = True,
        include_manual: bool = True,
    ):
        self.examples: list[dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        if include_artifacts:
            self._build_from_artifacts(artifact_dir)
        if include_manual:
            self._build_from_manual_pdf(manual_pdf)
        print(f"InstructionDataset: {len(self.examples)} examples")

    def _build_from_artifacts(self, artifact_dir: str):
        def _synth_feedback(quality: dict, active_faults: list[str]) -> str:
            grade = quality.get("grade", "?")
            overall = float(quality.get("overall", 0.0))
            if active_faults:
                faults_txt = ", ".join(active_faults)
                return (
                    f"Current grade {grade} ({overall:.2f}). Prioritize fixing: {faults_txt}. "
                    "Keep torso stable, maintain symmetrical arm path, and use controlled tempo through lockout and descent."
                )
            return (
                f"Current grade {grade} ({overall:.2f}). Form is relatively stable; "
                "focus on smoother tempo, consistent arm drive, and balanced lockout to improve efficiency."
            )

        for jp in sorted(Path(artifact_dir).rglob("analysis.json")):
            try:
                with open(jp) as f:
                    art = json.load(f)
            except Exception:
                continue

            quality = art.get("wave_features", {}).get("quality", {})
            faults = art.get("fault_flags", {})
            lang = art.get("language", {})
            feedback = lang.get("coach_feedback", "")
            active = [k for k, v in faults.items() if v]
            if not feedback:
                feedback = _synth_feedback(quality, active)
            metrics_text = (
                f"Quality grade: {quality.get('grade', '?')}, "
                f"Overall: {quality.get('overall', 0):.2f}, "
                f"Smoothness: {quality.get('smoothness', 0):.2f}, "
                f"Control: {quality.get('control', 0):.2f}, "
                f"Symmetry: {quality.get('symmetry', 0):.2f}. "
                f"Expert: {'Yes' if art.get('expert') else 'No'}. "
                f"Active faults: {', '.join(active) if active else 'none'}."
            )

            self.examples.append(
                {
                    "instruction": "Based on the biomechanical analysis of this overhead press, provide coaching feedback.",
                    "input": metrics_text,
                    "output": feedback,
                }
            )

            if active:
                self.examples.append(
                    {
                        "instruction": f"These form faults were detected: {', '.join(active)}. Explain corrections.",
                        "input": metrics_text,
                        "output": feedback,
                    }
                )

    def _build_from_manual_pdf(self, manual_pdf: str):
        manual_text = _extract_pdf_text(manual_pdf)
        if not manual_text:
            return

        chunks = _chunk_text(manual_text, chunk_chars=500)
        if not chunks:
            return

        for i, chunk in enumerate(chunks, start=1):
            self.examples.append(
                {
                    "instruction": (
                        "Using this strength and conditioning manual excerpt, write concise overhead press coaching guidance "
                        "using professional cue language such as 'Punch the ceiling', 'Active shoulders', "
                        "'Ribs down', and 'Brace before press'."
                    ),
                    "input": f"Manual excerpt {i}/{len(chunks)}:\n{chunk}",
                    "output": (
                        "Apply these principles to overhead press coaching: brace before press, keep ribs down, "
                        "drive a vertical bar path, keep active shoulders, and cue 'punch the ceiling' at lockout "
                        "while maintaining controlled tempo and symmetric arm drive."
                    ),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
        full = prompt + ex["output"]
        enc = self.tokenizer(
            full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        prompt_enc = self.tokenizer(prompt, max_length=self.max_length, truncation=True)
        prompt_len = len(prompt_enc["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Prevent all-masked labels, which can produce NaN CE loss.
        if torch.all(labels == -100):
            last_idx = int(attn_mask.sum().item()) - 1
            if last_idx < 0:
                last_idx = self.max_length - 1
            labels[last_idx] = input_ids[last_idx]
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def train_language_lora(
    artifact_dir: str,
    model_id: str = "Qwen/Qwen3-VL-2B-Thinking",
    output_dir: str = "checkpoints/language_lora",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    run_name: str = "language_lora",
    max_steps: int = 0,
    manual_pdf: str = "",
    language_data_mode: str = "manual",
    lang_max_length: int = 768,
) -> dict:
    """LoRA fine-tune Qwen3-VL on coaching instruction pairs."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _init_run_log(
        run_name=run_name,
        config=dict(
            model_id=model_id,
            epochs=epochs,
            bs=batch_size,
            lr=lr,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=str(device),
            manual_pdf=manual_pdf,
            lang_max_length=lang_max_length,
        ),
        tags=["language", "lora", "qwen-vl"],
    )

    print(f"Loading {model_id} ...")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    tokenizer = None
    last_err = None
    for attempt in range(1, 4):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=(os.environ.get("HF_TOKEN") or None),
            )
            break
        except Exception as e:
            last_err = e
            print(f"[warn] Tokenizer load attempt {attempt}/3 failed: {e}")
            time.sleep(3 * attempt)
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer for {model_id}: {last_err}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        dtype = torch.bfloat16 if use_bf16 else torch.float32
    else:
        dtype = torch.float32
    model = None
    last_err = None
    for attempt in range(1, 4):
        try:
            if "vl" in model_id.lower():
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    dtype=dtype,
                    token=(os.environ.get("HF_TOKEN") or None),
                ).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    dtype=dtype,
                    token=(os.environ.get("HF_TOKEN") or None),
                ).to(device)
            break
        except Exception as e:
            last_err = e
            print(f"[warn] Model load attempt {attempt}/3 failed: {e}")
            time.sleep(5 * attempt)
    if model is None:
        raise RuntimeError(f"Failed to load model {model_id}: {last_err}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    try:
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    except ModuleNotFoundError as e:
        if "neural_compressor" not in str(e):
            raise
        print("[warn] PEFT LoRA backend requires neural_compressor; falling back to lm_head-only fine-tune.")
        for p in model.parameters():
            p.requires_grad = False
        n_trainable = 0
        for name, p in model.named_parameters():
            if "lm_head" in name:
                p.requires_grad = True
                n_trainable += p.numel()
        if n_trainable == 0:
            # Last-resort tiny fallback: unfreeze final decoder block norms/biases.
            layer_ids = []
            for name, _ in model.named_parameters():
                m = re.search(r"layers\.(\d+)\.", name)
                if m:
                    layer_ids.append(int(m.group(1)))
            if layer_ids:
                last_layer = max(layer_ids)
                key = f"layers.{last_layer}."
                for name, p in model.named_parameters():
                    if key in name and (name.endswith("bias") or "norm" in name.lower()):
                        p.requires_grad = True
                        n_trainable += p.numel()
        if n_trainable == 0:
            # Absolute fallback: at least one trainable parameter.
            for _, p in model.named_parameters():
                p.requires_grad = True
                n_trainable += p.numel()
                break
        print(f"[lang] fallback trainable params: {n_trainable:,}")

    include_artifacts = language_data_mode in ("artifacts", "both")
    include_manual = language_data_mode in ("manual", "both")

    dataset = InstructionDataset(
        artifact_dir,
        tokenizer,
        max_length=int(lang_max_length),
        manual_pdf=manual_pdf,
        include_artifacts=include_artifacts,
        include_manual=include_manual,
    )
    if len(dataset) == 0:
        print("[warn] Language dataset is empty. Skipping language LoRA stage.")
        return {
            "lang/final_loss": float("nan"),
            "lang/best_loss": float("nan"),
            "lang/final_ppl": float("nan"),
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    amp_enabled = use_bf16

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, n_tok, step = 0.0, 0, 0
        for batch in tqdm(loader, desc=f"Lang Epoch {epoch+1}/{epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.amp.autocast("cuda", enabled=True, dtype=torch.float16)
                if amp_enabled
                else contextlib.nullcontext()
            )
            with amp_ctx:
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = out.loss
            if not torch.isfinite(loss):
                print("  [warn] non-finite language loss; skipping this batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                print("  [warn] non-finite grad norm; skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            total_loss += loss.item() * ids.size(0)
            n_tok += ids.size(0)
            step += 1
            if step % 10 == 0:
                print(
                    f"  [lang] step={step} loss={loss.item():.4f} grad_norm={float(gn):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}"
                )
            if max_steps > 0 and step >= max_steps:
                break

        scheduler.step()
        avg = total_loss / max(n_tok, 1)
        ppl = math.exp(min(avg, 20))
        print(f"  Epoch {epoch+1}: loss={avg:.4f}  ppl={ppl:.2f}")
        if avg < best_loss:
            best_loss = avg
        if max_steps > 0 and step >= max_steps:
            print(f"  Early stop after {step} steps (max_steps={max_steps}).")
            break

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA -> {output_dir}")

    summary = {
        "lang/final_loss": avg,
        "lang/best_loss": best_loss,
        "lang/final_ppl": ppl,
    }
    print(f"[lang] summary={summary}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Training loop driver
# ═════════════════════════════════════════════════════════════════════════════


def _train_model(trainer, train_loader, val_loader, epochs, prefix):
    """Generic epoch loop for adapter training."""
    best_val_score = -1.0
    best_val_metrics = {}
    best_unstable_f1 = -1.0

    for epoch in range(epochs):
        t0 = time.time()
        tr_m, _ = trainer.train_epoch(train_loader)
        va_m, val_preds = trainer.eval_epoch(val_loader)
        dt = time.time() - t0

        log = {"epoch": epoch + 1}
        for k, v in tr_m.items():
            log[f"{prefix}/train/{k}"] = v
        for k, v in va_m.items():
            log[f"{prefix}/val/{k}"] = v
        print(f"[metrics] {json.dumps(_to_json_safe(log), sort_keys=True)}")

        print(
            f"  [{prefix}] Epoch {epoch+1:3d}/{epochs} ({dt:.1f}s) | "
            f"train loss={tr_m['loss']:.4f}  val loss={va_m['loss']:.4f} | "
            f"val q_mae={va_m['quality/mae']:.4f}  f_f1={va_m['fault/f1_macro']:.3f}  "
            f"expert_f1={va_m['expert/f1']:.3f}"
        )
        unstable_key = "fault_f1/unstable_lockout"
        unstable_f1 = float(va_m.get(unstable_key, 0.0))
        print(f"  [anchor] val unstable_lockout_f1={unstable_f1:.3f}")
        if unstable_f1 > best_unstable_f1:
            best_unstable_f1 = unstable_f1
        elif best_unstable_f1 >= 0 and unstable_f1 < (best_unstable_f1 - 0.05):
            print(
                f"  [warn] unstable_lockout anchor dropped from {best_unstable_f1:.3f} to {unstable_f1:.3f}"
            )

        f1_score_obj = 0.7 * float(va_m.get("fault/f1_macro", 0.0)) + 0.3 * float(
            va_m.get("expert/f1", 0.0)
        )
        if f1_score_obj > best_val_score:
            best_val_score = f1_score_obj
            best_val_metrics = {f"{prefix}/best_{k}": v for k, v in va_m.items()}
            best_val_metrics[f"{prefix}/best_f1_priority_score"] = best_val_score

        if float(va_m.get("fault/f1_macro", 0.0)) == 0.0 or float(va_m.get("expert/f1", 0.0)) == 0.0:
            print("  [warn] Zero-F1 detected. Accuracy may be misleading under class imbalance.")

        if hasattr(trainer, "step_scheduler"):
            trainer.step_scheduler()

    return best_val_metrics, val_preds


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL adapter pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["adapter", "qwen_plain", "all"],
        default="all",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing training artifacts (analysis.json files)",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lang_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument(
        "--manual_pdf",
        default="/scratch/jnolas77/fitness/capstone/ml/basics_of_strength_and_conditioning_manual.pdf",
        help="Path to S&C manual PDF used for language LoRA examples",
    )
    parser.add_argument(
        "--language_data_mode",
        choices=["manual", "artifacts", "both"],
        default="manual",
        help="Language training examples source: manual text only, artifact examples only, or both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lang_steps", type=int, default=0)
    parser.add_argument(
        "--lang_max_length",
        type=int,
        default=768,
        help="Max token length for language LoRA training examples",
    )
    parser.add_argument(
        "--disable_oversampling",
        action="store_true",
        help="Disable weighted oversampling of minority positive classes",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=1.5,
        help="Focal modulation gamma for fault/expert losses",
    )
    parser.add_argument(
        "--max_pos_weight",
        type=float,
        default=15.0,
        help="Upper bound for automatic positive-class weights",
    )
    parser.add_argument(
        "--expert_pos_weight",
        type=float,
        default=0.0,
        help="Override expert positive weight (0 = auto)",
    )
    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.5,
        help="Dropout inside exercise adapter layers",
    )
    parser.add_argument(
        "--log_txt", default="", help="Path to plain text training log file"
    )
    args = parser.parse_args()

    log_fh = None
    if args.log_txt:
        log_fh = _enable_txt_logging(args.log_txt)
        print(f"[log] Writing training logs to: {args.log_txt}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log = os.path.join(args.output_dir, "logs", f"train_{ts}.txt")
        log_fh = _enable_txt_logging(default_log)
        print(f"[log] Writing training logs to: {default_log}")

    run_adapter = args.stage in ("adapter", "all")
    run_qwen_plain = args.stage == "qwen_plain"
    run_language = True

    # ── Data ──
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    dataset = OHPDataset(args.data_dir, max_frames=16, signal_len=128)
    train_set, val_set = make_splits(dataset, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train: {len(train_set)}  |  Val: {len(val_set)}")

    train_stats = dataset.label_stats(indices=list(train_set.indices))
    print(f"[data] grade distribution (train): {train_stats['grades']}")
    print(f"[data] expert positive rate (train): {train_stats['expert_rate']:.4f}")
    print(
        "[data] per-fault positive rates (train): "
        + ", ".join(
            f"{name}={float(rate):.4f}"
            for name, rate in zip(FAULT_NAMES, train_stats["fault_rate"])
        )
    )

    fault_pos_rate = np.asarray(train_stats["fault_rate"], dtype=np.float32)
    fault_pos_weight = (1.0 - fault_pos_rate) / np.clip(fault_pos_rate, 1e-4, 1.0)
    fault_pos_weight = np.clip(fault_pos_weight, 1.0, float(args.max_pos_weight))
    expert_rate = float(train_stats["expert_rate"])
    expert_pos_weight_auto = float(
        np.clip((1.0 - expert_rate) / max(expert_rate, 1e-4), 1.0, float(args.max_pos_weight))
    )
    expert_pos_weight = float(args.expert_pos_weight) if float(args.expert_pos_weight) > 0 else expert_pos_weight_auto

    print(
        "[data] pos_weight fault: "
        + ", ".join(f"{name}={w:.2f}" for name, w in zip(FAULT_NAMES, fault_pos_weight))
    )
    print(f"[data] pos_weight expert: {expert_pos_weight:.2f}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    print("[data] oversampling disabled (loss weighting only)")

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    adapter_best = {}
    plain_summary = {}
    lang_summary = {}

    # ── Model A: Qwen-VL + Adapter ──
    if run_adapter:
        print("\n" + "=" * 70)
        print("MODEL A: Qwen-VL + ExerciseAwareAdapter")
        print("=" * 70)
        _init_run_log(
            run_name="model_a_qwen_vl_adapter",
            config={
                "model": "qwen_vl_adapter",
                "d_model": 896,
                "lr": args.lr,
                "epochs": args.epochs,
                "bs": args.batch_size,
                "n_train": len(train_set),
                "n_val": len(val_set),
            },
            tags=["model_a", "qwen-vl", "adapter", "vision"],
        )
        trainer_a = VisionAdapterTrainer(
            d_model=896,
            lr=args.lr,
            fault_pos_weight=torch.tensor(fault_pos_weight, dtype=torch.float32),
            expert_pos_weight=expert_pos_weight,
            focal_gamma=args.focal_gamma,
            adapter_dropout=args.adapter_dropout,
            max_epochs=args.epochs,
        )
        adapter_best, _ = _train_model(
            trainer_a,
            train_loader,
            val_loader,
            args.epochs,
            "adapter",
        )
        trainer_a.save(os.path.join(args.output_dir, "vision_adapter.pt"))
        print(f"[adapter] best_metrics={adapter_best}")

    # ── Language LoRA (Qwen-VL only) ──
    if run_language:
        print("\n" + "=" * 70)
        print("STAGE 2: Language LoRA (Qwen-VL coaching text)")
        print("=" * 70)
        lang_summary = train_language_lora(
            artifact_dir=args.data_dir,
            model_id=args.model_id,
            output_dir=os.path.join(args.output_dir, "language_lora"),
            epochs=args.lang_epochs,
            batch_size=max(args.batch_size // 2, 1),
            lr=args.lr * 2,
            max_steps=args.max_lang_steps,
            manual_pdf=args.manual_pdf,
            language_data_mode=args.language_data_mode,
            lang_max_length=args.lang_max_length,
        )

    # ── Qwen3-VL plain (no adapter) ──
    if run_qwen_plain:
        print("\n" + "=" * 70)
        print("MODEL B: Qwen3-VL Plain (NO ADAPTER)")
        print("=" * 70)
        plain_summary = train_language_lora(
            artifact_dir=args.data_dir,
            model_id=args.model_id,
            output_dir=os.path.join(args.output_dir, "language_lora_plain"),
            epochs=args.lang_epochs,
            batch_size=max(args.batch_size // 2, 1),
            lr=args.lr * 2,
            run_name="model_b_qwen3_vl_plain",
            max_steps=args.max_lang_steps,
            manual_pdf=args.manual_pdf,
            language_data_mode=args.language_data_mode,
            lang_max_length=args.lang_max_length,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    if run_adapter:
        print(f"  Adapter checkpoint -> {args.output_dir}/vision_adapter.pt")
    if run_language:
        print(f"  LoRA weights       -> {args.output_dir}/language_lora/")
    if run_qwen_plain:
        print(f"  Plain Qwen LoRA    -> {args.output_dir}/language_lora_plain/")


if __name__ == "__main__":
    main()
