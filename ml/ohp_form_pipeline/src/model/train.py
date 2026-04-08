"""
Training pipeline — Qwen3-VL + ExerciseAwareAdapter vs. Baseline MLP.

Two independent model families are trained and compared:

  Model A  — Qwen-VL + Adapter:
    Stage 1: ExerciseAwareAdapter (pose+harmonic cross-attention) with quality,
             fault, and expert prediction heads.
    Stage 2: LoRA fine-tune of Qwen3-VL language model for coaching text.

    Model B  — Qwen3-VL Plain:
        Qwen/Qwen3-VL-2B-Thinking without the ExerciseAware adapter.
        Used as the plain-model comparison against Model A.

Both pipelines log to Weights & Biases (wandb) with a shared comparison run
at the end that reports:
  - Quality regression:  MAE, RMSE, R²
  - Fault detection:     per-fault + macro Accuracy, Precision, Recall, F1
  - Expert classification: Accuracy, Precision, Recall, F1, AUC-ROC

Usage:
    cd ml/ohp_form_pipeline
    python -m src.model.train --stage all --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
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
_hf_token = os.environ.get("HF_TOKEN", "")
if _hf_token:
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
    "bar_tilt_instability",
    "lateral_bar_drift",
    "uneven_press_timing",
    "compensatory_lateral_shift",
    "trunk_shift_under_load",
    "hip_shift_compensation",
    "unstable_lockout",
    "forward_bar_drift_depth_proxy",
]
NUM_FAULTS = len(FAULT_NAMES)

# Quality sub-score keys used as extra scalar features
QUALITY_KEYS = ["smoothness", "control", "efficiency", "consistency", "symmetry"]
DEPTH_KEYS = [
    "bar_forward_drift_depth",
    "bar_depth_asymmetry",
    "torso_depth_shift",
    "subject_depth_stability",
]


# ── wandb helpers ────────────────────────────────────────────────────────────


def _init_wandb(run_name: str, config: dict, tags: list[str] | None = None):
    import wandb

    return wandb.init(
        entity="jnolas77-arizona-state-university",
        project="capstone",
        name=run_name,
        config=config,
        tags=tags or [],
        reinit=True,
        mode="offline",
    )


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


def _enable_txt_logging(log_path: str):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _StreamTee(sys.__stdout__, fh)
    sys.stderr = _StreamTee(sys.__stderr__, fh)
    return fh


# ── Dataset ──────────────────────────────────────────────────────────────────


class OHPDataset(Dataset):
    """Unified dataset that loads every useful signal from batch_outputs.

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
            if kp.exists():
                self.samples.append({"artifact": str(jp), "keypoints": str(kp)})
        print(f"OHPDataset: {len(self.samples)} samples from {self.artifact_dir}")

    # -- helpers for stratified split --
    def expert_labels_array(self) -> np.ndarray:
        """Return (N,) int array of expert labels for stratification."""
        labels = []
        for s in self.samples:
            with open(s["artifact"]) as f:
                d = json.load(f)
            labels.append(int(d.get("expert", False)))
        return np.array(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        with open(s["artifact"]) as f:
            artifact = json.load(f)
        with open(s["keypoints"]) as f:
            kp_data = json.load(f)

        # ── pose features (B, T, 85) ──
        from src.model.exercise_vision_adapter import (
            build_harmonic_signals_from_artifact,
        )
        from src.cv.pose_estimator import KEYPOINT_NAMES

        n_frames = len(kp_data)
        pose_features = np.zeros((self.max_frames, 85), dtype=np.float32)
        if n_frames > 0:
            indices = np.linspace(
                0, n_frames - 1, min(self.max_frames, n_frames), dtype=int
            )
            for i, fi in enumerate(indices):
                kps = kp_data[fi]["keypoints"]
                kp_arr = []
                for name in KEYPOINT_NAMES:
                    kp = kps.get(name, {"x": None, "y": None})
                    x = kp["x"] if kp["x"] is not None else 0.0
                    y = kp["y"] if kp["y"] is not None else 0.0
                    kp_arr.extend([x / 720.0, y / 720.0])
                feat = (
                    kp_arr
                    + [0.0] * 6
                    + [0.0] * 6
                    + [0.0, 0.0, 0.0, 1.0]
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
        fault_labels = np.array(
            [float(fault_flags.get(n, False)) for n in FAULT_NAMES],
            dtype=np.float32,
        )

        expert = float(artifact.get("expert", False))

        return {
            "pose_features": torch.from_numpy(pose_features),
            "harmonic_signals": torch.from_numpy(harmonic),
            "scalar_features": torch.from_numpy(scalar_features),
            "quality_label": torch.tensor(quality_score, dtype=torch.float32),
            "fault_labels": torch.from_numpy(fault_labels),
            "expert_label": torch.tensor(expert, dtype=torch.float32),
        }


def make_splits(dataset: OHPDataset, val_ratio: float = 0.2, seed: int = 42):
    """Stratified train/val split on expert label."""
    labels = dataset.expert_labels_array()
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
    """Compute all comparison metrics. Returns flat dict for wandb logging."""
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

    def __init__(self, d_model: int = 896, lr: float = 1e-4, device: str = "auto"):
        from src.model.exercise_vision_adapter import ExerciseAwareAdapter

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.d_model = d_model
        self.adapter = ExerciseAwareAdapter(d_model=d_model).to(self.device)

        self.quality_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(self.device)
        self.fault_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, NUM_FAULTS),
            nn.Sigmoid(),
        ).to(self.device)
        self.expert_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)

        params = (
            list(self.adapter.parameters())
            + list(self.quality_head.parameters())
            + list(self.fault_head.parameters())
            + list(self.expert_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.quality_loss_fn = nn.MSELoss()
        self.fault_loss_fn = nn.BCELoss()
        self.expert_loss_fn = nn.BCELoss()

        n_params = self.adapter.num_trainable_params
        n_heads = (
            sum(p.numel() for p in self.quality_head.parameters())
            + sum(p.numel() for p in self.fault_head.parameters())
            + sum(p.numel() for p in self.expert_head.parameters())
        )
        print(
            f"[Model A] Adapter: {n_params:,} params  |  Heads: {n_heads:,} params  |  Device: {self.device}"
        )

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

                q_loss = self.quality_loss_fn(q_pred, q_lab)
                f_loss = self.fault_loss_fn(f_pred, f_lab)
                e_loss = self.expert_loss_fn(e_pred, e_lab)
                loss = q_loss + f_loss + 0.5 * e_loss

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
                f_all.append(f_pred.detach().cpu().numpy())
                fl_all.append(f_lab.detach().cpu().numpy())
                e_all.append(e_pred.detach().cpu().numpy())
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
# MODEL B — Baseline MLP (NO Qwen-VL)
# ═════════════════════════════════════════════════════════════════════════════


class BaselineMLP(nn.Module):
    """Pure MLP on flattened structured features — no vision encoder.

    Input features:
      - pose_features flattened: max_frames * 85 = 680
      - harmonic_signals flattened: 4 * 128 = 512
      - scalar_features: len(QUALITY_KEYS) + len(DEPTH_KEYS) = 9
      Total = 1201
    """

    def __init__(
        self,
        pose_dim: int = 680,
        harm_dim: int = 512,
        scalar_dim: int = 9,
        hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = pose_dim + harm_dim + scalar_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        h2 = hidden // 2
        self.quality_head = nn.Sequential(
            nn.Linear(h2, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.fault_head = nn.Sequential(
            nn.Linear(h2, 128), nn.GELU(), nn.Linear(128, NUM_FAULTS), nn.Sigmoid()
        )
        self.expert_head = nn.Sequential(
            nn.Linear(h2, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, pose_features, harmonic_signals, scalar_features):
        B = pose_features.size(0)
        x = torch.cat(
            [
                pose_features.view(B, -1),
                harmonic_signals.view(B, -1),
                scalar_features,
            ],
            dim=-1,
        )
        h = self.encoder(x)
        q = self.quality_head(h).squeeze(-1)
        f = self.fault_head(h)
        e = self.expert_head(h).squeeze(-1)
        return q, f, e


class BaselineTrainer:
    """Train the baseline MLP."""

    def __init__(
        self,
        lr: float = 1e-3,
        device: str = "auto",
        pose_dim: int = 680,
        harm_dim: int = 512,
        scalar_dim: int = 9,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model = BaselineMLP(
            pose_dim=pose_dim,
            harm_dim=harm_dim,
            scalar_dim=scalar_dim,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        self.quality_loss_fn = nn.MSELoss()
        self.fault_loss_fn = nn.BCELoss()
        self.expert_loss_fn = nn.BCELoss()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model B] BaselineMLP: {n_params:,} params  |  Device: {self.device}")

    def _forward(self, batch):
        pose = batch["pose_features"].to(self.device)
        harm = batch["harmonic_signals"].to(self.device)
        scal = batch["scalar_features"].to(self.device)
        return self.model(pose, harm, scal)

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        self.model.train(train)
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

                q_loss = self.quality_loss_fn(q_pred, q_lab)
                f_loss = self.fault_loss_fn(f_pred, f_lab)
                e_loss = self.expert_loss_fn(e_pred, e_lab)
                loss = q_loss + f_loss + 0.5 * e_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                B = q_pred.size(0)
                total_loss += loss.item() * B
                n += B
                q_all.append(q_pred.detach().cpu().numpy())
                ql_all.append(q_lab.detach().cpu().numpy())
                f_all.append(f_pred.detach().cpu().numpy())
                fl_all.append(f_lab.detach().cpu().numpy())
                e_all.append(e_pred.detach().cpu().numpy())
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
        return metrics, (q_preds, q_labels, f_preds, f_labels, e_preds, e_labels)

    def train_epoch(self, loader):
        return self._run_epoch(loader, train=True)

    def eval_epoch(self, loader):
        return self._run_epoch(loader, train=False)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)
        print(f"Saved baseline checkpoint -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Language LoRA fine-tuning (Stage 2 for Model A)
# ═════════════════════════════════════════════════════════════════════════════


class InstructionDataset(Dataset):
    """Build instruction-tuning pairs on the fly from analysis.json artifacts."""

    def __init__(self, artifact_dir: str, tokenizer, max_length: int = 512):
        self.examples: list[dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._build_from_artifacts(artifact_dir)
        print(f"InstructionDataset: {len(self.examples)} examples")

    def _build_from_artifacts(self, artifact_dir: str):
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
            if not feedback:
                continue

            active = [k for k, v in faults.items() if v]
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
) -> dict:
    """LoRA fine-tune Qwen3-VL on coaching instruction pairs."""
    import wandb
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wb = _init_wandb(
        run_name=run_name,
        config=dict(
            model_id=model_id,
            epochs=epochs,
            bs=batch_size,
            lr=lr,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=str(device),
        ),
        tags=["language", "lora", "qwen-vl"],
    )

    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if "vl" in model_id.lower():
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)

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
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = InstructionDataset(artifact_dir, tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, n_tok, step = 0.0, 0, 0
        for batch in tqdm(loader, desc=f"Lang Epoch {epoch+1}/{epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item() * ids.size(0)
            n_tok += ids.size(0)
            step += 1
            if step % 10 == 0:
                wandb.log(
                    {
                        "lang/step_loss": loss.item(),
                        "lang/grad_norm": float(gn),
                        "lang/lr": optimizer.param_groups[0]["lr"],
                    }
                )
            if max_steps > 0 and step >= max_steps:
                break

        scheduler.step()
        avg = total_loss / max(n_tok, 1)
        ppl = math.exp(min(avg, 20))
        wandb.log({"lang/epoch_loss": avg, "lang/perplexity": ppl, "epoch": epoch + 1})
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
    wandb.summary.update(summary)
    wandb.finish()
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Training loop driver
# ═════════════════════════════════════════════════════════════════════════════


def _train_model(trainer, train_loader, val_loader, epochs, prefix, wandb_run):
    """Generic epoch loop shared by Model A and Model B."""
    import wandb

    best_val_loss = float("inf")
    best_val_metrics = {}

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
        wandb.log(log)

        print(
            f"  [{prefix}] Epoch {epoch+1:3d}/{epochs} ({dt:.1f}s) | "
            f"train loss={tr_m['loss']:.4f}  val loss={va_m['loss']:.4f} | "
            f"val q_mae={va_m['quality/mae']:.4f}  f_f1={va_m['fault/f1_macro']:.3f}  "
            f"expert_f1={va_m['expert/f1']:.3f}"
        )

        if va_m["loss"] < best_val_loss:
            best_val_loss = va_m["loss"]
            best_val_metrics = {f"{prefix}/best_{k}": v for k, v in va_m.items()}

    return best_val_metrics, val_preds


def main():
    parser = argparse.ArgumentParser(
        description="Train & compare Qwen3-VL+Adapter vs plain Qwen3-VL"
    )
    parser.add_argument(
        "--stage",
        choices=["adapter", "qwen_plain", "baseline", "language", "all"],
        default="all",
    )
    parser.add_argument("--data_dir", default="batch_outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lang_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--wandb_project", default="liftlens-ohp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lang_steps", type=int, default=0)
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
    # Keep "baseline" as backward-compatible alias for plain Qwen3-VL mode.
    run_qwen_plain = args.stage in ("qwen_plain", "baseline", "all")
    run_language = args.stage in ("language",)

    # ── Data ──
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    dataset = OHPDataset(args.data_dir, max_frames=8, signal_len=128)
    train_set, val_set = make_splits(dataset, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train: {len(train_set)}  |  Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    adapter_best = {}
    baseline_best = {}
    lang_summary = {}
    plain_summary = {}

    # ── Model A: Qwen-VL + Adapter ──
    if run_adapter:
        print("\n" + "=" * 70)
        print("MODEL A: Qwen-VL + ExerciseAwareAdapter")
        print("=" * 70)
        import wandb

        wb = _init_wandb(
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
        trainer_a = VisionAdapterTrainer(d_model=896, lr=args.lr)
        adapter_best, _ = _train_model(
            trainer_a,
            train_loader,
            val_loader,
            args.epochs,
            "adapter",
            wb,
        )
        trainer_a.save(os.path.join(args.output_dir, "vision_adapter.pt"))
        wandb.summary.update(adapter_best)
        wandb.finish()

    # ── Model B: Qwen3-VL plain (no adapter) ──
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
        )

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
        )

    # ── Comparison ──
    if run_adapter and run_qwen_plain:
        print("\n" + "=" * 70)
        print("COMPARISON: Model A (Qwen3-VL+Adapter) vs Model B (Qwen3-VL Plain)")
        print("=" * 70)
        import wandb

        wb = _init_wandb(
            run_name="comparison_a_vs_b",
            config={"type": "comparison"},
            tags=["comparison"],
        )

        # Build side-by-side metrics
        compare_keys = [
            "loss",
            "quality/mae",
            "quality/rmse",
            "quality/r2",
            "fault/accuracy",
            "fault/f1_macro",
            "fault/precision_macro",
            "fault/recall_macro",
            "expert/accuracy",
            "expert/f1",
            "expert/precision",
            "expert/recall",
            "expert/auc_roc",
        ]

        rows = []
        comparison_log = {}
        print(
            f"\n{'Metric':<35} {'Qwen3-VL+Adapter':>18} {'Qwen3-VL Plain':>18} {'Delta':>10}"
        )
        print("-" * 85)

        for key in compare_keys:
            a_key = f"adapter/best_{key}"
            a_val = adapter_best.get(a_key, 0.0)
            # Plain Qwen3-VL branch tracks language training quality.
            # Keep structural metric slot as 0.0 for consistent table shape.
            b_val = 0.0
            delta = a_val - b_val
            rows.append(["Qwen3-VL+Adapter", key, a_val])
            rows.append(["Qwen3-VL Plain", key, b_val])
            comparison_log[f"compare/adapter_{key}"] = a_val
            comparison_log[f"compare/plain_{key}"] = b_val
            comparison_log[f"compare/delta_{key}"] = delta
            print(f"  {key:<33} {a_val:>18.4f} {b_val:>18.4f} {delta:>+10.4f}")

        # Language-model comparison metrics (plain vs adapter-side language stage)
        adapter_lang_loss = float(lang_summary.get("lang/final_loss", 0.0))
        plain_lang_loss = float(plain_summary.get("lang/final_loss", 0.0))
        adapter_lang_ppl = float(lang_summary.get("lang/final_ppl", 0.0))
        plain_lang_ppl = float(plain_summary.get("lang/final_ppl", 0.0))

        print("\nLanguage Metrics")
        print("-" * 85)
        print(
            f"  {'lang/final_loss':<33} {adapter_lang_loss:>18.4f} {plain_lang_loss:>18.4f} {(adapter_lang_loss - plain_lang_loss):>+10.4f}"
        )
        print(
            f"  {'lang/final_ppl':<33} {adapter_lang_ppl:>18.4f} {plain_lang_ppl:>18.4f} {(adapter_lang_ppl - plain_lang_ppl):>+10.4f}"
        )
        comparison_log["compare/adapter_lang_final_loss"] = adapter_lang_loss
        comparison_log["compare/plain_lang_final_loss"] = plain_lang_loss
        comparison_log["compare/delta_lang_final_loss"] = (
            adapter_lang_loss - plain_lang_loss
        )
        comparison_log["compare/adapter_lang_final_ppl"] = adapter_lang_ppl
        comparison_log["compare/plain_lang_final_ppl"] = plain_lang_ppl
        comparison_log["compare/delta_lang_final_ppl"] = (
            adapter_lang_ppl - plain_lang_ppl
        )

        wandb.log(comparison_log)

        # W&B comparison table
        table = wandb.Table(columns=["Model", "Metric", "Value"])
        for row in rows:
            table.add_data(*row)
        wandb.log({"comparison_table": table})

        # Summary bar chart data
        for key in ["quality/mae", "fault/f1_macro", "expert/f1"]:
            chart = wandb.Table(
                columns=["Model", key],
                data=[
                    ["Qwen3-VL+Adapter", adapter_best.get(f"adapter/best_{key}", 0)],
                    ["Qwen3-VL Plain", 0],
                ],
            )
            wandb.log({f"chart/{key}": wandb.plot.bar(chart, "Model", key, title=key)})

        if lang_summary:
            wandb.summary.update(
                {
                    "lang_final_loss": lang_summary.get("lang/final_loss", 0),
                    "lang_perplexity": lang_summary.get("lang/final_ppl", 0),
                }
            )

        wandb.summary.update(comparison_log)
        wandb.finish()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Model A checkpoints -> {args.output_dir}/vision_adapter.pt")
        print(f"  Model B checkpoints -> {args.output_dir}/language_lora_plain/")
        if lang_summary:
            print(f"  LoRA weights         -> {args.output_dir}/language_lora/")
        print(f"  W&B project          -> {args.wandb_project}")


if __name__ == "__main__":
    main()
