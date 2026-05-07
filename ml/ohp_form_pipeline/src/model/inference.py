"""
Inference pipeline — ExerciseAware Qwen3-VL for coaching feedback.

Combines:
    1. Qwen3-VL vision encoder (frozen)
  2. ExerciseAwareAdapter (trained pose+harmonic fusion layer)
  3. LoRA-finetuned language model (S&C domain knowledge)

Usage:
    python -m src.model.inference \
        --video path/to/video.mp4 \
        --adapter checkpoints/vision_adapter.pt \
        --lora checkpoints/language_lora
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import torch

weave = None

try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv(*args, **kwargs):
        return False


PIPELINE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Load env for WANDB_API_KEY
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
sys.path.insert(0, PIPELINE_ROOT)


class ExerciseAwareCoach:
    """Full inference pipeline combining vision adapter + LoRA language model."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Thinking",
        adapter_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._adapter = None
        self._quality_head = None
        self._fault_head = None
        self._expert_head = None
        self._adapter_path = adapter_path
        self._lora_path = lora_path

    _ADAPTER_FAULT_NAMES = [
        "left_right_lockout_asymmetry",
        "uneven_press_timing",
        "compensatory_lateral_shift",
        "trunk_shift_under_load",
        "hip_shift_compensation",
        "unstable_lockout",
    ]

    @staticmethod
    def _resample_1d(arr: np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(n, dtype=np.float32)
        if len(arr) == n:
            return arr.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return np.interp(x_new, x_old, arr).astype(np.float32)

    def _build_adapter_inputs(self, artifact: dict, max_frames: int = 16, signal_len: int = 128):
        raw = artifact.get("raw_signals", {})
        lw = np.asarray(raw.get("left_wrist_y", []), dtype=np.float32)
        rw = np.asarray(raw.get("right_wrist_y", []), dtype=np.float32)
        bx = np.asarray(raw.get("bar_center_x", []), dtype=np.float32)
        by = np.asarray(raw.get("bar_center_y", []), dtype=np.float32)
        core_x = np.asarray(raw.get("trunk_center_x", []), dtype=np.float32)

        n = max(len(lw), len(rw), len(by), len(core_x), 1)
        lw_n = self._resample_1d(lw, n)
        rw_n = self._resample_1d(rw, n)
        bx_n = self._resample_1d(bx, n)
        by_n = self._resample_1d(by, n)
        cx_n = self._resample_1d(core_x, n)

        # Build 85-dim feature template expected by adapter trainer.
        pose = np.zeros((max_frames, 85), dtype=np.float32)
        idx = np.linspace(0, n - 1, min(max_frames, n), dtype=int)
        for i, j in enumerate(idx):
            left = float(lw_n[j])
            right = float(rw_n[j])
            barx = float(bx_n[j])
            bary = float(by_n[j])
            core = float(cx_n[j])
            arm_mean = 0.5 * (left + right)
            asym = left - right
            # mask-feature slots used in training fallback: [66..71]
            pose[i, 66] = arm_mean / 720.0
            pose[i, 67] = asym / 720.0
            pose[i, 68] = core / 720.0
            pose[i, 69] = (arm_mean - core) / 720.0
            pose[i, 70] = barx / 720.0
            pose[i, 71] = bary / 720.0
            # phase one-hot fallback (unknown)
            pose[i, 81:85] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        from src.model.exercise_vision_adapter import build_harmonic_signals_from_artifact

        harmonic = build_harmonic_signals_from_artifact(artifact, target_len=signal_len).astype(np.float32)
        return (
            torch.from_numpy(pose).unsqueeze(0).to(self.device),
            torch.from_numpy(harmonic).unsqueeze(0).to(self.device),
        )

    def _adapter_conditioning_text(self, artifact: dict) -> str:
        if self._adapter is None or self._quality_head is None:
            return ""
        with torch.no_grad():
            pose, harm = self._build_adapter_inputs(artifact)
            # Mirror training forward path: adapter receives dummy visual tokens.
            dummy_visual = torch.randn(1, 16, self._adapter.d_model, device=self.device)
            fused = self._adapter(dummy_visual, pose, harm)
            pooled = fused.mean(dim=1)
            q = float(self._quality_head(pooled).squeeze(-1).item())
            f = torch.sigmoid(self._fault_head(pooled)).squeeze(0).detach().cpu().numpy().tolist()
            e = float(torch.sigmoid(self._expert_head(pooled).squeeze(-1)).item())

        top_faults = sorted(
            [(name, float(prob)) for name, prob in zip(self._ADAPTER_FAULT_NAMES, f)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        active = [name for name, prob in top_faults if prob >= 0.5]
        top_txt = ", ".join(f"{name}:{prob:.2f}" for name, prob in top_faults)
        return (
            "Adapter Signals:\n"
            f"- Predicted quality score: {q:.2f}\n"
            f"- Expert probability: {e:.2f}\n"
            f"- Top fault probabilities: {top_txt}\n"
            f"- Active adapter faults (>=0.50): {', '.join(active) if active else 'none'}\n"
            "Use these adapter signals as primary conditioning evidence along with video frames.\n"
        )

    def _load(self):
        if self._model is not None:
            return

        resolved_model_id = self.model_id
        use_peft_lora = False
        if self._lora_path and os.path.isdir(self._lora_path):
            peft_cfg = os.path.join(self._lora_path, "adapter_config.json")
            full_ckpt = os.path.join(self._lora_path, "model.safetensors")
            if os.path.exists(peft_cfg):
                use_peft_lora = True
            elif os.path.exists(full_ckpt):
                # Fallback path: training may have saved a full finetuned model dir.
                resolved_model_id = self._lora_path
                print(f"Using full finetuned checkpoint from {self._lora_path} ...")

        print(f"Loading {resolved_model_id} ...")
        from transformers import (
            AutoModelForImageTextToText,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
        )

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        is_vl_model = "vl" in resolved_model_id.lower()
        is_local_full_ckpt = os.path.isdir(resolved_model_id) and (resolved_model_id != self.model_id)
        processor_source = self.model_id if is_local_full_ckpt else resolved_model_id

        # Load base model (choose VL or text-only stack based on model_id)
        if is_vl_model:
            self._processor = AutoProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                resolved_model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                processor_source,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                resolved_model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)

        # Load LoRA weights if available
        if use_peft_lora:
            print(f"Loading LoRA weights from {self._lora_path} ...")
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._lora_path)
            print("LoRA weights loaded.")

        # Load vision adapter
        if self._adapter_path and os.path.exists(self._adapter_path):
            print(f"Loading ExerciseAwareAdapter from {self._adapter_path} ...")
            from src.model.exercise_vision_adapter import ExerciseAwareAdapter
            import torch.nn as nn

            self._adapter = ExerciseAwareAdapter(d_model=896).to(self.device)
            ckpt = torch.load(
                self._adapter_path, map_location=self.device, weights_only=True
            )
            self._adapter.load_state_dict(ckpt["adapter"])
            self._quality_head = nn.Sequential(
                nn.Linear(896, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ).to(self.device)
            self._fault_head = nn.Sequential(
                nn.Linear(896, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, len(self._ADAPTER_FAULT_NAMES)),
            ).to(self.device)
            self._expert_head = nn.Sequential(
                nn.Linear(896, 32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            ).to(self.device)
            if "quality_head" in ckpt:
                self._quality_head.load_state_dict(ckpt["quality_head"])
            if "fault_head" in ckpt:
                self._fault_head.load_state_dict(ckpt["fault_head"])
            if "expert_head" in ckpt:
                self._expert_head.load_state_dict(ckpt["expert_head"])
            self._adapter.eval()
            self._quality_head.eval()
            self._fault_head.eval()
            self._expert_head.eval()
            print(f"Adapter loaded ({self._adapter.num_trainable_params:,} params).")

        self._model.eval()
        print("Model ready.")

    def generate_feedback(
        self,
        video_path: str,
        artifact: Optional[dict] = None,
        max_new_tokens: int = 768,
        temperature: float = 0.3,
    ) -> dict:
        """Generate coaching feedback for a video.

        Args:
            video_path: Path to the video file
            artifact: Pre-computed analysis artifact (from batch_process). If None,
                      will run the pipeline first.
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            dict with 'feedback', 'quality', 'faults', 'reasoning'
        """
        self._load()
        import cv2

        # Run pipeline if no artifact provided
        if artifact is None:
            from src.app.run_single_video import run, load_config

            config_path = os.path.join(PIPELINE_ROOT, "configs", "default.yaml")
            artifact = run(video_path, config_path)

        # Extract key frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), 4, dtype=int)
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                # Resize for VLM
                h, w = frame.shape[:2]
                if h > 480:
                    scale = 480 / h
                    frame = cv2.resize(frame, (int(w * scale), 480))
                frames.append(frame)
        cap.release()

        # Build context from artifact
        quality = artifact.get("wave_features", {}).get("quality", {})
        fault_flags = artifact.get("fault_flags", {})
        active_faults = [k for k, v in fault_flags.items() if v]

        metrics_text = (
            f"Exercise: Overhead Press (back view)\n"
            f"Quality Grade: {quality.get('grade', '?')}\n"
            f"Overall Score: {quality.get('overall', 0):.2f}\n"
            f"Smoothness: {quality.get('smoothness', 0):.2f}\n"
            f"Symmetry: {quality.get('symmetry', 0):.2f}\n"
            f"Control: {quality.get('control', 0):.2f}\n"
            f"Active Faults: {', '.join(active_faults) if active_faults else 'None detected'}\n"
        )

        # Build prompt input. Prefer multimodal format, but fall back to
        # text-only for templates that expect string content.
        content = []
        from PIL import Image

        pil_images = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
            content.append({"type": "image"})

        adapter_ctx = self._adapter_conditioning_text(artifact)

        prompt_text = (
            "[System]\n"
            "You are a world-class Strength and Conditioning Coach.\n"
            "You are provided with a video of an Overhead Press and real-time biomechanical adapter signals.\n"
            "The Overhead Press is an upper-body push exercise (shoulders/triceps), not a back exercise.\n\n"
            "[Constraints]\n"
            "- INTERNAL REASONING: Use the <think> block to analyze Quality Scores, Adapter Signals, and Symmetry.\n"
            "- USER FEEDBACK: Use the <coach_feedback> block for the final response.\n"
            "- STYLE: Professional, encouraging, and technical. Use S&C manual cues.\n"
            "- RULE: If any adapter fault signal is above 0.50, prioritize that fault in correction strategy.\n"
            "- FORBIDDEN IN <coach_feedback>: metadata, adapter, probabilities, logits, and raw decimals from score fields.\n\n"
            "[Input Data]\n"
            f"Video Artifacts:\n{metrics_text}\n"
            f"Vision Adapter Signals:\n{adapter_ctx}\n"
            "[Response Format]\n"
            "<think>\n"
            "(Perform biomechanical audit here)\n"
            "</think>\n\n"
            "<coach_feedback>\n"
            "### Form Assessment\n"
            "(Briefly describe overall quality using grade language only)\n\n"
            "### Critical Correction\n"
            "(Address the highest-priority fault)\n\n"
            "### Coaching Cues\n"
            "- (Actionable cue 1)\n"
            "- (Actionable cue 2)\n"
            "</coach_feedback>\n"
        )

        content.append(
            {
                "type": "text",
                "text": prompt_text,
            }
        )

        if self._processor is not None:
            use_images = bool(pil_images)
            messages = [{"role": "user", "content": content}]
            try:
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                # Some chat templates require plain string content.
                messages = [{"role": "user", "content": prompt_text}]
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                use_images = False

            inputs = self._processor(
                text=[text],
                images=pil_images if (use_images and pil_images) else None,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            prompt_len = inputs["input_ids"].shape[1]
        else:
            messages = [{"role": "user", "content": prompt_text}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
            ).to(self.device)
            prompt_len = inputs["input_ids"].shape[1]

        # Inject adapter if available
        # (In a production setup, this would hook into the model's forward pass.
        #  For now, the adapter pretraining aligns the representations, and the
        #  LoRA handles domain-specific language generation.)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        if self._processor is not None:
            generated = self._processor.batch_decode(
                output_ids[:, prompt_len:],
                skip_special_tokens=True,
            )[0].strip()
        else:
            generated = self._tokenizer.batch_decode(
                output_ids[:, prompt_len:],
                skip_special_tokens=True,
            )[0].strip()

        # Enforce structured output for downstream consumption.
        generated = self._enforce_structured_output(
            generated=generated,
            quality=quality,
            active_faults=active_faults,
            adapter_ctx=adapter_ctx,
        )

        return {
            "feedback": generated,
            "quality": quality,
            "faults": active_faults,
            "reasoning": f"Model: {self.model_id}, Adapter: {self._adapter_path is not None}, LoRA: {self._lora_path is not None}",
        }

    def _enforce_structured_output(
        self,
        generated: str,
        quality: dict,
        active_faults: list[str],
        adapter_ctx: str,
    ) -> str:
        def _clean_reasoning(s: str) -> str:
            s = re.sub(r"\b(wait\s*,?\s*no|wait\s+no|hmm+|maybe|i\s+guess)\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s+", " ", s).strip(" ,.;\n\t")
            return s

        def _sanitize_forbidden_feedback_terms(s: str) -> str:
            # Remove forbidden technical leakage from athlete-facing feedback.
            s = re.sub(r"\bmetadata\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\badapter\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\bprobabilities?\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\blogits?\b", "", s, flags=re.IGNORECASE)
            # Remove raw decimal score literals like 0.66 in feedback.
            s = re.sub(r"\b0\.\d+\b", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _ensure_feedback_sections(s: str, grade: str, priority_fault: str) -> str:
            if "### Form Assessment" in s and "### Critical Correction" in s and "### Coaching Cues" in s:
                return s
            # Preserve generated content; just wrap it into the requested sections.
            text = s.strip()
            if not text:
                return "### Form Assessment\n\n### Critical Correction\n\n### Coaching Cues\n"

            # Split into rough chunks while keeping model-produced wording.
            parts = re.split(r"\n\n+", text)
            assess = parts[0].strip() if len(parts) > 0 else ""
            correction = parts[1].strip() if len(parts) > 1 else ""
            cues_src = "\n".join(parts[2:]).strip() if len(parts) > 2 else ""

            if not assess:
                assess = ""
            if not correction:
                correction = ""

            cue_lines = [ln.strip("- ") for ln in re.split(r"\n|;", cues_src) if ln.strip()]
            cue_lines = cue_lines[:3]
            cues_block = "\n".join(f"- {c}" for c in cue_lines)
            return (
                "### Form Assessment\n"
                f"{assess}\n\n"
                "### Critical Correction\n"
                f"{correction}\n\n"
                "### Coaching Cues\n"
                f"{cues_block}"
            )

        def _clean_coach(s: str, grade: str, overall: float, priority_fault: str) -> str:
            s = re.sub(r"</?(think|coach_feedback)>", "", s, flags=re.IGNORECASE).strip()
            wait_hits = len(re.findall(r"\bwait\b", s, flags=re.IGNORECASE))
            if wait_hits >= 2:
                s = re.sub(r"\bwait\b[^.?!]*[.?!]?", "", s, flags=re.IGNORECASE)
            if len(s) > 2800:
                s = s[:2800].rsplit(" ", 1)[0]
            s = _sanitize_forbidden_feedback_terms(s)
            s = _ensure_feedback_sections(s, grade, priority_fault)
            return s

        text = (generated or "").strip()

        grade = quality.get("grade", "?")
        overall = float(quality.get("overall", 0.0))
        adapter_active = "none"
        top_probs = ""

        m_active = re.search(r"Active adapter faults \(>=0\.50\):\s*(.+)", adapter_ctx)
        if m_active:
            adapter_active = m_active.group(1).strip()
        m_top = re.search(r"Top fault probabilities:\s*(.+)", adapter_ctx)
        if m_top:
            top_probs = m_top.group(1).strip()

        priority_fault = adapter_active if adapter_active and adapter_active != "none" else (
            active_faults[0] if active_faults else "unstable_lockout"
        )

        if "<think>" in text and "</think>" in text and "<coach_feedback>" in text and "</coach_feedback>" in text:
            m_think = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
            m_coach = re.search(r"<coach_feedback>(.*?)</coach_feedback>", text, flags=re.DOTALL | re.IGNORECASE)
            think_txt = m_think.group(1).strip() if m_think else ""
            coach_txt = m_coach.group(1).strip() if m_coach else text
            # Remove any nested tags to keep a single top-level pair.
            think_txt = re.sub(r"</?(think|coach_feedback)>", "", think_txt, flags=re.IGNORECASE).strip()
            think_txt = _clean_reasoning(think_txt)
            coach_txt = _clean_coach(coach_txt, grade, overall, priority_fault)
            return f"<think>{think_txt}</think>\n<coach_feedback>{coach_txt}</coach_feedback>"

        think = (
            f"Quality grade {grade}, overall {overall:.2f}. "
            f"Metadata active faults: {', '.join(active_faults) if active_faults else 'none'}. "
            f"Adapter active faults: {adapter_active}. "
            f"Adapter top probabilities: {top_probs if top_probs else 'n/a'}. "
            "Prioritize any adapter fault above 0.50."
        )
        think = _clean_reasoning(think)
        coach = _clean_coach(text, grade, overall, priority_fault)
        return f"<think>{think}</think>\n<coach_feedback>{coach}</coach_feedback>"


def main():
    parser = argparse.ArgumentParser(description="ExerciseAware inference")
    parser.add_argument("--video", required=True)
    parser.add_argument("--artifact", default=None, help="Pre-computed analysis.json")
    parser.add_argument("--adapter", default="checkpoints/vision_adapter.pt")
    parser.add_argument("--lora", default="checkpoints/language_lora")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--max_tokens", type=int, default=768)
    args = parser.parse_args()

    artifact = None
    if args.artifact:
        with open(args.artifact) as f:
            artifact = json.load(f)

    adapter_path = args.adapter if os.path.exists(args.adapter) else None
    lora_path = args.lora if os.path.isdir(args.lora) else None

    coach = ExerciseAwareCoach(
        model_id=args.model_id,
        adapter_path=adapter_path,
        lora_path=lora_path,
    )
    result = coach.generate_feedback(
        args.video,
        artifact=artifact,
        max_new_tokens=args.max_tokens,
    )

    print("\n" + "=" * 60)
    print("COACHING FEEDBACK")
    print("=" * 60)
    print(
        f"\nQuality: {result['quality'].get('grade', '?')} ({result['quality'].get('overall', 0):.2f})"
    )
    print(f"Faults: {result['faults']}")
    print(f"\n{result['feedback']}")


if __name__ == "__main__":
    main()
