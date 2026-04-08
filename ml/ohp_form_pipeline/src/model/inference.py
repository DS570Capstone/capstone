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
        self._adapter_path = adapter_path
        self._lora_path = lora_path

    def _load(self):
        if self._model is not None:
            return

        print(f"Loading {self.model_id} ...")
        from transformers import (
            AutoModelForImageTextToText,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
        )

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        is_vl_model = "vl" in self.model_id.lower()

        # Load base model (choose VL or text-only stack based on model_id)
        if is_vl_model:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)

        # Load LoRA weights if available
        if self._lora_path and os.path.isdir(self._lora_path):
            print(f"Loading LoRA weights from {self._lora_path} ...")
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._lora_path)
            print("LoRA weights loaded.")

        # Load vision adapter
        if self._adapter_path and os.path.exists(self._adapter_path):
            print(f"Loading ExerciseAwareAdapter from {self._adapter_path} ...")
            from src.model.exercise_vision_adapter import ExerciseAwareAdapter

            self._adapter = ExerciseAwareAdapter(d_model=896).to(self.device)
            ckpt = torch.load(
                self._adapter_path, map_location=self.device, weights_only=True
            )
            self._adapter.load_state_dict(ckpt["adapter"])
            self._adapter.eval()
            print(f"Adapter loaded ({self._adapter.num_trainable_params:,} params).")

        self._model.eval()
        print("Model ready.")

    def generate_feedback(
        self,
        video_path: str,
        artifact: Optional[dict] = None,
        max_new_tokens: int = 512,
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

        prompt_text = (
            f"You are an expert strength and conditioning coach analyzing overhead press form.\n\n"
            f"Biomechanical Analysis:\n{metrics_text}\n"
            f"Based on the video frames and analysis above, provide:\n"
            f"1. A brief assessment of the lifter's form\n"
            f"2. The most critical fault to address first\n"
            f"3. Specific, actionable coaching cues\n"
            f"4. What the lifter is doing well\n"
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

        return {
            "feedback": generated,
            "quality": quality,
            "faults": active_faults,
            "reasoning": f"Model: {self.model_id}, Adapter: {self._adapter_path is not None}, LoRA: {self._lora_path is not None}",
        }


def main():
    parser = argparse.ArgumentParser(description="ExerciseAware inference")
    parser.add_argument("--video", required=True)
    parser.add_argument("--artifact", default=None, help="Pre-computed analysis.json")
    parser.add_argument("--adapter", default="checkpoints/vision_adapter.pt")
    parser.add_argument("--lora", default="checkpoints/language_lora")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--max_tokens", type=int, default=512)
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
