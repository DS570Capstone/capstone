"""VLM-based feedback generator — Qwen2.5-VL-0.5B with ExerciseAware adapter."""
from __future__ import annotations

import base64
import os
from typing import Optional
import numpy as np


SYSTEM_PROMPT = """You are an expert strength and conditioning coach analyzing overhead press (OHP) form from a back-view camera. You have deep knowledge of biomechanics, exercise physiology, and the harmonic nature of human movement during resistance training.

You will be given:
1. Structured biomechanical metrics extracted from video analysis
2. Key frames from the video
3. Detected movement archetypes and fault flags
4. Harmonic movement quality indicators

Your job is to produce concise, specific, actionable coaching feedback grounded in the provided metrics.

Rules:
- Only mention faults that are supported by the provided metrics and fault_flags
- Do NOT hallucinate faults not present in the data
- Be specific: reference measurements when possible
- Keep feedback to 3-5 sentences maximum
- Mention uncertainty when depth-estimated metrics are used
- Focus on back-view observable mechanics: symmetry, bar tilt, lateral drift, trunk shift
- Consider the harmonic quality of the movement — smooth, rhythmic reps indicate good motor control"""


def _encode_frame(bgr_frame: np.ndarray) -> str:
    """Encode BGR frame as base64 JPEG string."""
    import cv2
    _, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_prompt(artifact: dict, frames: Optional[list[np.ndarray]] = None) -> list[dict]:
    """Build Qwen2.5-VL style message list."""
    fault_flags = artifact.get("fault_flags", {})
    active_faults = [k for k, v in fault_flags.items() if v]
    quality = artifact.get("wave_features", {}).get("quality", {})
    cluster = artifact.get("unsupervised", {}).get("consensus_cluster_name", "unknown")
    depth_enabled = artifact.get("depth_features", {}).get("depth_enabled", False)
    harmonic = artifact.get("wave_features", {}).get("harmonic", {})
    energy = artifact.get("wave_features", {}).get("energy", {})

    metrics_text = (
        f"Movement archetype: {cluster}\n"
        f"Quality grade: {quality.get('grade', '?')} "
        f"(overall={quality.get('overall', 0):.2f}, "
        f"smoothness={quality.get('smoothness', 0):.2f}, "
        f"symmetry={quality.get('symmetry', 0):.2f}, "
        f"control={quality.get('control', 0):.2f})\n"
        f"Active fault flags: {', '.join(active_faults) if active_faults else 'none'}\n"
        f"Depth estimation used: {depth_enabled}\n"
        f"Harmonic movement: {'yes' if harmonic.get('is_harmonic', False) else 'no'} "
        f"(oscillation_count={harmonic.get('oscillation_count', 0)})\n"
        f"Energy efficiency: {energy.get('efficiency_pct', 0):.1f}%\n"
    )

    if depth_enabled:
        df = artifact.get("depth_features", {})
        metrics_text += (
            f"Bar depth asymmetry: {df.get('bar_depth_asymmetry', 0):.3f}\n"
            f"Torso depth shift: {df.get('torso_depth_shift', 0):.3f}\n"
        )

    content = []
    if frames:
        for frame in frames[:4]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_frame(frame)}"
                },
            })

    content.append({"type": "text", "text": metrics_text})
    content.append({
        "type": "text",
        "text": (
            "Based on the above data, provide specific coaching feedback for this OHP rep. "
            "Consider the harmonic quality of the movement — does the lifter maintain smooth, "
            "rhythmic control throughout the press?"
        ),
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


class VLMFeedbackGenerator:
    """
    Qwen2.5-VL-0.5B with ExerciseAware adapter + LoRA fine-tuning.
    Falls back to rule-based output if VLM not available.
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", False)
        self._model = None
        self._processor = None

    def _load(self):
        if not self.enabled or self._model is not None:
            return
        try:
            import torch
            model_id = self.config.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
            device = self.config.get("device", "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            print(f"[VLM] Loading {model_id} ...")

            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)

            # Load LoRA weights if available
            lora_path = self.config.get("lora_path")
            if lora_path and os.path.isdir(lora_path):
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, lora_path)
                print(f"[VLM] LoRA weights loaded from {lora_path}")

            self._model.eval()
            print(f"[VLM] Loaded {model_id} on {device}.")
        except Exception as e:
            print(f"[VLM] Failed to load: {e}. Using rule-based fallback.")
            self.enabled = False

    def generate(
        self,
        artifact: dict,
        key_frames: Optional[list[np.ndarray]] = None,
        rule_based_fallback: Optional[dict] = None,
    ) -> dict:
        """Generate language feedback. Falls back to rule_based_fallback if VLM unavailable."""
        if not self.enabled:
            return rule_based_fallback or {
                "summary": "VLM disabled.",
                "coach_feedback": "",
                "reasoning_trace_short": "",
            }

        self._load()
        if not self.enabled:
            return rule_based_fallback or {}

        import torch
        from PIL import Image
        import cv2

        messages = _build_prompt(artifact, key_frames)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Convert frames to PIL images for the processor
        pil_images = []
        if key_frames:
            for frame in key_frames[:4]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))

        inputs = self._processor(
            text=[text],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                temperature=self.config.get("temperature", 0.3),
                do_sample=True,
            )
        generated = self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()

        return {
            "summary": generated[:200] if len(generated) > 200 else generated,
            "coach_feedback": generated,
            "reasoning_trace_short": f"VLM: {self.config.get('model_id', 'unknown')}",
        }
