# Checkpoints

Trained model weights are gitignored due to size. Three files are needed:

## vision_adapter.pt (83 MB)
Trained ExerciseAwareAdapter — pose + harmonic cross-attention with quality/fault/expert heads.
Required only if `vlm.enabled: true` in `configs/default.yaml`.

## baseline_mlp.pt (4.2 MB)
Baseline MLP trained for comparison. Used by `src/model/train.py` only.

## language_lora/ (34 MB)
LoRA fine-tuned weights for Qwen2.5-0.5B-Instruct.
Required only if `vlm.enabled: true` in `configs/default.yaml`.

Contains:
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

## Notes
- With `vlm.enabled: false` (the default), **none of these are needed** to run the pipeline.
- To obtain these weights, contact the project maintainer or re-run training via `run_full_pipeline.py --step train`.
