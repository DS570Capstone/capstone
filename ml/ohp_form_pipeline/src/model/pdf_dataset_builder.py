"""
PDF Dataset Builder — extract text from the strength & conditioning manual,
create instruction-tuning pairs for exercise coaching fine-tuning.

Generates a JSONL dataset of (instruction, response) pairs from:
1. Extracted PDF text (biomechanics, exercise technique, coaching cues)
2. Processed video artifacts (metrics + feedback pairs)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from PDF, organized by page with section detection.

    Returns list of {page: int, text: str, sections: list[str]}
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if not text.strip():
            continue

        # Detect section headers (lines in ALL CAPS or with specific patterns)
        sections = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if (len(line) > 5 and line.isupper() and len(line) < 100):
                sections.append(line)
            elif re.match(r"^(Chapter|Section|Part)\s+\d+", line, re.IGNORECASE):
                sections.append(line)

        pages.append({
            "page": page_num + 1,
            "text": text,
            "sections": sections,
        })
    doc.close()
    return pages


def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            # Keep overlap from end of current
            words = current.split()
            overlap_words = words[-min(len(words), overlap // 5):]
            current = " ".join(overlap_words) + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


INSTRUCTION_TEMPLATES = [
    {
        "pattern": r"(exercise|lift|press|squat|deadlift|bench|overhead|clean|snatch|jerk)",
        "templates": [
            "Explain the proper technique for {exercise} based on strength and conditioning principles.",
            "What are the key coaching cues for {exercise}?",
            "Describe common form faults during {exercise} and how to correct them.",
            "How should a coach assess {exercise} form from a back view?",
        ],
    },
    {
        "pattern": r"(muscle|joint|biomechanic|anatomy|movement|kinetic|kinematic)",
        "templates": [
            "Explain the biomechanics described in this passage and how they apply to exercise coaching.",
            "What muscles and joints are involved according to this description?",
            "How does this biomechanical principle affect exercise form assessment?",
        ],
    },
    {
        "pattern": r"(symmetry|asymmetr|bilateral|unilateral|imbalance)",
        "templates": [
            "How should a coach identify and address bilateral asymmetries during lifting?",
            "What are the signs of bilateral imbalance described here and their coaching implications?",
        ],
    },
    {
        "pattern": r"(safety|injury|risk|prevent|protect|stabiliz)",
        "templates": [
            "What safety considerations are outlined for this exercise pattern?",
            "How can injury risk be minimized based on these guidelines?",
        ],
    },
    {
        "pattern": r"(program|periodiz|volume|intensity|frequency|set|rep)",
        "templates": [
            "Summarize the programming principles described in this passage.",
            "How do these programming guidelines relate to form quality and fatigue management?",
        ],
    },
]


def _generate_qa_from_chunk(chunk: str, chunk_idx: int) -> list[dict]:
    """Generate instruction-response pairs from a text chunk."""
    pairs = []
    chunk_lower = chunk.lower()

    for template_group in INSTRUCTION_TEMPLATES:
        if re.search(template_group["pattern"], chunk_lower):
            # Find specific exercise names if present
            exercises = re.findall(
                r"\b(overhead press|OHP|squat|deadlift|bench press|clean|snatch|"
                r"jerk|barbell row|pull.?up|push.?up|lunge|plank)\b",
                chunk, re.IGNORECASE,
            )
            exercise_name = exercises[0] if exercises else "the described exercise"

            for tmpl in template_group["templates"]:
                instruction = tmpl.format(exercise=exercise_name)
                pairs.append({
                    "instruction": instruction,
                    "input": f"Reference material:\n{chunk[:800]}",
                    "output": chunk,
                    "source": "pdf_manual",
                    "chunk_idx": chunk_idx,
                })
                if len(pairs) >= 3:
                    break
            break

    # Always generate a generic summarization pair
    if len(chunk) > 200:
        pairs.append({
            "instruction": "Summarize the following strength and conditioning content and explain how it applies to exercise form analysis.",
            "input": chunk[:1000],
            "output": chunk,
            "source": "pdf_manual",
            "chunk_idx": chunk_idx,
        })

    return pairs


def _generate_qa_from_artifact(artifact: dict) -> list[dict]:
    """Generate instruction-response pairs from a processed video artifact."""
    pairs = []
    vid_id = artifact.get("video_id", "unknown")
    quality = artifact.get("wave_features", {}).get("quality", {})
    fault_flags = artifact.get("fault_flags", {})
    active_faults = [k for k, v in fault_flags.items() if v]
    language = artifact.get("language", {})
    coach_feedback = language.get("coach_feedback", "")

    if not coach_feedback:
        return pairs

    # Metrics summary for instruction context
    metrics = (
        f"Quality grade: {quality.get('grade', '?')}, "
        f"Overall: {quality.get('overall', 0):.2f}, "
        f"Smoothness: {quality.get('smoothness', 0):.2f}, "
        f"Symmetry: {quality.get('symmetry', 0):.2f}. "
        f"Active faults: {', '.join(active_faults) if active_faults else 'none'}."
    )

    pairs.append({
        "instruction": "Based on the biomechanical analysis of this overhead press video, provide specific coaching feedback.",
        "input": metrics,
        "output": coach_feedback,
        "source": f"video_{vid_id}",
    })

    if active_faults:
        pairs.append({
            "instruction": f"The following form faults were detected in an overhead press: {', '.join(active_faults)}. Explain what these mean and how to correct them.",
            "input": metrics,
            "output": coach_feedback,
            "source": f"video_{vid_id}",
        })

    return pairs


def build_dataset(
    pdf_path: str,
    artifact_dir: Optional[str] = None,
    output_path: str = "training_data.jsonl",
) -> str:
    """Build complete instruction-tuning dataset.

    Args:
        pdf_path: Path to strength & conditioning manual PDF
        artifact_dir: Optional path to batch_outputs/ with processed video JSON files
        output_path: Where to save the JSONL dataset

    Returns:
        Path to the saved dataset
    """
    all_pairs = []

    # Part 1: PDF-based pairs
    print(f"Extracting text from {pdf_path} ...")
    pages = extract_pdf_text(pdf_path)
    full_text = "\n\n".join(p["text"] for p in pages)
    chunks = _chunk_text(full_text, max_chars=1500)
    print(f"  {len(pages)} pages → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        pairs = _generate_qa_from_chunk(chunk, i)
        all_pairs.extend(pairs)

    print(f"  Generated {len(all_pairs)} Q&A pairs from PDF")

    # Part 2: Video artifact-based pairs
    if artifact_dir and os.path.isdir(artifact_dir):
        json_files = list(Path(artifact_dir).rglob("analysis.json"))
        print(f"Loading {len(json_files)} video artifacts ...")
        for jf in json_files:
            try:
                with open(jf) as f:
                    artifact = json.load(f)
                pairs = _generate_qa_from_artifact(artifact)
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"  [WARN] Failed to load {jf}: {e}")
        print(f"  Total pairs after video artifacts: {len(all_pairs)}")

    # Write JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {output_path} ({len(all_pairs)} examples)")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to S&C manual PDF")
    parser.add_argument("--artifacts", default=None, help="Path to batch_outputs dir")
    parser.add_argument("--output", default="data/training_data.jsonl")
    args = parser.parse_args()
    build_dataset(args.pdf, args.artifacts, args.output)
