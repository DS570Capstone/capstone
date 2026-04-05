"""Rule-based fault evaluation and coaching cue selection."""
from __future__ import annotations

import yaml
import os
from typing import Optional


def load_rules(rules_path: str) -> dict:
    with open(rules_path) as f:
        return yaml.safe_load(f)


def select_rules(
    fault_flags: dict[str, bool],
    rules: dict,
    max_rules: int = 4,
) -> list[dict]:
    """Return ordered list of triggered rule dicts."""
    triggered = []
    severity_order = {"high": 0, "moderate": 1, "low": 2}
    for rule in rules.get("rules", []):
        fault = rule.get("fault", "")
        if fault_flags.get(fault, False):
            triggered.append(rule)
    triggered.sort(key=lambda r: severity_order.get(r.get("severity", "low"), 2))
    return triggered[:max_rules]


def format_coaching_feedback(
    triggered_rules: list[dict],
    cluster_name: str,
    wave_quality: dict,
    rules: dict,
    uncertainty: bool = False,
) -> dict:
    """
    Build the language output dict from triggered rules and quality metrics.
    """
    grade = wave_quality.get("grade", "?")
    overall = wave_quality.get("overall", 0.0)
    smoothness = wave_quality.get("smoothness", 0.0)

    # Summary
    if not triggered_rules:
        # No faults — use positive cues
        general = rules.get("general_cues", {})
        summary = general.get("symmetric_press", "Press mechanics look balanced from this view.")
        coach_feedback = " ".join([
            general.get("stable_lockout", ""),
            general.get("smooth_bar_path", ""),
        ]).strip()
    else:
        summary = (
            f"Movement grade: {grade} (overall score {overall:.2f}). "
            f"Detected {len(triggered_rules)} area(s) for improvement."
        )
        cues = [r["cue"] for r in triggered_rules]
        coach_feedback = " | ".join(cues)

    reasoning = (
        f"Archetype: {cluster_name}. "
        f"Quality — smoothness={smoothness:.2f}, overall={overall:.2f}. "
        f"Faults: {', '.join(r['fault'] for r in triggered_rules) or 'none'}."
    )

    if uncertainty:
        disclaimer = rules.get("uncertainty_disclaimer", "")
        if disclaimer:
            coach_feedback += f" Note: {disclaimer}"

    return {
        "summary": summary,
        "coach_feedback": coach_feedback,
        "reasoning_trace_short": reasoning,
    }
