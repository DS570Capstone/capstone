"""Map cluster IDs to human-readable OHP archetype names using feature centroids."""
from __future__ import annotations

import numpy as np
import yaml
import os


def load_archetype_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def name_clusters_from_features(
    cluster_labels: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    archetype_config: dict,
    thresholds_config: dict,
) -> tuple[dict[int, str], dict[int, dict]]:
    """
    Assign archetype names to cluster IDs by evaluating each cluster's centroid
    against the threshold rules in priority order.

    Returns:
        cluster_names: {cluster_id: archetype_name}
        cluster_summaries: {cluster_id: {metrics...}}
    """
    unique_ids = sorted(set(cluster_labels))
    archetypes = archetype_config.get("archetypes", {})
    priority = archetype_config.get("priority_order", list(archetypes.keys()))
    fault_thresholds = thresholds_config.get("faults", {})

    cluster_names: dict[int, str] = {}
    cluster_summaries: dict[int, dict] = {}

    for cid in unique_ids:
        if cid == -1:
            cluster_names[cid] = "outlier_unknown_pattern"
            cluster_summaries[cid] = {}
            continue

        mask = cluster_labels == cid
        centroid = feature_matrix[mask].mean(axis=0)
        centroid_dict = {name: float(val) for name, val in zip(feature_names, centroid)}

        # Walk priority list and pick first matching archetype
        assigned = "outlier_unknown_pattern"
        for arch_name in priority:
            if arch_name == "smooth_symmetric_press":
                # Only assign if no fault thresholds are breached
                any_fault = False
                for fault_name, fault_cfg in fault_thresholds.items():
                    metric = fault_cfg.get("metric", "")
                    thresh = fault_cfg.get("threshold", 0.0)
                    direction = fault_cfg.get("direction", "gt")
                    val = centroid_dict.get(metric, 0.0)
                    if direction == "gt" and val > thresh:
                        any_fault = True
                        break
                    elif direction == "lt" and val < thresh:
                        any_fault = True
                        break
                if not any_fault:
                    assigned = arch_name
                    break
            elif arch_name == "outlier_unknown_pattern":
                assigned = arch_name
                break
            else:
                fault_name = arch_name
                fault_cfg = fault_thresholds.get(fault_name, {})
                metric = fault_cfg.get("metric", "")
                thresh = fault_cfg.get("threshold", 0.0)
                direction = fault_cfg.get("direction", "gt")
                val = centroid_dict.get(metric, 0.0)
                match = (direction == "gt" and val > thresh) or \
                        (direction == "lt" and val < thresh)
                if match:
                    assigned = arch_name
                    break

        cluster_names[cid] = assigned
        cluster_summaries[cid] = {
            "n_samples": int(mask.sum()),
            "centroid_metrics": {k: round(v, 4) for k, v in centroid_dict.items()},
            "archetype": assigned,
        }

    return cluster_names, cluster_summaries


def assign_clip_fault_flags(
    features: dict,
    thresholds_config: dict,
) -> dict[str, bool]:
    """Evaluate per-clip feature dict against all fault thresholds."""
    faults = thresholds_config.get("faults", {})
    flags: dict[str, bool] = {}
    for fault_name, fault_cfg in faults.items():
        metric = fault_cfg.get("metric", "")
        thresh = fault_cfg.get("threshold", 0.0)
        direction = fault_cfg.get("direction", "gt")
        val = float(features.get(metric, 0.0))
        if direction == "gt":
            flags[fault_name] = val > thresh
        elif direction == "lt":
            flags[fault_name] = val < thresh
        else:
            flags[fault_name] = False
    return flags
