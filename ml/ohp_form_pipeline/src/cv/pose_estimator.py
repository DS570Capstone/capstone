"""Pose module with estimation removed (returns empty keypoints)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


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
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


@dataclass
class PoseResult:
    keypoints: np.ndarray
    confidences: np.ndarray
    visible: np.ndarray
    world_landmarks: Optional[np.ndarray] = None
    frame_idx: int = 0
    confidence_threshold: float = 0.5


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class PoseEstimator:
    """No-op pose estimator: emits empty keypoints for every frame."""

    def __init__(self, backend: str = "disabled", confidence_threshold: float = 0.5):
        self.backend = backend
        self.conf_thresh = confidence_threshold

    def _empty_pose(self, frame_idx: int) -> PoseResult:
        n_kp = len(KEYPOINT_NAMES)
        keypoints = np.full((n_kp, 2), np.nan, dtype=float)
        confs = np.zeros(n_kp, dtype=float)
        visible = np.zeros(n_kp, dtype=bool)

        return PoseResult(
            keypoints=keypoints,
            confidences=confs,
            visible=visible,
            world_landmarks=None,
            frame_idx=frame_idx,
            confidence_threshold=self.conf_thresh,
        )

    def process_frame(self, bgr_frame: np.ndarray, frame_idx: int) -> PoseResult:
        return self._empty_pose(frame_idx)

    def process_video(self, frames: list[np.ndarray]) -> list[PoseResult]:
        return [self.process_frame(frame, i) for i, frame in enumerate(frames)]

    def close(self):
        return


def extract_angles_from_pose(pose: PoseResult) -> dict[str, float]:
    """Extract key joint angles relevant to back-view OHP."""
    kp = pose.keypoints
    angles: dict[str, float] = {}

    def safe_angle(a_idx, b_idx, c_idx, name: str):
        pts = [kp[a_idx], kp[b_idx], kp[c_idx]]
        if any(np.isnan(p).any() for p in pts):
            angles[name] = float("nan")
        else:
            angles[name] = _angle_3pts(pts[0], pts[1], pts[2])

    safe_angle(
        KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"], "left_elbow_angle_deg"
    )
    safe_angle(
        KP["right_shoulder"],
        KP["right_elbow"],
        KP["right_wrist"],
        "right_elbow_angle_deg",
    )
    safe_angle(KP["left_hip"], KP["left_knee"], KP["left_ankle"], "left_knee_angle_deg")
    safe_angle(
        KP["right_hip"], KP["right_knee"], KP["right_ankle"], "right_knee_angle_deg"
    )

    ls = kp[KP["left_shoulder"]]
    rs = kp[KP["right_shoulder"]]
    if not (np.isnan(ls).any() or np.isnan(rs).any()):
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        angles["shoulder_line_tilt_deg"] = float(np.degrees(np.arctan2(dy, dx)))
    else:
        angles["shoulder_line_tilt_deg"] = float("nan")

    lh = kp[KP["left_hip"]]
    rh = kp[KP["right_hip"]]
    if not (np.isnan(lh).any() or np.isnan(rh).any()):
        dx = rh[0] - lh[0]
        dy = rh[1] - lh[1]
        angles["hip_line_tilt_deg"] = float(np.degrees(np.arctan2(dy, dx)))
    else:
        angles["hip_line_tilt_deg"] = float("nan")

    return angles
