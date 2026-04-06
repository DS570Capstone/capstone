"""Pose estimation — MediaPipe backend with multi-pass fallback for back-view videos."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Keypoints critical for back-view OHP analysis
_CRITICAL_KP = [
    KP["left_shoulder"], KP["right_shoulder"],
    KP["left_elbow"], KP["right_elbow"],
    KP["left_wrist"], KP["right_wrist"],
    KP["left_hip"], KP["right_hip"],
]


@dataclass
class PoseResult:
    keypoints: np.ndarray       # (N_KP, 2) in pixel coords, NaN if missing
    confidences: np.ndarray     # (N_KP,)
    visible: np.ndarray         # (N_KP,) bool
    world_landmarks: Optional[np.ndarray] = None  # (N_KP, 3) metric coords
    frame_idx: int = 0
    confidence_threshold: float = 0.5


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _count_critical_visible(visible: np.ndarray) -> int:
    return int(sum(visible[idx] for idx in _CRITICAL_KP))


def _find_model_path() -> str:
    """Locate the pose_landmarker model file."""
    import os
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "models", "pose_landmarker_heavy.task"),
        os.path.join(os.getcwd(), "models", "pose_landmarker_heavy.task"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "pose_landmarker_heavy.task not found. Download it:\n"
        "  curl -L -o models/pose_landmarker_heavy.task "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )


class MediaPipePoseEstimator:
    """MediaPipe PoseLandmarker (Tasks API >=0.10.14) with fallback for back-view."""

    def __init__(self, confidence_threshold: float = 0.5):
        import mediapipe as mp
        self.conf_thresh = confidence_threshold
        self._last_good_result: Optional[PoseResult] = None

        model_path = _find_model_path()

        # Primary landmarker (IMAGE mode — we process frame-by-frame)
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=confidence_threshold,
                min_pose_presence_confidence=max(confidence_threshold - 0.1, 0.2),
                min_tracking_confidence=max(confidence_threshold - 0.15, 0.2),
            )
        )

        # Fallback landmarker with lower thresholds
        lower = max(confidence_threshold - 0.25, 0.1)
        self._fallback = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=lower,
                min_pose_presence_confidence=lower,
                min_tracking_confidence=lower,
            )
        )

    def _result_to_pose(self, result, h: int, w: int, frame_idx: int,
                        threshold: float) -> PoseResult:
        n_kp = len(KEYPOINT_NAMES)
        keypoints = np.full((n_kp, 2), np.nan)
        confs = np.zeros(n_kp)
        visible = np.zeros(n_kp, dtype=bool)
        world = None

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]  # first detected person
            for i, lm in enumerate(lms):
                if i >= n_kp:
                    break
                vis = lm.visibility if hasattr(lm, "visibility") else 1.0
                confs[i] = vis
                if vis >= threshold:
                    keypoints[i] = [lm.x * w, lm.y * h]
                    visible[i] = True

        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world = np.full((n_kp, 3), np.nan)
            wlms = result.pose_world_landmarks[0]
            for i, lm in enumerate(wlms):
                if i >= n_kp:
                    break
                if confs[i] >= threshold:
                    world[i] = [lm.x, lm.y, lm.z]

        return PoseResult(
            keypoints=keypoints,
            confidences=confs,
            visible=visible,
            world_landmarks=world,
            frame_idx=frame_idx,
            confidence_threshold=threshold,
        )

    def _empty_pose(self, frame_idx: int) -> PoseResult:
        n_kp = len(KEYPOINT_NAMES)
        return PoseResult(
            keypoints=np.full((n_kp, 2), np.nan),
            confidences=np.zeros(n_kp),
            visible=np.zeros(n_kp, dtype=bool),
            frame_idx=frame_idx,
            confidence_threshold=self.conf_thresh,
        )

    def process_frame(self, bgr_frame: np.ndarray, frame_idx: int) -> PoseResult:
        import cv2
        import mediapipe as mp

        h, w = bgr_frame.shape[:2]
        # Reject frames with dimensions outside the PIL/MediaPipe hard limit.
        if h == 0 or w == 0 or h > 65535 or w > 65535:
            return self._empty_pose(frame_idx)

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        except Exception:
            return self._empty_pose(frame_idx)

        # Pass 1: primary landmarker
        result = self._landmarker.detect(mp_image)
        pose = self._result_to_pose(result, h, w, frame_idx, self.conf_thresh)
        n_critical = _count_critical_visible(pose.visible)

        # Pass 2: fallback with lower thresholds
        if n_critical < 4:
            result2 = self._fallback.detect(mp_image)
            lower_thresh = max(self.conf_thresh - 0.25, 0.1)
            pose2 = self._result_to_pose(result2, h, w, frame_idx, lower_thresh)
            n_critical2 = _count_critical_visible(pose2.visible)
            if n_critical2 > n_critical:
                pose = pose2

        # Pass 3: CLAHE enhanced image
        n_critical = _count_critical_visible(pose.visible)
        if n_critical < 4:
            lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            enhanced = cv2.merge([l_ch, a_ch, b_ch])
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            mp_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_rgb)
            result3 = self._fallback.detect(mp_enhanced)
            lower_thresh = max(self.conf_thresh - 0.3, 0.1)
            pose3 = self._result_to_pose(result3, h, w, frame_idx, lower_thresh)
            n_critical3 = _count_critical_visible(pose3.visible)
            if n_critical3 > n_critical:
                pose = pose3

        # Temporal carry-forward for missing critical keypoints
        n_critical = _count_critical_visible(pose.visible)
        if n_critical < 6 and self._last_good_result is not None:
            prev = self._last_good_result
            for idx in _CRITICAL_KP:
                if not pose.visible[idx] and prev.visible[idx]:
                    pose.keypoints[idx] = prev.keypoints[idx]
                    pose.confidences[idx] = prev.confidences[idx] * 0.7
                    pose.visible[idx] = True

        if _count_critical_visible(pose.visible) >= 5:
            self._last_good_result = pose

        return pose

    def close(self):
        self._landmarker.close()
        self._fallback.close()


class PoseEstimator:
    """Unified pose estimator factory."""

    def __init__(self, backend: str = "mediapipe", confidence_threshold: float = 0.5):
        self.backend = backend
        self.conf_thresh = confidence_threshold
        if backend == "mediapipe":
            self._impl = MediaPipePoseEstimator(confidence_threshold)
        else:
            raise NotImplementedError(f"Backend '{backend}' not yet implemented. Use 'mediapipe'.")

    def process_frame(self, bgr_frame: np.ndarray, frame_idx: int) -> PoseResult:
        return self._impl.process_frame(bgr_frame, frame_idx)

    def process_video(self, frames: list[np.ndarray]) -> list[PoseResult]:
        results = []
        for i, frame in enumerate(frames):
            results.append(self.process_frame(frame, i))
        return results

    def close(self):
        self._impl.close()


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

    # Elbow angles
    safe_angle(KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"], "left_elbow_angle_deg")
    safe_angle(KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"], "right_elbow_angle_deg")
    # Knee angles
    safe_angle(KP["left_hip"], KP["left_knee"], KP["left_ankle"], "left_knee_angle_deg")
    safe_angle(KP["right_hip"], KP["right_knee"], KP["right_ankle"], "right_knee_angle_deg")
    # Shoulder line tilt (angle of shoulder midline from horizontal)
    ls = kp[KP["left_shoulder"]]
    rs = kp[KP["right_shoulder"]]
    if not (np.isnan(ls).any() or np.isnan(rs).any()):
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        angles["shoulder_line_tilt_deg"] = float(np.degrees(np.arctan2(dy, dx)))
    else:
        angles["shoulder_line_tilt_deg"] = float("nan")
    # Hip line tilt
    lh = kp[KP["left_hip"]]
    rh = kp[KP["right_hip"]]
    if not (np.isnan(lh).any() or np.isnan(rh).any()):
        dx = rh[0] - lh[0]
        dy = rh[1] - lh[1]
        angles["hip_line_tilt_deg"] = float(np.degrees(np.arctan2(dy, dx)))
    else:
        angles["hip_line_tilt_deg"] = float("nan")

    return angles
