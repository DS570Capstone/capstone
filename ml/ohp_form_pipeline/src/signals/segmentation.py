"""Rep phase segmentation from bar vertical velocity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from .smoothing import savgol_smooth, compute_derivatives


@dataclass
class PhaseSegment:
    phase_type: str          # "setup" | "concentric" | "lockout" | "eccentric"
    start_frame: int
    end_frame: int
    duration_sec: float


def segment_phases(
    bar_y: np.ndarray,
    fps: float,
    smooth_window: int = 9,
) -> list[PhaseSegment]:
    """
    Segment OHP rep into phases using bar vertical velocity.
    bar_y: y-pixel of bar center (smaller y = higher in image).
    In image coords, concentric = bar going UP = y decreasing = negative velocity.
    """
    n = len(bar_y)
    if n < 10:
        return [PhaseSegment("unknown", 0, n - 1, n / fps)]

    smoothed = savgol_smooth(bar_y, smooth_window)
    derivs = compute_derivatives(smoothed, fps)
    vel = derivs["velocity"]

    # Normalize velocity
    vel_std = np.std(vel) + 1e-8
    vel_norm = vel / vel_std

    # Concentric: velocity significantly negative (bar rising)
    # Eccentric: velocity significantly positive (bar descending)
    THRESH = 0.5

    phases: list[PhaseSegment] = []
    state = "setup"
    state_start = 0
    setup_done = False

    for i in range(1, n):
        new_state = state
        if not setup_done:
            # Detect start of concentric from quiescence
            if vel_norm[i] < -THRESH:
                new_state = "concentric"
                setup_done = True
        elif state == "concentric":
            if vel_norm[i] > -THRESH * 0.3:
                new_state = "lockout"
        elif state == "lockout":
            if vel_norm[i] > THRESH:
                new_state = "eccentric"
        elif state == "eccentric":
            if vel_norm[i] > -THRESH * 0.3 and vel_norm[i] < THRESH * 0.3:
                new_state = "rest"

        if new_state != state:
            phases.append(PhaseSegment(
                phase_type=state,
                start_frame=state_start,
                end_frame=i - 1,
                duration_sec=(i - state_start) / fps,
            ))
            state = new_state
            state_start = i

    phases.append(PhaseSegment(
        phase_type=state,
        start_frame=state_start,
        end_frame=n - 1,
        duration_sec=(n - state_start) / fps,
    ))

    return phases


def phases_to_dicts(phases: list[PhaseSegment]) -> list[dict]:
    return [
        {
            "type": p.phase_type,
            "start_frame": p.start_frame,
            "end_frame": p.end_frame,
            "duration_sec": round(p.duration_sec, 3),
        }
        for p in phases
    ]
