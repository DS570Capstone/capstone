"""Signal smoothing utilities."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def savgol_smooth(signal: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    arr = _fill_nans(signal.copy().astype(np.float64))
    if len(arr) < window:
        return arr
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(arr) - (0 if len(arr) % 2 == 1 else 1))
    if w < poly + 2:
        return arr
    return savgol_filter(arr, w, poly)


def lowpass_smooth(signal: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    arr = _fill_nans(signal.copy().astype(np.float64))
    nyq = fps / 2.0
    if cutoff_hz >= nyq or len(arr) < 15:
        return arr
    b, a = butter(2, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, arr)


def _fill_nans(arr: np.ndarray) -> np.ndarray:
    nans = np.isnan(arr)
    if nans.all():
        arr[:] = 0.0
        return arr
    if nans.any():
        xs = np.where(~nans)[0]
        arr[nans] = np.interp(np.where(nans)[0], xs, arr[xs])
    return arr


def compute_derivatives(
    signal: np.ndarray, fps: float
) -> dict[str, np.ndarray]:
    """Return velocity, acceleration, jerk from a smoothed signal."""
    dt = 1.0 / fps
    vel = np.gradient(signal, dt)
    acc = np.gradient(vel, dt)
    jerk = np.gradient(acc, dt)
    return {"velocity": vel, "acceleration": acc, "jerk": jerk}
