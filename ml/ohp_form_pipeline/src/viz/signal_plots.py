"""Signal and trajectory visualization plots."""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..signals.smoothing import compute_derivatives


def plot_trajectories(
    trajectories: dict[str, list],
    phases: list[dict],
    out_path: str,
    fps: float = 30.0,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    keys = [
        "bar_path_trajectory",
        "arm_trajectory",
        "legs_trajectory",
        "core_trajectory",
    ]
    labels = [
        "Bar Path (vertical, norm.)",
        "Arm (elevation symmetry)",
        "Legs (knee stability)",
        "Core (trunk shift)",
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for ax, key, label, color in zip(axes, keys, labels, colors):
        sig = np.array(trajectories.get(key, []))
        if len(sig) == 0:
            continue
        t = np.arange(len(sig)) / max(len(sig) - 1, 1)
        ax.plot(t, sig, color=color, linewidth=1.5, label=label)
        # Phase shading
        for phase in phases:
            s = phase["start_frame"] / max(len(sig) - 1, 1)
            e = phase["end_frame"] / max(len(sig) - 1, 1)
            phase_colors = {
                "setup": "#e3f2fd",
                "concentric": "#e8f5e9",
                "lockout": "#fff9c4",
                "eccentric": "#fce4ec",
                "rest": "#f3e5f5",
                "unknown": "#f5f5f5",
            }
            ax.axvspan(
                s,
                e,
                alpha=0.25,
                color=phase_colors.get(phase["type"], "#f5f5f5"),
                label=phase["type"],
            )
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Normalized Time")
    fig.suptitle("OHP Trajectory Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_signal_dashboard(
    bar_cy: np.ndarray,
    fps: float,
    phases: list[dict],
    out_path: str,
) -> None:
    from scipy.fft import rfft, rfftfreq

    valid = bar_cy[~np.isnan(bar_cy)]
    if len(valid) < 8:
        return

    derivs = compute_derivatives(valid, fps)
    t = np.arange(len(valid)) / fps
    freqs = rfftfreq(len(valid), d=1.0 / fps)
    power = np.abs(rfft(valid - valid.mean())) ** 2

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax_disp = fig.add_subplot(gs[0, :])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_acc = fig.add_subplot(gs[1, 1])
    ax_jerk = fig.add_subplot(gs[2, 0])
    ax_fft = fig.add_subplot(gs[2, 1])

    phase_colors = {
        "setup": "#e3f2fd",
        "concentric": "#e8f5e9",
        "lockout": "#fff9c4",
        "eccentric": "#fce4ec",
        "rest": "#f3e5f5",
    }

    def shade_phases(ax, n):
        for ph in phases:
            s = ph["start_frame"] / max(n - 1, 1) * t[-1]
            e = ph["end_frame"] / max(n - 1, 1) * t[-1]
            ax.axvspan(s, e, alpha=0.2, color=phase_colors.get(ph["type"], "#eee"))
            mid = (s + e) / 2.0
            ax.text(
                mid, ax.get_ylim()[1] * 0.9, ph["type"][:3], ha="center", fontsize=7
            )

    ax_disp.plot(t, valid, color="#2196F3")
    ax_disp.set_title("Bar Y (displacement)")
    ax_disp.set_ylabel("pixels")
    shade_phases(ax_disp, len(valid))
    ax_vel.plot(t, derivs["velocity"], color="#4CAF50")
    ax_vel.set_title("Velocity")
    ax_vel.axhline(0, color="gray", lw=0.5)
    ax_acc.plot(t, derivs["acceleration"], color="#FF9800")
    ax_acc.set_title("Acceleration")
    ax_jerk.plot(t, derivs["jerk"], color="#9C27B0")
    ax_jerk.set_title("Jerk")
    ax_fft.plot(freqs[1:], power[1:], color="#F44336")
    ax_fft.set_title("FFT Power Spectrum")
    ax_fft.set_xlabel("Frequency (Hz)")

    for ax in [ax_vel, ax_acc, ax_jerk]:
        ax.set_xlabel("Time (s)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Bar Path Signal Dashboard", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_bilateral_symmetry(
    left_wrist_y: np.ndarray,
    right_wrist_y: np.ndarray,
    left_elbow_deg: np.ndarray,
    right_elbow_deg: np.ndarray,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(left_wrist_y))
    axes[0].plot(t, left_wrist_y, label="Left wrist Y", color="#2196F3")
    axes[0].plot(
        t, right_wrist_y, label="Right wrist Y", color="#F44336", linestyle="--"
    )
    axes[0].set_ylabel("Y (pixels)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Wrist Height")
    axes[1].plot(t, left_elbow_deg, label="Left elbow angle", color="#4CAF50")
    axes[1].plot(
        t, right_elbow_deg, label="Right elbow angle", color="#FF9800", linestyle="--"
    )
    axes[1].set_ylabel("Degrees")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Elbow Angles")
    axes[1].set_xlabel("Frame")
    fig.suptitle("Bilateral Symmetry", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_harmonic_wave_patterns(
    bar_cy: np.ndarray,
    fps: float,
    wave_features: dict,
    out_path: str,
) -> None:
    valid = bar_cy[~np.isnan(bar_cy)]
    if len(valid) < 8:
        return

    from scipy.fft import rfft, rfftfreq

    t = np.arange(len(valid)) / max(fps, 1e-6)
    centered = valid - np.nanmean(valid)
    fft = np.abs(rfft(centered)) ** 2
    freqs = rfftfreq(len(centered), d=1.0 / max(fps, 1e-6))

    band = wave_features.get("frequency", {}).get("band_power", {})
    harmonic = wave_features.get("harmonic", {})
    dom = float(wave_features.get("frequency", {}).get("dominant_hz", 0.0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, valid, color="#1565C0", linewidth=1.4)
    axes[0, 0].set_title("Bar Path Harmonic Signal")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Bar Y (px)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs[1:], fft[1:], color="#C62828", linewidth=1.2)
    axes[0, 1].axvline(dom, color="#2E7D32", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title(f"Frequency Spectrum (dominant={dom:.3f} Hz)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Power")
    axes[0, 1].grid(True, alpha=0.3)

    names = ["slow", "medium", "fast", "harmonic"]
    vals = [float(band.get(k, 0.0)) for k in names]
    axes[1, 0].bar(names, vals, color=["#5E35B1", "#1E88E5", "#FB8C00", "#00897B"])
    axes[1, 0].set_title("Spectral Band Power")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    osc_count = int(harmonic.get("oscillation_count", 0))
    is_h = bool(harmonic.get("is_harmonic", False))
    summary = [
        f"Oscillation count: {osc_count}",
        f"Harmonic pattern: {'YES' if is_h else 'NO'}",
        f"Spectral entropy: {wave_features.get('frequency', {}).get('spectral_entropy', 0.0):.3f}",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Harmonic Summary")
    axes[1, 1].text(0.02, 0.95, "\n".join(summary), va="top", fontsize=11)

    fig.suptitle("Harmonic Wave Pattern Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
