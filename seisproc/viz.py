"""Matplotlib displays: wiggle, variable density, semblance spectrum."""

import matplotlib.pyplot as plt
import numpy as np


def wiggle(data, dt, offsets=None, ax=None, gain=1.0, fill=True, color="black"):
    """Variable-area wiggle plot. Traces drawn vertically, time down."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    n_traces, n_samples = data.shape
    t = np.arange(n_samples) * dt
    x = np.arange(n_traces, dtype=float) if offsets is None else np.asarray(offsets, float)
    spacing = np.median(np.diff(np.sort(x))) if n_traces > 1 else 1.0
    peak = np.abs(data).max() or 1.0
    scale = gain * spacing / peak

    for i in range(n_traces):
        trace = x[i] + data[i] * scale
        ax.plot(trace, t, color=color, linewidth=0.5)
        if fill:
            ax.fill_betweenx(t, x[i], trace, where=trace > x[i], color=color, linewidth=0)

    ax.set_ylim(t[-1], 0)
    ax.set_xlabel("Offset (m)" if offsets is not None else "Trace")
    ax.set_ylabel("Time (s)")
    return ax


def density(data, dt, offsets=None, ax=None, cmap="gray_r", clip=0.95):
    """Variable-density (image) display with percentile amplitude clipping."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    n_traces, n_samples = data.shape
    vmax = np.percentile(np.abs(data), clip * 100) or 1.0
    extent = [
        0 if offsets is None else offsets[0],
        n_traces if offsets is None else offsets[-1],
        (n_samples - 1) * dt,
        0,
    ]
    im = ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent)
    ax.set_xlabel("Offset (m)" if offsets is not None else "Trace")
    ax.set_ylabel("Time (s)")
    return ax, im


def semblance_panel(spectrum, velocities, dt, picks=None, ax=None, cmap="viridis"):
    """Semblance spectrum with optional (times, vels) picks overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))
    n_vel, n_samples = spectrum.shape
    extent = [velocities[0], velocities[-1], (n_samples - 1) * dt, 0]
    im = ax.imshow(spectrum.T, aspect="auto", cmap=cmap, extent=extent)
    if picks is not None:
        times, vels = picks
        ax.plot(vels, times, "w-o", markersize=5, markeredgecolor="red", linewidth=1.5)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Time (s)")
    return ax, im
