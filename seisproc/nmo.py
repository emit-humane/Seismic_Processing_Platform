"""Normal moveout correction and CMP stacking.

NMO maps each trace's amplitude at time t(x) = sqrt(t0^2 + (x/v(t0))^2)
back to t0. Far-offset shallow samples get stretched; samples whose
stretch factor (t - t0)/t0 exceeds the mute limit are zeroed.
"""

import numpy as np


def velocity_function(t: np.ndarray, pick_times, pick_vels) -> np.ndarray:
    """Interpolate (t0, v) picks onto the time axis, flat extrapolation."""
    pick_times = np.asarray(pick_times, dtype=float)
    pick_vels = np.asarray(pick_vels, dtype=float)
    order = np.argsort(pick_times)
    return np.interp(t, pick_times[order], pick_vels[order])


def nmo_correct(
    data: np.ndarray,
    offsets: np.ndarray,
    dt: float,
    velocity,
    stretch_mute: float | None = 0.5,
) -> np.ndarray:
    """Apply NMO correction.

    velocity: scalar (constant) or array of length n_samples giving v_rms(t0).
    stretch_mute: max allowed NMO stretch (t - t0)/t0; None disables muting.
    """
    n_traces, n_samples = data.shape
    t0 = np.arange(n_samples) * dt
    v = np.broadcast_to(np.asarray(velocity, dtype=float), t0.shape)

    corrected = np.zeros_like(data)
    for i, x in enumerate(offsets):
        t_src = np.sqrt(t0**2 + (x / v) ** 2)
        corrected[i] = np.interp(t_src, t0, data[i], left=0.0, right=0.0)
        if stretch_mute is not None:
            stretch = (t_src - t0) / np.maximum(t0, dt)
            corrected[i, stretch > stretch_mute] = 0.0
    return corrected


def stack(corrected: np.ndarray) -> np.ndarray:
    """Fold-normalized stack: mean over live (non-muted) traces per sample."""
    live = corrected != 0.0
    fold = live.sum(axis=0)
    total = corrected.sum(axis=0)
    return np.divide(total, fold, out=np.zeros_like(total), where=fold > 0)
