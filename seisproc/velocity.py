"""Semblance velocity analysis.

Semblance at (t0, v) measures the coherence of a hyperbolic event with
that velocity: NMO-correct the gather at constant velocity v, then
S(t0) = (sum_i a_i)^2 / (N * sum_i a_i^2), smoothed over a time window.
S is 1 for perfectly coherent events, ~1/N for noise.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .nmo import nmo_correct


def semblance(
    data: np.ndarray,
    offsets: np.ndarray,
    dt: float,
    velocities: np.ndarray,
    smooth: float = 0.04,
) -> np.ndarray:
    """Semblance spectrum, shape (n_velocities, n_samples)."""
    n_traces, n_samples = data.shape
    win = max(1, int(round(smooth / dt)))
    spectrum = np.zeros((len(velocities), n_samples))

    for j, v in enumerate(velocities):
        corrected = nmo_correct(data, offsets, dt, v, stretch_mute=None)
        num = uniform_filter1d(corrected.sum(axis=0) ** 2, size=win, mode="nearest")
        den = uniform_filter1d((corrected**2).sum(axis=0), size=win, mode="nearest")
        spectrum[j] = num / (n_traces * den + 1e-12)
    return spectrum


def pick_velocities(
    spectrum: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    min_semblance: float = 0.3,
    min_separation: float = 0.15,
):
    """Automatic velocity picks from semblance peaks.

    Finds peaks of the velocity-maximum semblance trace, then reads off
    the best velocity at each peak time. Returns (times, vels, values).
    """
    best = spectrum.max(axis=0)
    distance = max(1, int(round(min_separation / dt)))
    peaks, _ = find_peaks(best, height=min_semblance, distance=distance)
    times = peaks * dt
    vels = velocities[spectrum[:, peaks].argmax(axis=0)]
    return times, vels, best[peaks]
