"""Semblance-based coherence (Marfurt-style C2) for fault highlighting.

Coherence at each trace position measures how similar a trace is to its
spatial neighbours over a vertical analysis window: 1 for laterally
continuous reflectors, dropping sharply across faults, channels, and
other discontinuities. Faults appear as low-coherence lineaments,
clearest on time slices.

For a cell of N neighbouring traces a_j(t):

    C(t) = sum_w [ (sum_j a_j)^2 ] / ( N * sum_w [ sum_j a_j^2 ] )

where sum_w is the vertical window sum. Both numerator and denominator
reduce to box means, so the whole cube is three uniform filters.
"""

import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d


def coherence(data: np.ndarray, dt: float, window: float = 0.02, cell: int = 3) -> np.ndarray:
    """Semblance coherence in (0, 1].

    data: (n_il, n_xl, n_samples) volume or (n_traces, n_samples) section.
    window: vertical analysis window in seconds.
    cell: spatial box size in traces (3 -> 3x3 neighbours in 3D).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 3:
        size = (cell, cell, 1)
    elif data.ndim == 2:
        size = (cell, 1)
    else:
        raise ValueError(f"expected 2D or 3D data, got ndim={data.ndim}")

    # spatial box means: mean = S1/N, mean_sq = S2/N, so the N's cancel
    mean = uniform_filter(data, size=size, mode="nearest")
    mean_sq = uniform_filter(data**2, size=size, mode="nearest")

    n_w = max(1, int(round(window / dt)))
    num = uniform_filter1d(mean**2, size=n_w, axis=-1, mode="nearest")
    den = uniform_filter1d(mean_sq, size=n_w, axis=-1, mode="nearest")
    return num / (den + 1e-12 * den.max() + 1e-300)


def fault_likelihood(coh: np.ndarray) -> np.ndarray:
    """1 - coherence, rescaled to [0, 1]: high values flag discontinuities."""
    fl = 1.0 - coh
    peak = fl.max()
    return fl / peak if peak > 0 else fl
