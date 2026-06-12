"""Trace-domain signal processing: bandpass, normalization, AGC.

All functions take and return (n_traces, n_samples) arrays and never
modify the input in place.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt


def bandpass(data: np.ndarray, dt: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. low/high are corner frequencies in Hz."""
    nyq = 0.5 / dt
    if not 0 < low < high < nyq:
        raise ValueError(f"need 0 < low < high < Nyquist ({nyq:.1f} Hz), got [{low}, {high}]")
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def normalize(data: np.ndarray, mode: str = "rms") -> np.ndarray:
    """Per-trace normalization. mode: 'rms' or 'max'."""
    if mode == "rms":
        scale = np.sqrt(np.mean(data**2, axis=-1, keepdims=True))
    elif mode == "max":
        scale = np.abs(data).max(axis=-1, keepdims=True)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    safe = np.where(scale > 0, scale, 1.0)
    return data / safe


def agc(data: np.ndarray, dt: float, window: float = 0.5) -> np.ndarray:
    """Automatic gain control: divide by sliding-window RMS amplitude.

    window is the AGC gate length in seconds.
    """
    n = max(1, int(round(window / dt)))
    rms = np.sqrt(uniform_filter1d(data**2, size=n, axis=-1, mode="nearest"))
    eps = 1e-8 * (rms.max() if rms.max() > 0 else 1.0)
    return data / (rms + eps)
