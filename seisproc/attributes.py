"""Complex-trace (Hilbert) and amplitude attributes.

All functions operate along the last axis (time), so they work
unchanged on single traces, 2D sections (n_traces, n_samples), and
3D volumes (n_il, n_xl, n_samples).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert


def envelope(data: np.ndarray) -> np.ndarray:
    """Reflection strength: magnitude of the analytic signal."""
    return np.abs(hilbert(data, axis=-1))


def instantaneous_phase(data: np.ndarray) -> np.ndarray:
    """Phase of the analytic signal, radians in (-pi, pi]."""
    return np.angle(hilbert(data, axis=-1))


def instantaneous_frequency(data: np.ndarray, dt: float, smooth: float = 0.016) -> np.ndarray:
    """Time derivative of unwrapped instantaneous phase, in Hz.

    Raw instantaneous frequency spikes wherever the envelope is near
    zero, so a short smoothing window (seconds) is applied by default.
    """
    phase = np.unwrap(np.angle(hilbert(data, axis=-1)), axis=-1)
    freq = np.gradient(phase, dt, axis=-1) / (2.0 * np.pi)
    if smooth and smooth > 0:
        n = max(1, int(round(smooth / dt)))
        freq = uniform_filter1d(freq, size=n, axis=-1, mode="nearest")
    return freq


def rms_amplitude(data: np.ndarray, dt: float, window: float = 0.1) -> np.ndarray:
    """Sliding-window RMS amplitude; window length in seconds."""
    n = max(1, int(round(window / dt)))
    return np.sqrt(uniform_filter1d(data.astype(float) ** 2, size=n, axis=-1, mode="nearest"))


ATTRIBUTES = {
    "Amplitude": None,
    "Envelope": envelope,
    "Instantaneous phase": instantaneous_phase,
    "Instantaneous frequency": instantaneous_frequency,
    "RMS amplitude": rms_amplitude,
}


def compute(name: str, data: np.ndarray, dt: float) -> np.ndarray:
    """Dispatch by display name; 'Amplitude' returns the input."""
    func = ATTRIBUTES[name]
    if func is None:
        return data
    if func in (instantaneous_frequency, rms_amplitude):
        return func(data, dt)
    return func(data)
