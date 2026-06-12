"""Synthetic forward modelling: Ricker wavelet, convolutional model, CMP gathers.

The CMP gather generator places events on hyperbolic moveout curves
t(x) = sqrt(t0^2 + (x/v_rms)^2), which is the ground truth that the
semblance/NMO/stack pipeline must recover. Velocities here are RMS
velocities, matching what semblance analysis estimates.
"""

from dataclasses import dataclass

import numpy as np


def ricker(f_peak: float, dt: float, length: float = 0.128):
    """Ricker (Mexican hat) wavelet.

    Returns (t, w) with t centred on zero. Peak frequency f_peak in Hz.
    """
    n = int(round(length / dt)) // 2
    t = np.arange(-n, n + 1) * dt
    a = (np.pi * f_peak * t) ** 2
    return t, (1.0 - 2.0 * a) * np.exp(-a)


@dataclass(frozen=True)
class Layer:
    """A single reflector for synthetic modelling."""

    t0: float  # zero-offset two-way time (s)
    v_rms: float  # RMS velocity down to this reflector (m/s)
    amplitude: float = 1.0


# A sensible default earth: velocities increase with depth.
DEFAULT_LAYERS = (
    Layer(t0=0.30, v_rms=1600.0, amplitude=1.0),
    Layer(t0=0.60, v_rms=1850.0, amplitude=-0.7),
    Layer(t0=0.95, v_rms=2100.0, amplitude=0.8),
    Layer(t0=1.30, v_rms=2400.0, amplitude=-0.5),
    Layer(t0=1.70, v_rms=2700.0, amplitude=0.6),
)


def reflectivity_to_trace(reflectivity: np.ndarray, dt: float, f_peak: float = 25.0) -> np.ndarray:
    """Convolutional model: trace = reflectivity * ricker."""
    _, w = ricker(f_peak, dt)
    return np.convolve(reflectivity, w, mode="same")


def cmp_gather(
    layers=DEFAULT_LAYERS,
    offsets: np.ndarray | None = None,
    dt: float = 0.002,
    t_max: float = 2.0,
    f_peak: float = 25.0,
    noise: float = 0.0,
    seed: int = 0,
):
    """Generate a synthetic CMP gather with hyperbolic moveout.

    noise is the std of additive white noise as a fraction of the peak
    absolute amplitude of the clean gather.

    Returns (data, t, offsets) with data shaped (n_traces, n_samples).
    """
    if offsets is None:
        offsets = np.arange(100.0, 3100.0, 100.0)
    offsets = np.asarray(offsets, dtype=float)

    n_samples = int(round(t_max / dt)) + 1
    t = np.arange(n_samples) * dt
    spikes = np.zeros((len(offsets), n_samples))

    for layer in layers:
        t_event = np.sqrt(layer.t0**2 + (offsets / layer.v_rms) ** 2)
        idx = t_event / dt
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = i0 + 1 < n_samples
        rows = np.arange(len(offsets))[valid]
        # linear interpolation of the spike between the two nearest samples
        spikes[rows, i0[valid]] += layer.amplitude * (1.0 - frac[valid])
        spikes[rows, i0[valid] + 1] += layer.amplitude * frac[valid]

    _, w = ricker(f_peak, dt)
    data = np.apply_along_axis(np.convolve, 1, spikes, w, mode="same")

    if noise > 0.0:
        rng = np.random.default_rng(seed)
        data = data + noise * np.abs(data).max() * rng.standard_normal(data.shape)

    return data, t, offsets


def synthetic_volume(
    n_il: int = 40,
    n_xl: int = 50,
    dt: float = 0.004,
    t_max: float = 1.0,
    f_peak: float = 30.0,
    noise: float = 0.02,
    seed: int = 0,
):
    """Small post-stack 3D volume: three reflectors draped over a dome,
    with a bright-spot amplitude anomaly on the middle one. Ground truth
    for attribute displays — the bright spot should light up in envelope
    and RMS amplitude.

    Returns (data, t) with data shaped (n_il, n_xl, n_samples).
    """
    n_samples = int(round(t_max / dt)) + 1
    t = np.arange(n_samples) * dt
    il = np.arange(n_il)[:, None]
    xl = np.arange(n_xl)[None, :]

    # structural high (time pull-up) in the volume centre
    dome = 0.06 * np.exp(-(((il - n_il / 2) / (n_il / 3)) ** 2 + ((xl - n_xl / 2) / (n_xl / 3)) ** 2))

    # bright spot: amplitude boost on a patch of the middle reflector
    bright = 1.0 + 1.5 * np.exp(
        -(((il - n_il * 0.5) / (n_il / 6)) ** 2 + ((xl - n_xl * 0.5) / (n_xl / 6)) ** 2
    ))

    events = [
        (0.30, 0.8, np.ones((n_il, n_xl))),
        (0.55, -0.6, bright),
        (0.80, 0.7, np.ones((n_il, n_xl))),
    ]

    spikes = np.zeros((n_il, n_xl, n_samples))
    for t0, amp, amap in events:
        idx = np.clip(np.round((t0 - dome) / dt).astype(int), 0, n_samples - 1)
        np.put_along_axis(spikes, idx[:, :, None], amp * amap[:, :, None], axis=2)

    _, w = ricker(f_peak, dt)
    data = np.apply_along_axis(np.convolve, 2, spikes, w, mode="same")

    if noise > 0.0:
        rng = np.random.default_rng(seed)
        data = data + noise * np.abs(data).max() * rng.standard_normal(data.shape)
    return data, t
