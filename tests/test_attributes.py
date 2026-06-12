"""Attributes validated against analytic signals with known answers."""

import numpy as np

from seisproc.attributes import (
    compute,
    envelope,
    instantaneous_frequency,
    instantaneous_phase,
    rms_amplitude,
)
from seisproc.synthetic import ricker, synthetic_volume

DT = 0.002
INTERIOR = slice(100, -100)  # Hilbert/gradient edge effects excluded


def _t(n=1000):
    return np.arange(n) * DT


def test_envelope_recovers_am_modulation():
    t = _t()
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    signal = modulation * np.cos(2 * np.pi * 40.0 * t)
    env = envelope(signal[None, :])[0]
    np.testing.assert_allclose(env[INTERIOR], modulation[INTERIOR], rtol=0.05)


def test_instantaneous_frequency_of_pure_tone():
    t = _t()
    signal = np.cos(2 * np.pi * 30.0 * t)
    freq = instantaneous_frequency(signal[None, :], DT)[0]
    assert abs(np.median(freq[INTERIOR]) - 30.0) < 1.0


def test_instantaneous_phase_zero_at_ricker_peak():
    _, w = ricker(25.0, DT)
    trace = np.zeros(500)
    trace[200 : 200 + len(w)] = w
    phase = instantaneous_phase(trace[None, :])[0]
    peak = np.abs(trace).argmax()
    assert abs(phase[peak]) < 0.1  # zero-phase wavelet -> phase ~0 at peak


def test_rms_amplitude_of_steady_sine():
    t = _t()
    a = 3.0
    signal = a * np.sin(2 * np.pi * 25.0 * t)
    rms = rms_amplitude(signal[None, :], DT, window=0.2)[0]
    np.testing.assert_allclose(rms[INTERIOR], a / np.sqrt(2), rtol=0.05)


def test_attributes_preserve_3d_shape():
    data, t = synthetic_volume(n_il=8, n_xl=10)
    for name in ["Envelope", "Instantaneous phase", "Instantaneous frequency", "RMS amplitude"]:
        out = compute(name, data, 0.004)
        assert out.shape == data.shape


def test_bright_spot_lights_up_in_envelope():
    data, t = synthetic_volume(noise=0.0)
    env = envelope(data)
    # window must span the dome pull-up (event is ~0.06 s shallower at centre)
    i_evt = int(round(0.55 / 0.004))
    window = env[:, :, i_evt - 20 : i_evt + 10].max(axis=2)
    centre = window[20, 25]
    corner = window[2, 2]
    assert centre > 2.0 * corner  # anomaly clearly brighter than background
