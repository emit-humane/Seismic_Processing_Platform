import numpy as np

from seisproc.processing import agc, bandpass, normalize


def _sine_trace(freq, dt=0.002, n=1000):
    t = np.arange(n) * dt
    return np.sin(2 * np.pi * freq * t)


def test_bandpass_keeps_inband_rejects_outband():
    dt = 0.002
    inband = _sine_trace(30.0, dt)
    outband = _sine_trace(120.0, dt)
    data = np.vstack([inband + outband])

    out = bandpass(data, dt, low=10.0, high=60.0)

    # in-band energy preserved, out-of-band crushed (interior samples only —
    # zero-phase filtering leaves transients at the trace edges)
    interior = slice(100, -100)
    corr_in = np.corrcoef(out[0, interior], inband[interior])[0, 1]
    assert corr_in > 0.95
    residual = out[0, interior] - inband[interior]
    assert np.sqrt(np.mean(residual**2)) < 0.1 * np.sqrt(np.mean(outband**2))


def test_bandpass_rejects_bad_corners():
    data = np.zeros((1, 100))
    try:
        bandpass(data, 0.002, low=60.0, high=10.0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_normalize_rms_and_max():
    data = np.array([[1.0, -2.0, 3.0], [10.0, -20.0, 30.0]])
    rms = normalize(data, "rms")
    assert np.allclose(np.sqrt(np.mean(rms**2, axis=-1)), 1.0)
    mx = normalize(data, "max")
    assert np.allclose(np.abs(mx).max(axis=-1), 1.0)


def test_normalize_handles_dead_trace():
    data = np.zeros((1, 50))
    out = normalize(data, "rms")
    assert np.all(np.isfinite(out))


def test_agc_balances_amplitude_decay():
    dt = 0.002
    n = 1000
    t = np.arange(n) * dt
    decaying = np.exp(-2.0 * t) * np.sin(2 * np.pi * 25 * t)
    out = agc(decaying[None, :], dt, window=0.3)[0]

    early = np.sqrt(np.mean(out[:200] ** 2))
    late = np.sqrt(np.mean(out[-200:] ** 2))
    raw_ratio = np.sqrt(np.mean(decaying[:200] ** 2)) / np.sqrt(np.mean(decaying[-200:] ** 2))
    assert early / late < 0.1 * raw_ratio  # gain decay largely removed
