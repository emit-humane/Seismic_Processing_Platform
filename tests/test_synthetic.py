import numpy as np

from seisproc.synthetic import DEFAULT_LAYERS, Layer, cmp_gather, ricker


def test_ricker_peak_at_zero_and_unit_amplitude():
    t, w = ricker(25.0, 0.002)
    assert w.max() == w[len(w) // 2] == 1.0
    assert t[len(t) // 2] == 0.0


def test_ricker_dominant_frequency():
    dt = 0.001
    t, w = ricker(30.0, dt, length=0.512)
    freqs = np.fft.rfftfreq(len(w), dt)
    spec = np.abs(np.fft.rfft(w))
    assert abs(freqs[spec.argmax()] - 30.0) < 3.0


def test_gather_shape_and_axes():
    data, t, offsets = cmp_gather(dt=0.002, t_max=2.0)
    assert data.shape == (len(offsets), len(t))
    assert np.isclose(t[-1], 2.0)


def test_event_arrives_at_hyperbolic_time():
    layer = Layer(t0=0.8, v_rms=2000.0, amplitude=1.0)
    offsets = np.array([1500.0])
    data, t, _ = cmp_gather(layers=[layer], offsets=offsets, dt=0.002, noise=0.0)
    t_expected = np.sqrt(0.8**2 + (1500.0 / 2000.0) ** 2)
    t_peak = t[np.abs(data[0]).argmax()]
    assert abs(t_peak - t_expected) < 0.004  # within 2 samples


def test_noise_is_reproducible():
    a, _, _ = cmp_gather(noise=0.1, seed=42)
    b, _, _ = cmp_gather(noise=0.1, seed=42)
    np.testing.assert_array_equal(a, b)
