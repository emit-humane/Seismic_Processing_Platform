import numpy as np

from seisproc.segy import read_segy, write_segy
from seisproc.synthetic import cmp_gather


def test_segy_roundtrip(tmp_path):
    data, t, offsets = cmp_gather(noise=0.05)
    path = str(tmp_path / "gather.sgy")

    write_segy(path, data, dt=0.002, offsets=offsets)
    loaded = read_segy(path)

    assert loaded.n_traces == data.shape[0]
    assert loaded.n_samples == data.shape[1]
    assert np.isclose(loaded.dt, 0.002)
    np.testing.assert_array_equal(loaded.offsets, offsets)
    np.testing.assert_allclose(loaded.data, data.astype(np.float32), rtol=1e-6)


def test_summary_fields(tmp_path):
    data, _, offsets = cmp_gather()
    path = str(tmp_path / "gather.sgy")
    write_segy(path, data, dt=0.002, offsets=offsets)

    s = read_segy(path).summary()
    assert s["n_traces"] == data.shape[0]
    assert np.isclose(s["dt_ms"], 2.0)
    assert s["offset_max"] == offsets.max()
    assert s["format_code"] == 5
