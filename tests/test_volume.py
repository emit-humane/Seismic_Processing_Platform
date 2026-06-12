import numpy as np

from seisproc.synthetic import synthetic_volume
from seisproc.volume import Volume, from_npy, read_segy_volume, write_segy_volume


def _make_volume():
    data, t = synthetic_volume(n_il=10, n_xl=12, dt=0.004)
    return Volume(
        data=data.astype(np.float32),
        ilines=np.arange(100, 110),
        xlines=np.arange(300, 312),
        dt=0.004,
    )


def test_volume_segy_roundtrip(tmp_path):
    vol = _make_volume()
    path = str(tmp_path / "vol.sgy")
    write_segy_volume(path, vol)
    loaded = read_segy_volume(path)

    np.testing.assert_array_equal(loaded.ilines, vol.ilines)
    np.testing.assert_array_equal(loaded.xlines, vol.xlines)
    assert np.isclose(loaded.dt, vol.dt)
    np.testing.assert_allclose(loaded.data, vol.data, rtol=1e-6)


def test_slicing_matches_direct_indexing():
    vol = _make_volume()
    np.testing.assert_array_equal(vol.inline(105), vol.data[5])
    np.testing.assert_array_equal(vol.xline(303), vol.data[:, 3])
    np.testing.assert_array_equal(vol.time_slice(0.4), vol.data[:, :, 100])


def test_from_npy(tmp_path):
    vol = _make_volume()
    path = str(tmp_path / "vol.npy")
    np.save(path, vol.data)
    loaded = from_npy(path, dt=0.004, first_iline=100, first_xline=300)
    np.testing.assert_array_equal(loaded.data, vol.data)
    np.testing.assert_array_equal(loaded.ilines, vol.ilines)


def test_summary():
    vol = _make_volume()
    s = vol.summary()
    assert s["n_ilines"] == 10
    assert s["iline_range"] == (100, 109)
    assert np.isclose(s["dt_ms"], 4.0)
