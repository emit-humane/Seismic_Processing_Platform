"""3D post-stack volumes: geometry-aware SEG-Y I/O and slicing.

Convention: volume data is (n_ilines, n_xlines, n_samples).
"""

from dataclasses import dataclass

import numpy as np
import segyio


@dataclass
class Volume:
    data: np.ndarray  # (n_il, n_xl, n_samples), float32
    ilines: np.ndarray
    xlines: np.ndarray
    dt: float  # seconds

    @property
    def n_samples(self) -> int:
        return self.data.shape[2]

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt

    def inline(self, il: int) -> np.ndarray:
        """Section at inline il: (n_xlines, n_samples)."""
        return self.data[int(np.where(self.ilines == il)[0][0])]

    def xline(self, xl: int) -> np.ndarray:
        """Section at crossline xl: (n_ilines, n_samples)."""
        return self.data[:, int(np.where(self.xlines == xl)[0][0])]

    def time_slice(self, t: float) -> np.ndarray:
        """Map view at time t: (n_ilines, n_xlines)."""
        i = int(round(t / self.dt))
        return self.data[:, :, np.clip(i, 0, self.n_samples - 1)]

    def summary(self) -> dict:
        return {
            "n_ilines": len(self.ilines),
            "n_xlines": len(self.xlines),
            "n_samples": self.n_samples,
            "dt_ms": self.dt * 1e3,
            "iline_range": (int(self.ilines[0]), int(self.ilines[-1])),
            "xline_range": (int(self.xlines[0]), int(self.xlines[-1])),
        }


def read_segy_volume(path: str, fallback_dt: float = 0.004) -> Volume:
    """Read a 3D SEG-Y with inline/crossline geometry (strict mode)."""
    with segyio.open(path) as f:
        data = segyio.tools.cube(f).astype(np.float32)
        dt_us = segyio.tools.dt(f, fallback_dt=fallback_dt * 1e6)
        ilines = np.asarray(f.ilines)
        xlines = np.asarray(f.xlines)
    return Volume(data=data, ilines=ilines, xlines=xlines, dt=dt_us / 1e6)


def write_segy_volume(path: str, vol: Volume) -> None:
    """Write a Volume as an inline-sorted IEEE-float 3D SEG-Y."""
    n_il, n_xl, n_samples = vol.data.shape
    dt_us = int(round(vol.dt * 1e6))

    spec = segyio.spec()
    spec.ilines = list(int(i) for i in vol.ilines)
    spec.xlines = list(int(x) for x in vol.xlines)
    spec.samples = list(np.arange(n_samples) * vol.dt * 1e3)
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5

    data = np.ascontiguousarray(vol.data, dtype=np.float32)
    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Samples] = n_samples
        f.bin[segyio.BinField.Format] = 5
        tr = 0
        for i, il in enumerate(vol.ilines):
            for j, xl in enumerate(vol.xlines):
                f.header[tr] = {
                    segyio.TraceField.INLINE_3D: int(il),
                    segyio.TraceField.CROSSLINE_3D: int(xl),
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                }
                f.trace[tr] = data[i, j]
                tr += 1


def from_npy(path: str, dt: float = 0.004, first_iline: int = 1, first_xline: int = 1) -> Volume:
    """Load a (n_il, n_xl, n_samples) .npy volume (e.g. the Zenodo F3
    facies benchmark, record 3755060) as a Volume."""
    data = np.load(path).astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {data.shape}")
    n_il, n_xl, _ = data.shape
    return Volume(
        data=data,
        ilines=np.arange(first_iline, first_iline + n_il),
        xlines=np.arange(first_xline, first_xline + n_xl),
        dt=dt,
    )
