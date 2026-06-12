"""SEG-Y reading and writing via segyio.

Reads any SEG-Y unstructured (trace-sequential), which covers both
pre-stack gathers and 2D post-stack lines. Geometry-aware 3D loading
is a Milestone 2 concern.
"""

from dataclasses import dataclass, field

import numpy as np
import segyio


@dataclass
class SegyData:
    data: np.ndarray  # (n_traces, n_samples), float32
    dt: float  # sample interval, seconds
    offsets: np.ndarray  # source-receiver offset per trace, metres
    text_header: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt

    def summary(self) -> dict:
        return {
            "n_traces": self.n_traces,
            "n_samples": self.n_samples,
            "dt_ms": self.dt * 1e3,
            "trace_length_s": (self.n_samples - 1) * self.dt,
            "offset_min": float(self.offsets.min()) if len(self.offsets) else None,
            "offset_max": float(self.offsets.max()) if len(self.offsets) else None,
            **self.extra,
        }


def _decode_text_header(raw: bytes) -> str:
    """segyio returns the textual header already converted to ASCII bytes."""
    text = raw.decode("ascii", errors="replace")
    return "\n".join(text[i : i + 80] for i in range(0, len(text), 80))


def read_segy(path: str, fallback_dt: float = 0.002) -> SegyData:
    """Read a SEG-Y file without assuming inline/crossline geometry."""
    with segyio.open(path, ignore_geometry=True) as f:
        data = np.asarray(f.trace.raw[:], dtype=np.float32)
        dt_us = segyio.tools.dt(f, fallback_dt=fallback_dt * 1e6)
        offsets = np.asarray(f.attributes(segyio.TraceField.offset)[:], dtype=float)
        text = _decode_text_header(f.text[0])
        cdps = np.asarray(f.attributes(segyio.TraceField.CDP)[:])
        extra = {
            "format_code": int(f.bin[segyio.BinField.Format]),
            "cdp_range": (int(cdps.min()), int(cdps.max())),
        }
    return SegyData(data=data, dt=dt_us / 1e6, offsets=offsets, text_header=text, extra=extra)


def write_segy(path: str, data: np.ndarray, dt: float, offsets: np.ndarray | None = None) -> None:
    """Write a (n_traces, n_samples) array as an unstructured IEEE-float SEG-Y."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    n_traces, n_samples = data.shape
    if offsets is None:
        offsets = np.zeros(n_traces)
    dt_us = int(round(dt * 1e6))

    spec = segyio.spec()
    spec.samples = list(np.arange(n_samples) * dt * 1e3)  # label only
    spec.tracecount = n_traces
    spec.format = 5  # 4-byte IEEE float

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Samples] = n_samples
        f.bin[segyio.BinField.Format] = 5
        for i in range(n_traces):
            f.header[i] = {
                segyio.TraceField.offset: int(round(offsets[i])),
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                segyio.TraceField.CDP: 1,
                segyio.TraceField.TraceNumber: i + 1,
            }
            f.trace[i] = data[i]
