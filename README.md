# Seismic Workbench

A lightweight seismic processing & interpretation package in Python:
SEG-Y I/O, gather visualization, signal processing, semblance velocity
analysis, NMO correction, and CMP stacking — validated against a built-in
synthetic forward model with known ground truth.

## Quick start

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Generate a synthetic CMP gather in the sidebar (or upload a pre-stack
SEG-Y), then walk the pipeline: view → filter/AGC → semblance →
pick velocities → NMO → stack.

## Library

```python
from seisproc.synthetic import cmp_gather
from seisproc.velocity import semblance, pick_velocities
from seisproc.nmo import nmo_correct, stack, velocity_function

data, t, offsets = cmp_gather(noise=0.1)
vels = np.arange(1300, 3300, 25.0)
spec = semblance(data, offsets, 0.002, vels)
times, picks, _ = pick_velocities(spec, vels, 0.002)
v_t = velocity_function(t, times, picks)
section = stack(nmo_correct(data, offsets, 0.002, v_t))
```

Convention: arrays are `(n_traces, n_samples)`, time axis last; seconds,
metres, m/s throughout.

## Validation

`pytest tests/` — the suite proves the physics, not just the plumbing:
semblance peaks recover true RMS velocities within 5%, NMO flattens
synthetic hyperbolic events, and stacking improves SNR over a single
trace as theory predicts.

## Roadmap

See [ROADMAP.md](ROADMAP.md). M1 (this) — pre-stack pipeline.
M2 — post-stack attributes on F3 Netherlands. M3 — one interpretation
feature (coherence faults or assisted horizon tracking).

Future work (out of scope by design): inversion, ML fault/salt detection,
3D rendering, full interpretation workspace.
