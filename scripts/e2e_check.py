"""End-to-end smoke check: drive both full pipelines on real inputs and
assert concrete, physically meaningful outcomes. Exits non-zero on any
failure so it can gate CI or a manual 'does it work' check.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seisproc.coherence import coherence
from seisproc.nmo import nmo_correct, stack, velocity_function
from seisproc.processing import agc, bandpass
from seisproc.segy import read_segy, write_segy
from seisproc.synthetic import DEFAULT_LAYERS, cmp_gather
from seisproc.velocity import pick_velocities, semblance
from seisproc.volume import Volume, from_npy, read_segy_volume, write_segy_volume

ok = True


def check(label, condition, detail=""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")


print("=" * 70)
print("PRE-STACK PIPELINE  (synthetic gather -> SEG-Y -> process -> stack)")
print("=" * 70)

# 1. generate + round-trip through SEG-Y on disk
data, t, offsets = cmp_gather(noise=0.15, seed=1)
with tempfile.TemporaryDirectory() as d:
    path = str(Path(d) / "gather.sgy")
    write_segy(path, data, dt=0.002, offsets=offsets)
    sd = read_segy(path)
    check("SEG-Y write/read round-trip", np.allclose(sd.data, data.astype(np.float32), rtol=1e-5),
          f"{sd.n_traces} traces x {sd.n_samples} samples, dt={sd.dt*1e3:.0f} ms")

# 2. processing chain runs and changes the data
proc = agc(bandpass(sd.data, sd.dt, 8, 60), sd.dt, 0.4)
check("bandpass + AGC produce finite output", np.all(np.isfinite(proc)))

# 3. semblance + auto-pick recovers true velocities
vels = np.arange(1300.0, 3300.0, 25.0)
t0 = time.perf_counter()
spec = semblance(sd.data, sd.offsets, sd.dt, vels)
times, pvels, _ = pick_velocities(spec, vels, sd.dt)
dt_sem = time.perf_counter() - t0
check("semblance picks >= number of layers", len(times) >= len(DEFAULT_LAYERS),
      f"{len(times)} picks in {dt_sem:.2f} s")
worst = 0.0
for layer in DEFAULT_LAYERS:
    k = np.abs(times - layer.t0).argmin()
    worst = max(worst, abs(pvels[k] - layer.v_rms) / layer.v_rms)
check("picked v_rms within 5% of truth (all layers)", worst < 0.05,
      f"worst error {worst*100:.1f}%")

# 4. NMO + stack improves SNR vs a single trace
v_t = velocity_function(sd.t, times, pvels)
clean, _, _ = cmp_gather(noise=0.0)
ref = stack(nmo_correct(clean, offsets, 0.002, velocity_function(t, [l.t0 for l in DEFAULT_LAYERS], [l.v_rms for l in DEFAULT_LAYERS])))
stk = stack(nmo_correct(sd.data, sd.offsets, sd.dt, v_t))
single_snr = np.std(clean[0]) / np.std((data - clean)[0])
stack_snr = np.std(ref) / np.std(stk - ref)
check("stacking improves SNR over single trace", stack_snr > 1.5 * single_snr,
      f"single={single_snr:.1f}  stack={stack_snr:.1f}  ({stack_snr/single_snr:.1f}x)")

print()
print("=" * 70)
print("POST-STACK PIPELINE  (F3 volume -> coherence -> attribute SEG-Y)")
print("=" * 70)

f3 = Path("data/f3/data/train/train_seismic.npy")
if not f3.exists():
    print(f"  [SKIP] {f3} not present — run scripts/download_f3.py")
else:
    vol = from_npy(str(f3), dt=0.004)
    vol.data = vol.data.astype(np.float32)
    check("F3 volume loads", vol.data.shape == (401, 701, 255), f"shape {vol.data.shape}")

    t0 = time.perf_counter()
    coh = coherence(vol.data, vol.dt)
    dt_coh = time.perf_counter() - t0
    check("coherence in (0,1], finite", coh.min() >= 0 and coh.max() <= 1.0 + 1e-6 and np.all(np.isfinite(coh)),
          f"median {np.median(coh):.3f} in {dt_coh:.1f} s")
    low = 100.0 * (coh < 0.5).mean()
    check("faults flagged but not everywhere (1-25%)", 1.0 < low < 25.0, f"{low:.1f}% below 0.5")

    # export a cropped attribute back to SEG-Y and reload it (full cube write is slow)
    sub = Volume(data=coh[:30, :40].astype(np.float32), ilines=vol.ilines[:30],
                 xlines=vol.xlines[:40], dt=vol.dt)
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "coh.sgy")
        write_segy_volume(out, sub)
        reloaded = read_segy_volume(out)
        check("coherence attribute SEG-Y round-trips", np.allclose(reloaded.data, sub.data, rtol=1e-5),
              f"{reloaded.data.shape}")

print()
print("=" * 70)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
print("=" * 70)
sys.exit(0 if ok else 1)
