"""QC of coherence on the real F3 Netherlands volume (M3 exit criterion).

Loads the Zenodo benchmark train volume, computes semblance coherence on
the full cube, and writes comparison figures (amplitude | coherence |
fault overlay) to docs/img/. F3's polygonal fault systems should appear
as clear low-coherence lineament networks on time slices.

Usage: python scripts/f3_coherence_qc.py [path-to-train_seismic.npy]
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seisproc.coherence import coherence
from seisproc.volume import from_npy

NPY = sys.argv[1] if len(sys.argv) > 1 else "data/f3/data/train/train_seismic.npy"
OUT = Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)

print(f"Loading {NPY} ...")
vol = from_npy(NPY, dt=0.004)
vol.data = vol.data.astype(np.float32)
print(f"  volume: {vol.data.shape}, dt={vol.dt * 1e3:.0f} ms")

print("Computing coherence (full cube) ...")
t0 = time.perf_counter()
coh = coherence(vol.data, vol.dt, window=0.02, cell=3)
print(f"  done in {time.perf_counter() - t0:.1f} s | "
      f"min={coh.min():.3f} median={np.median(coh):.3f}")

amp_max = np.percentile(np.abs(vol.data), 98)

for t_s in (0.50, 0.75, 0.95):
    k = int(round(t_s / vol.dt))
    amp = vol.data[:, :, k]
    c = coh[:, :, k]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    axes[0].imshow(amp.T, cmap="gray_r", vmin=-amp_max, vmax=amp_max,
                   aspect="auto", origin="lower")
    axes[0].set_title(f"Amplitude — t = {t_s:.2f} s")
    axes[1].imshow(c.T, cmap="gray", vmin=np.percentile(c, 2), vmax=1.0,
                   aspect="auto", origin="lower")
    axes[1].set_title("Coherence (semblance)")

    axes[2].imshow(amp.T, cmap="gray_r", vmin=-amp_max, vmax=amp_max,
                   aspect="auto", origin="lower")
    mask = (c < 0.5).T.astype(float)
    rgba = np.zeros(mask.shape + (4,))
    rgba[..., 0] = 0.85
    rgba[..., 3] = 0.7 * mask
    axes[2].imshow(rgba, aspect="auto", origin="lower")
    axes[2].set_title("Faults (coherence < 0.5) on amplitude")

    for ax in axes:
        ax.set_xlabel("Inline index")
    axes[0].set_ylabel("Crossline index")
    fig.tight_layout()
    name = OUT / f"f3_coherence_t{int(t_s * 1000)}ms.png"
    fig.savefig(name, dpi=110)
    plt.close(fig)
    low_pct = 100.0 * (c < 0.5).mean()
    print(f"  {name}  ({low_pct:.1f}% of slice below 0.5 coherence)")

# inline section comparison
il_idx = vol.data.shape[0] // 2
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
axes[0].imshow(vol.data[il_idx].T, cmap="gray_r", vmin=-amp_max, vmax=amp_max,
               aspect="auto", extent=[0, vol.data.shape[1], vol.t[-1], 0])
axes[0].set_title(f"Amplitude — inline index {il_idx}")
axes[0].set_ylabel("Time (s)")
axes[1].imshow(coh[il_idx].T, cmap="gray", vmin=np.percentile(coh[il_idx], 2),
               vmax=1.0, aspect="auto", extent=[0, vol.data.shape[1], vol.t[-1], 0])
axes[1].set_title("Coherence")
axes[1].set_xlabel("Crossline index")
axes[1].set_ylabel("Time (s)")
fig.tight_layout()
fig.savefig(OUT / "f3_coherence_inline.png", dpi=110)
plt.close(fig)
print(f"  {OUT / 'f3_coherence_inline.png'}")
print("QC complete.")
