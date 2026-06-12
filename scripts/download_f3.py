"""Download the F3 Netherlands benchmark volume (Zenodo record 3755060).

This is the facies-classification benchmark release (Alaudah et al. 2019):
~1.2 GB zip containing the F3 seismic as .npy volumes plus facies labels.
Seismic amplitude volumes are (n_ilines, n_xlines, n_samples), dt = 4 ms.

Usage:
    python scripts/download_f3.py [target_dir]   # default: data/f3
Then in the app: Post-stack mode -> "F3 .npy volume" -> point at
data/f3/data/train/train_seismic.npy
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/record/3755060/files/data.zip"


def _progress(blocks, block_size, total):
    done = blocks * block_size
    pct = 100.0 * done / total if total > 0 else 0
    sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/f3")
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "data.zip"

    if not zip_path.exists():
        print(f"Downloading {URL} (~1.2 GB) ...")
        urllib.request.urlretrieve(URL, zip_path, reporthook=_progress)
        print()
    else:
        print(f"{zip_path} already present, skipping download.")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)

    npys = sorted(target.rglob("*seismic*.npy"))
    print("Seismic volumes found:")
    for p in npys:
        print(f"  {p}")


if __name__ == "__main__":
    main()
