"""Step 10 (Track 2): download PanNuke and keep only what we need: the tiles and their tissue type.

PanNuke: about 7,900 H&E tiles, 256 x 256 pixels, from 19 tissue types, in 3 official folds.
Licence CC BY-NC-SA 4.0 (non-commercial, share alike), see DATA_SOURCES.md.

What this does, per fold:
1. Download fold_k.zip from the Warwick TIA site (about 0.7 GB each), unless it is already in
   data/raw/pannuke/zips/. A dropped connection is retried and resumed, not restarted.
   If it still fails, run `wget -c` in that folder (see README), or download the zips in a
   browser from https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke and put them there.
2. Copy images.npy and types.npy out of the zip. The zip also holds the nucleus masks; we skip
   them, because organ classification does not need them.
3. Convert the images from float64 to uint8 in chunks, which cuts each fold from about 4 GB to
   about 0.5 GB (see src/prss/slides/data.py for why this is lossless).

Output: data/raw/pannuke/fold{1,2,3}/images.npy and types.npy, then a table of tiles per type.

Disk: about 6.5 GB while it runs (zips plus one temporary float64 file), about 3.6 GB after.
Delete data/raw/pannuke/zips/ afterwards if you need the space.
"""
import shutil
import zipfile

import numpy as np

from prss.download import fetch
from prss.slides.data import FOLDS, PANNUKE, load_fold

URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{k}.zip"
ZIPS = PANNUKE / "zips"


def download(k):
    """Fold k's zip, downloaded with resume and retry (see src/prss/download.py)."""
    dest = ZIPS / f"fold_{k}.zip"
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"fold {k}: zip already here, skipping download")
        return dest
    if dest.exists():                     # an unfinished zip, e.g. from wget: resume it
        dest.rename(dest.with_name(dest.name + ".part"))
    print(f"fold {k}: downloading {URL.format(k=k)}")
    return fetch(URL.format(k=k), dest, check=zipfile.is_zipfile, label=f"fold {k}")


def member(zf, name):
    """Find e.g. 'Fold 1/images/fold1/images.npy' without hard-coding the folder names."""
    hits = [m for m in zf.namelist() if m.endswith("/" + name) and "__MACOSX" not in m]
    if len(hits) != 1:
        raise SystemExit(f"expected one {name} in the zip, found {hits}")
    return hits[0]


def extract(k, zpath):
    out = PANNUKE / f"fold{k}"
    if (out / "images.npy").exists() and (out / "types.npy").exists():
        print(f"fold {k}: already extracted")
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "images_float.npy"
    with zipfile.ZipFile(zpath) as zf:
        with zf.open(member(zf, "types.npy")) as src, open(out / "types.npy", "wb") as dst:
            shutil.copyfileobj(src, dst)
        with zf.open(member(zf, "images.npy")) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)

    src = np.load(tmp, mmap_mode="r")
    print(f"fold {k}: {src.shape} {src.dtype} -> uint8")
    if src.min() < 0 or src.max() > 255:      # a cheap check that the pixels really are 0-255
        raise SystemExit("pixel values outside 0-255: uint8 would change them, stopping")
    dst = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=src.shape)
    for i in range(0, len(src), 256):         # 256 tiles at a time keeps RAM use near 400 MB
        dst[i:i + 256] = np.rint(src[i:i + 256]).astype(np.uint8)
    dst.flush()
    del src, dst
    tmp.unlink()


for k in FOLDS:
    extract(k, download(k))

print("\ntiles per tissue type")
counts = {}
for k in FOLDS:
    images, types = load_fold(k)
    assert len(images) == len(types), f"fold {k}: {len(images)} images but {len(types)} labels"
    for t in types:
        counts.setdefault(t, [0, 0, 0])[k - 1] += 1
print(f"{'type':<16}{'fold1':>7}{'fold2':>7}{'fold3':>7}")
for t, c in sorted(counts.items(), key=lambda kv: -sum(kv[1])):
    print(f"{t:<16}{c[0]:>7}{c[1]:>7}{c[2]:>7}")
print(f"{'total':<16}" + "".join(f"{sum(c[i] for c in counts.values()):>7}" for i in range(3)))
