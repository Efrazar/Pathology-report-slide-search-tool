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
import time
import zipfile

import numpy as np
import requests

from prss.slides.data import FOLDS, PANNUKE, load_fold

URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{k}.zip"
ZIPS = PANNUKE / "zips"


def download(k, tries=30):
    """Download fold k, resuming after a dropped connection instead of starting again.

    How resuming works: an HTTP "Range: bytes=N-" header asks the server for the file from
    byte N onwards. We keep what has arrived in a .part file, and after a drop ask only for
    the rest. A .part file is renamed to .zip only once it opens as a valid zip, so a broken
    download never looks finished (this also catches a zip left half done by wget).
    """
    dest, part, url = ZIPS / f"fold_{k}.zip", ZIPS / f"fold_{k}.part", URL.format(k=k)
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"fold {k}: zip already here, skipping download")
        return dest
    ZIPS.mkdir(parents=True, exist_ok=True)
    if dest.exists():                              # an unfinished zip: resume it
        dest.rename(part)
    print(f"fold {k}: downloading {url}")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url, stream=True, timeout=60,
                              headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:           # "range not satisfiable": we already have it all
                    break
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the Range header: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as fh:
                    for i, chunk in enumerate(r.iter_content(chunk_size=1 << 20)):
                        fh.write(chunk)
                        if i % 100 == 0:
                            print(f"  {fh.tell() / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
            break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.Timeout) as e:
            got = part.stat().st_size if part.exists() else 0
            print(f"  connection dropped at {got / 1e6:.0f} MB ({type(e).__name__}); "
                  f"retry {attempt}/{tries} in 5 s")
            time.sleep(5)
    else:
        raise SystemExit(f"fold {k}: gave up after {tries} tries; run the script again to resume")
    if not zipfile.is_zipfile(part):
        raise SystemExit(f"fold {k}: download finished but is not a valid zip. Delete {part} and run again.")
    part.rename(dest)
    return dest


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
