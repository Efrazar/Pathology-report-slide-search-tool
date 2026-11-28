"""Step 21 (Track 2b): download each chosen TCGA slide, cut tissue tiles, then delete the slide.

A whole slide is one huge image (often 100,000 x 80,000 pixels, 0.3 to 1.5 GB). We keep only
--tiles small tiles per slide, so disk use stays at one slide plus the tiles.

Per slide:
1. Download from the GDC API with resume and retry (src/prss/download.py).
2. Find the tissue on a small thumbnail. Glass is white and grey; stained tissue is coloured.
   "Saturation" (how coloured a pixel is) separates the two, and Otsu's method picks the
   cut-off for each slide automatically, so a pale slide and a dark one both work.
3. Lay a grid of 224 x 224 tiles at 0.5 micrometre per pixel (20x, H-optimus-0's training
   magnification) over the slide, keep the grid cells that are mostly tissue, and pick
   --tiles of them at random (fixed seed).
4. Save the tiles, plus a small preview sheet you can open to check they look like tissue.
5. Delete the slide (unless --keep-slides).

Finished slides are skipped, so after a crash or a dropped connection just run it again.
At the end all tiles are combined into data/interim/tcga_tiles/images.npy plus meta.csv
(one row per tile: slide, patient, site), which script 11 reads with --dataset tcga.

Needs: pip install openslide-python openslide-bin
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from prss.download import fetch
from prss.slides.data import ROOT, TCGA_TILES

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", type=int, default=64, help="tiles per slide")
ap.add_argument("--min-tissue", type=float, default=0.6, help="share of a tile that must be tissue")
ap.add_argument("--keep-slides", action="store_true")
ap.add_argument("--slides", default=str(ROOT / "data" / "interim" / "tcga_slides.csv"))
ap.add_argument("--from-dir", help="read slides already on disk (file_name in this folder) instead of downloading")
args = ap.parse_args()

TILE, MPP = 224, 0.5
RAW = ROOT / "data" / "raw" / "tcga_slides"
PER_SLIDE = TCGA_TILES / "per_slide"
PREVIEW = TCGA_TILES / "previews"


def otsu(values):
    """Otsu's threshold: the cut-off that best splits the values into two groups."""
    hist, edges = np.histogram(values, bins=256, range=(0, 256))
    p = hist / hist.sum()
    w = np.cumsum(p)                                # share of pixels below each cut-off
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return edges[int(np.argmax(between))]


def slide_mpp(slide):
    """Micrometres per pixel at full resolution. TCGA slides usually say; if not, use the objective."""
    import openslide
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp)
    mag = slide.properties.get("aperio.AppMag")
    return {"40": 0.25, "20": 0.5}.get(str(mag).split(".")[0])


def tile_slide(path, seed):
    import openslide
    slide = openslide.OpenSlide(str(path))
    mpp = slide_mpp(slide)
    if mpp is None:
        print("  no pixel size recorded: skipping this slide")
        return None
    W, H = slide.dimensions
    side0 = TILE * MPP / mpp                         # tile side in full-resolution pixels

    # tissue mask on a thumbnail about 2,000 pixels wide
    thumb = np.asarray(slide.get_thumbnail((2048, 2048)).convert("HSV"))
    sat, val = thumb[..., 1], thumb[..., 2]
    tissue = (sat > max(otsu(sat), 20)) & (val > 30)  # coloured, and not black (pen marks, edges)
    sx, sy = tissue.shape[1] / W, tissue.shape[0] / H

    cells = []
    for y in np.arange(0, H - side0, side0):
        for x in np.arange(0, W - side0, side0):
            t = tissue[int(y * sy):int((y + side0) * sy) + 1, int(x * sx):int((x + side0) * sx) + 1]
            if t.size and t.mean() >= args.min_tissue:
                cells.append((int(x), int(y)))
    if len(cells) < args.tiles:
        print(f"  only {len(cells)} tissue tiles (wanted {args.tiles}); keeping all of them")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cells), size=min(args.tiles, len(cells)), replace=False) if cells else []

    # read each tile from the pyramid level closest to 0.5 um/px, then resize to exactly 224
    level = slide.get_best_level_for_downsample(MPP / mpp)
    scale = slide.level_downsamples[level]
    size = int(round(side0 / scale))
    tiles = []
    for i in sorted(pick):
        x, y = cells[i]
        im = slide.read_region((x, y), level, (size, size)).convert("RGB")
        tiles.append(np.asarray(im.resize((TILE, TILE), Image.BILINEAR)))
    slide.close()
    return np.stack(tiles) if tiles else None


def preview(tiles, path):
    """An 8-wide sheet of the tiles at half size, to check by eye that they are tissue."""
    n = len(tiles)
    cols, rows = 8, -(-n // 8)
    sheet = Image.new("RGB", (cols * 112, rows * 112), "white")
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t).resize((112, 112)), ((i % cols) * 112, (i // cols) * 112))
    sheet.save(path, quality=85)


slides = pd.read_csv(args.slides, dtype=str)
PER_SLIDE.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
for n, s in enumerate(slides.itertuples(), 1):
    out = PER_SLIDE / f"{s.patient}.npy"
    print(f"[{n}/{len(slides)}] {s.patient} (site {s.site}, {int(s.size) / 1e9:.2f} GB)")
    if out.exists():
        print("  already tiled, skipping")
        continue
    if args.from_dir:
        path = Path(args.from_dir) / s.file_name
    else:
        # check: the finished file must have exactly the size GDC listed
        path = fetch(f"https://api.gdc.cancer.gov/data/{s.file_id}", RAW / s.file_name, label=s.patient,
                     check=lambda p, size=int(s.size): p.stat().st_size == size)
    tiles = tile_slide(path, seed=zlib.crc32(s.patient.encode()))   # seed from the patient: reproducible
    if tiles is not None:
        np.save(out, tiles)
        preview(tiles, PREVIEW / f"{s.site}_{s.patient}.jpg")
        print(f"  saved {len(tiles)} tiles")
    if not args.keep_slides and not args.from_dir:
        path.unlink()

# combine everything tiled so far into one array + one table, in a fixed order
images, meta = [], []
for s in slides.itertuples():
    f = PER_SLIDE / f"{s.patient}.npy"
    if f.exists():
        t = np.load(f)
        images.append(t)
        meta += [dict(slide=s.file_name, patient=s.patient, site=s.site)] * len(t)
if not images:
    raise SystemExit("no tiles yet")
np.save(TCGA_TILES / "images.npy", np.concatenate(images))
meta = pd.DataFrame(meta)
meta.to_csv(TCGA_TILES / "meta.csv", index=False)
print(f"\ncombined {len(meta)} tiles from {meta.patient.nunique()} patients:")
print(meta.groupby("site").agg(patients=("patient", "nunique"), tiles=("patient", "size")).to_string())
print(f"previews to check by eye: {PREVIEW.relative_to(ROOT)}/")
