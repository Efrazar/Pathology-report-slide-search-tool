"""Load PanNuke tiles and tissue labels, after scripts/10_get_pannuke.py has prepared them.

Layout on disk (made by script 10), one folder per official fold:
    data/raw/pannuke/fold1/images.npy   uint8, shape (N, 256, 256, 3), RGB
    data/raw/pannuke/fold1/types.npy    str,   shape (N,), tissue type per tile, e.g. "Breast"

Why uint8: the original PanNuke files store pixels as float64, 8 bytes per value. A fold is
then about 4 GB. Pixels are whole numbers from 0 to 255, so uint8 (1 byte) holds them exactly
and the fold drops to about 0.5 GB.

Why mmap_mode="r": numpy maps the file instead of reading it all into RAM. Only the tiles a
batch needs are read from disk, so memory use stays small whatever the fold size.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PANNUKE = ROOT / "data" / "raw" / "pannuke"
TCGA_TILES = ROOT / "data" / "interim" / "tcga_tiles"
FOLDS = (1, 2, 3)


def load_fold(k, root=PANNUKE):
    """(images, types) for fold k. images is a read-only memory map; types is a normal array."""
    d = Path(root) / f"fold{k}"
    return np.load(d / "images.npy", mmap_mode="r"), np.load(d / "types.npy", allow_pickle=True).astype(str)


def load_tcga_tiles(root=TCGA_TILES):
    """(images, sites) for the hospital study, as combined by script 21.

    images.npy: uint8 (N, 224, 224, 3). meta.csv: one row per tile (slide, patient, site);
    the row order matches images.npy.
    """
    import pandas as pd
    meta = pd.read_csv(Path(root) / "meta.csv", dtype=str)   # str: keep codes like "3C" as text
    return np.load(Path(root) / "images.npy", mmap_mode="r"), meta["site"].astype(str).to_numpy()
