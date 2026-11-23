"""Step 1: load the TCGA-Reports text into one tidy file.

Before running: download the dataset by hand from Mendeley Data
(https://data.mendeley.com/datasets/hyg5xkznpx/1, "Download All"), unzip it, and put the
CSV in data/raw/. It is CC BY 4.0, so it is free to use with credit, but we still keep it
out of git: the repo holds code, and anyone can rebuild data/ by following these steps.

Output: data/interim/reports.jsonl, one {"patient": "TCGA-XX-XXXX", "text": "..."} per line.

Why the column guessing below: I could not open the file from my side to check its exact
column names, so the script looks for them and tells you what it found. Check the printout.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "interim"

csv.field_size_limit(sys.maxsize)   # reports are long; the default cell limit is too small
files = sorted(RAW.glob("*.csv"))
if not files:
    sys.exit(f"No CSV in {RAW}. Download TCGA-Reports from Mendeley first (see this file's docstring).")
src = files[0]
with open(src, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    cols = reader.fieldnames
    id_col = next(c for c in cols if "file" in c.lower() or "patient" in c.lower() or "id" in c.lower())
    text_col = next(c for c in cols if "text" in c.lower())
    print(f"{src.name}: columns {cols} -> id='{id_col}', text='{text_col}'")
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "reports.jsonl", "w") as out:
        for r in reader:
            # A TCGA patient barcode is the first 12 characters, e.g. TCGA-A1-A0SB.
            out.write(json.dumps({"patient": r[id_col][:12], "text": r[text_col]}) + "\n")
            n += 1
print(f"wrote {n} reports to {OUT / 'reports.jsonl'}")
