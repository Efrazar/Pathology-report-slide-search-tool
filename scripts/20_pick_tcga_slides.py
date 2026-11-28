"""Step 20 (Track 2b, hospital study): choose which TCGA breast cancer slides to download.

The question: does the foundation model also encode WHICH HOSPITAL a slide came from? Every
TCGA barcode names the hospital that sent the tissue (the "tissue source site", TSS):
in TCGA-A2-A0T2, "A2" is the site. Keeping one cancer type (breast) keeps the tissue roughly
the same, so a hospital signal is lab style (staining, scanning), not organ.

What this does:
1. Asks the GDC API for every open-access breast cancer diagnostic slide (FFPE, the "DX" ones).
2. Keeps one slide per patient (the first DX slide), and counts patients per site.
3. Takes the --sites sites with the most patients and, from each, --per-site patients at random
   (fixed seed, so the choice is the same every run).

The pilot and the full run use the SAME shuffled order per site: the pilot's 3 slides per
site are the first 3 of the full run's 12, so nothing downloaded for the pilot is wasted.

Output: data/interim/tcga_slides.csv (file_id, file_name, size, patient, site) and the total
download size.

Pilot:  python scripts/20_pick_tcga_slides.py --per-site 3
Full:   python scripts/20_pick_tcga_slides.py
"""
import argparse
import random

import pandas as pd
import requests

from prss.slides.data import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="TCGA-BRCA")
ap.add_argument("--sites", type=int, default=6)
ap.add_argument("--per-site", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

query = {
    "filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [args.project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]},
    "fields": "file_id,file_name,file_size,cases.submitter_id",
    "format": "JSON",
    "size": 10000,
}
resp = requests.post("https://api.gdc.cancer.gov/files", json=query, timeout=120)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]

rows = []
for h in hits:
    patient = h["cases"][0]["submitter_id"]                 # e.g. TCGA-A2-A0T2
    rows.append(dict(file_id=h["file_id"], file_name=h["file_name"], size=h["file_size"],
                     patient=patient, site=patient.split("-")[1]))
slides = pd.DataFrame(rows).sort_values("file_name")         # DX1 sorts before DX2
slides = slides.drop_duplicates("patient")                    # one slide per patient
print(f"{len(hits)} diagnostic slides, {len(slides)} patients, {slides.site.nunique()} sites")

per_site = slides.site.value_counts()
print("\npatients per site, largest 12:")
print(per_site.head(12).to_string())
chosen_sites = [s for s in per_site.index if per_site[s] >= args.per_site][:args.sites]
if len(chosen_sites) < args.sites:
    raise SystemExit(f"only {len(chosen_sites)} sites have {args.per_site} patients or more")

picked = []
for s in chosen_sites:
    pts = sorted(slides[slides.site == s].patient)
    random.Random(f"{args.seed}-{s}").shuffle(pts)            # same order for pilot and full run
    picked.append(slides[slides.patient.isin(pts[:args.per_site])])
picked = pd.concat(picked)

out = ROOT / "data" / "interim" / "tcga_slides.csv"
out.parent.mkdir(parents=True, exist_ok=True)
picked.to_csv(out, index=False)
print(f"\npicked {len(picked)} slides from sites {', '.join(chosen_sites)}")
print(f"download size: {picked['size'].sum() / 1e9:.1f} GB "
      f"(median {picked['size'].median() / 1e9:.2f} GB, largest {picked['size'].max() / 1e9:.2f} GB per slide)")
print(f"saved {out.relative_to(ROOT)}")
