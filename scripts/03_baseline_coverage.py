"""Step 3: run the parser, almost unchanged from the hackathon, and measure how much it finds.

No scoring yet. This answers a simpler question first: on real reports, how often does the
parser find each field at all? On the synthetic hackathon reports the answer was 100%. The gap
here is the to-do list for Track 1.

Output: printed table, and results/baseline_coverage.json
"""
import json
from collections import Counter
from pathlib import Path
from prss.reports.parse import FIELD_NAMES, parse
from prss.reports.fields import tnm

ROOT = Path(__file__).resolve().parents[1]
gdc = {json.loads(l)["patient"]: json.loads(l) for l in open(ROOT / "data/interim/gdc_clinical.jsonl")}
reports = [json.loads(l) for l in open(ROOT / "data/interim/reports.jsonl")]
reports = [r for r in reports if r["patient"] in gdc]   # keep only the projects we asked GDC for

found, by_project, tnm_hits = Counter(), Counter(), Counter()
empty = 0
for r in reports:
    p = gdc[r["patient"]]["project"]
    by_project[p] += 1
    fields = parse(r["text"])["fields"]
    empty += not fields
    found.update(fields.keys())   # .keys(): Counter.update(dict) would ADD the values, not count keys
    if tnm(r["text"])[0]:
        tnm_hits[p] += 1

n = len(reports)
print(f"{n} reports in {dict(by_project)}; {empty} ({empty / n:.0%}) gave no labelled fields at all\n")
print(f"{'field':28s} found in")
for k in FIELD_NAMES:
    print(f"{k:28s} {found[k] / n:6.1%}")
print("\nT category found anywhere in text, by project:")
for p, c in by_project.items():
    print(f"  {p}: {tnm_hits[p] / c:.0%}")

(ROOT / "results").mkdir(exist_ok=True)
json.dump({"n": n, "empty": empty, "field_coverage": {k: found[k] / n for k in FIELD_NAMES},
           "t_found_by_project": {p: tnm_hits[p] / c for p, c in by_project.items()}},
          open(ROOT / "results/baseline_coverage.json", "w"), indent=1)
