"""Step 2: download the public labels we score against, from the NCI Genomic Data Commons (GDC).

For each patient: the TCGA project (cancer type) and the pathologic T, N and stage that GDC
curators recorded. These are our "answer key". They were abstracted from the same reports by
people, so a few will be wrong. Expect to find some of those.

Output: data/interim/gdc_clinical.jsonl

Learning note: this is a plain REST API. We send one POST request with a filter (which projects)
and a list of fields (which columns), and get JSON back. No API key is needed for open data.
Status: ran successfully on 24 Nov 2026 (2,648 cases for the four projects below).
"""
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "gdc_clinical.jsonl"
PROJECTS = ["TCGA-BRCA", "TCGA-COAD", "TCGA-LUAD", "TCGA-LUSC"]
FIELDS = [
    "submitter_id", "project.project_id",
    "diagnoses.primary_diagnosis", "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_stage",
]

query = {
    "filters": {"op": "in", "content": {"field": "project.project_id", "value": PROJECTS}},
    "fields": ",".join(FIELDS),
    "format": "JSON",
    "size": 10000,
}
resp = requests.post("https://api.gdc.cancer.gov/cases", json=query, timeout=120)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as fh:
    for h in hits:
        d = (h.get("diagnoses") or [{}])[0]   # TCGA cases usually have one primary diagnosis
        fh.write(json.dumps({
            "patient": h["submitter_id"], "project": h["project"]["project_id"],
            "diagnosis": d.get("primary_diagnosis"), "pT": d.get("ajcc_pathologic_t"),
            "pN": d.get("ajcc_pathologic_n"), "stage": d.get("ajcc_pathologic_stage"),
        }) + "\n")
print(f"wrote {len(hits)} cases to {OUT}")
