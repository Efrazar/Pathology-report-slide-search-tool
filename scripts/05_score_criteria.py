"""Step 5: the headline test. Do the trial criteria pick the same patients from the report text
as they would from the curated GDC labels?

For each criterion we build two patient lists:
- predicted: the criterion run on what we EXTRACTED from each report (criteria.row)
- "truth":   the same criterion run on the GDC labels for that patient (criteria.gdc_row)
and score precision, recall and F1, exactly as in the hackathon.

Which patients count: only those whose GDC record has every label the criterion needs
(T for T criteria, N for node criteria). A patient GDC cannot judge is left out, not counted
as a "no".

Remember what "truth" means here: GDC labels, which we have already seen disagree with the
report text in some cases. A low score can be our miss OR a label problem. Read the
disagreements (data/interim/criteria_disagreements.jsonl) before believing any number.

Output: printed table, results/criteria_scores.json
"""
import json, subprocess
from pathlib import Path
from prss.reports.parse import parse
from prss.reports.criteria import CRITERIA, gdc_row, row

ROOT = Path(__file__).resolve().parents[1]
gdc = {json.loads(l)["patient"]: json.loads(l) for l in open(ROOT / "data/interim/gdc_clinical.jsonl")}
reports = [json.loads(l) for l in open(ROOT / "data/interim/reports.jsonl")]

# For each criterion: its cancer type, and which GDC labels it needs before GDC can say yes or no.
NEEDS = {
    "breast_node_positive_over_20mm": ("TCGA-BRCA", ("pT", "pN")),
    "colon_stage_iii": ("TCGA-COAD", ("pN",)),
    "colon_pt4": ("TCGA-COAD", ("pT",)),
    "lung_adeno_node_positive": ("TCGA-LUAD", ("pN",)),
    "lung_squamous_pt3_or_higher": ("TCGA-LUSC", ("pT",)),
}

rows = []
for r in reports:
    g = gdc.get(r["patient"])
    if g:
        rows.append((r["patient"], row(g["project"], parse(r["text"])), gdc_row(g), r["text"]))

out, bad = {}, []
print(f"{'criterion':32s} {'judged':>6s} {'GDC yes':>7s} {'ours yes':>8s} {'precision':>9s} {'recall':>7s} {'F1':>6s}")
for name, fn in CRITERIA.items():
    tp = fp = fn_ = judged = 0
    for pid, ours, truth, text in rows:
        project, needed = NEEDS[name]
        if truth["project"] != project or any(truth[k] is None for k in needed):
            continue           # other cancer types, or GDC cannot judge this patient
        judged += 1
        e_ours, e_truth = fn(ours), fn(truth)
        if e_ours and e_truth:
            tp += 1
        elif e_ours or e_truth:
            fp += bool(e_ours); fn_ += bool(e_truth)
            bad.append({"criterion": name, "patient": pid, "ours": bool(e_ours), "gdc": bool(e_truth),
                        "our_values": {k: ours[k] for k in ("pT", "pN", "nodes_pos", "size_mm")},
                        "gdc_values": {k: truth[k] for k in ("pT", "pN")}, "evidence": e_ours or {}})
    if tp + fp + fn_ == 0:          # nobody qualifies on either side: F1 is undefined, not 0
        out[name] = {"judged": judged, "gdc_yes": 0, "ours_yes": 0, "precision": None, "recall": None, "f1": None}
        print(f"{name:32s} {judged:6d} {0:7d} {0:8d}   (no patient qualifies on either side)")
        continue
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    out[name] = {"judged": judged, "gdc_yes": tp + fn_, "ours_yes": tp + fp, "precision": p, "recall": rc, "f1": f1}
    print(f"{name:32s} {judged:6d} {tp + fn_:7d} {tp + fp:8d} {p:9.1%} {rc:7.1%} {f1:6.2f}")

scored = [v["f1"] for v in out.values() if v["f1"] is not None]
mean_f1 = sum(scored) / len(scored) if scored else 0.0
print(f"\nmean F1 over {len(scored)} criteria: {mean_f1:.2f}   (hackathon, synthetic reports: 1.00)")

# Same regression check as scripts 03 and 04.
try:
    committed = json.loads(subprocess.run(["git", "show", "HEAD:results/criteria_scores.json"], cwd=ROOT,
                                          capture_output=True, text=True, check=True).stdout)["criteria"]
    drops = [(k, committed[k]["f1"], v["f1"]) for k, v in out.items() if k in committed and v["f1"] is not None and committed[k]["f1"] is not None
             and v["f1"] < committed[k]["f1"] - 0.01]
    for k, before, now in drops:
        print(f"WARNING: {k} F1 {before:.2f} -> {now:.2f} (worse than committed)")
    if not drops:
        print("Regression check: nothing worse than the committed results.")
except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, KeyError):
    print("(no committed criteria_scores.json yet: this run becomes the first baseline once you commit it)")

(ROOT / "data/interim/criteria_disagreements.jsonl").write_text("\n".join(json.dumps(b) for b in bad))
(ROOT / "results").mkdir(exist_ok=True)
json.dump({"mean_f1": mean_f1, "criteria": out}, open(ROOT / "results/criteria_scores.json", "w"), indent=1)
print(f"{len(bad)} disagreements written to data/interim/criteria_disagreements.jsonl")
