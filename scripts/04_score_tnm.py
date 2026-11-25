"""Step 4: first real accuracy numbers. Does the T and N we extract match what GDC recorded?

Why T and N first: they are the only report facts GDC has for all four cancer types, and most
of our criteria depend on them. If T and N are wrong, every criterion built on them is wrong.

Three numbers per project, for T and N separately:
- coverage: share of reports where we extracted a value at all
- accuracy when extracted: of those, how often the main category matches GDC (T2a vs T2b
  both count as "T2"; staging editions differ in sub-categories, so main category is fairer)
- end-to-end: matches / all reports that have a GDC label. This is the honest headline,
  because a value we fail to find is as useless to a trial team as a wrong one.

Disagreements go to data/interim/tnm_disagreements.jsonl (git-ignored, it contains report text).
Read 20 of them: some will be our bugs, some will be GDC label errors. Knowing the split
matters more than the score.
"""
import json, re, subprocess
from collections import defaultdict
from pathlib import Path
from prss.reports.criteria import extract_tnm
from prss.reports.parse import parse

ROOT = Path(__file__).resolve().parents[1]
gdc = {json.loads(l)["patient"]: json.loads(l) for l in open(ROOT / "data/interim/gdc_clinical.jsonl")}
reports = [json.loads(l) for l in open(ROOT / "data/interim/reports.jsonl")]


def main_cat(v, letter):
    """'T2b' -> 'T2', 'N0 (i+)' -> 'N0', 'TX' -> 'TX'. Anything else (None, 'Not Reported', '--') -> None."""
    m = re.match(letter + r"(is|X|\d)", v or "")
    return letter + m.group(1) if m else None


stats = defaultdict(lambda: defaultdict(int))
bad = []
for r in reports:
    g = gdc.get(r["patient"])
    if not g:
        continue
    p = parse(r["text"])
    t_val, n_val, t_pos = extract_tnm(p)
    ours = {"T": t_val, "N": n_val}
    for letter in "TN":
        truth = main_cat(g["p" + letter], letter)
        if truth is None:
            continue                       # no usable GDC label: cannot score this one
        got = main_cat(ours[letter], letter)
        s = stats[(g["project"], letter)]
        s["labelled"] += 1
        s["extracted"] += got is not None
        s["correct"] += got == truth
        if got is not None and got != truth:
            # Show the words the T was actually read from (v1 searched for "pT" and could land
            # inside unrelated words such as "HercepTest", or find nothing for a bare "T2").
            i = t_pos if t_pos is not None else -1
            bad.append({"patient": r["patient"], "project": g["project"], "field": letter, "ours": got,
                        "gdc": truth, "context": r["text"][max(0, i - 80): i + 100] if i >= 0 else ""})

print(f"{'project':10s} {'':2s} {'labelled':>8s} {'coverage':>9s} {'acc|extracted':>14s} {'end-to-end':>11s}")
out = {}
for (proj, letter), s in sorted(stats.items()):
    cov = s["extracted"] / s["labelled"]
    acc = s["correct"] / s["extracted"] if s["extracted"] else 0.0
    e2e = s["correct"] / s["labelled"]
    print(f"{proj:10s} {letter:2s} {s['labelled']:8d} {cov:9.1%} {acc:14.1%} {e2e:11.1%}")
    out[f"{proj}_{letter}"] = {"labelled": s["labelled"], "coverage": cov, "accuracy_when_extracted": acc, "end_to_end": e2e}

# ---- Regression check: compare with the scores committed in git (not the file on disk, which
# the last run may already have overwritten with worse numbers). Added after a run where N
# coverage silently fell 9-15 points because two changes were combined in the wrong order.
TOLERANCE = 0.01   # ignore changes under 1 percentage point
try:
    committed = json.loads(subprocess.run(["git", "show", "HEAD:results/tnm_scores.json"], cwd=ROOT,
                                          capture_output=True, text=True, check=True).stdout)
except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
    committed = None
if committed:
    drops = [(key, metric, committed[key][metric], now[metric])
             for key, now in out.items() if key in committed
             for metric in ("coverage", "accuracy_when_extracted", "end_to_end")
             if now[metric] < committed[key][metric] - TOLERANCE]
    if drops:
        print("\nWARNING: worse than the committed results (git HEAD):")
        for key, metric, before, now in drops:
            print(f"  {key:12s} {metric:24s} {before:6.1%} -> {now:6.1%}")
        print("Check the branch you are on and what changed before committing these numbers.")
    else:
        print("\nRegression check: nothing worse than the committed results.")

(ROOT / "data/interim/tnm_disagreements.jsonl").write_text("\n".join(json.dumps(b) for b in bad))
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results/tnm_scores.json", "w"), indent=1)
print(f"\n{len(bad)} disagreements written to data/interim/tnm_disagreements.jsonl - read some before fixing anything")
