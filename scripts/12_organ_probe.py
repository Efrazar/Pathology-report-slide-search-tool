"""Step 12 (Track 2): how well do the embeddings tell the tissue types apart? A "linear probe".

We freeze the foundation model and train only a logistic regression on its vectors. If a
straight-line classifier can separate the organs, the model already "knows" them; nothing was
learned from our labels except where to draw the lines. This is the standard first test of a
foundation model, and it trains in seconds on the CPU.

How the folds are used: PanNuke comes in 3 official folds. Each round trains on one fold,
picks the regularisation strength C on a second, and is scored on the third. We run all
3 rotations and report the mean, so the result does not depend on one lucky split. A test
fold is never seen while training or choosing C.

Metrics:
- accuracy: share of tiles labelled correctly (the hackathon metric for organ labels)
- balanced accuracy: accuracy per tissue type, averaged, so the big breast class cannot hide
  a small class the model always gets wrong
- macro F1: F1 per tissue type, averaged

Output: printed table, results/organ_probe.json (with a confusion matrix per rotation)
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prss.slides.data import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hoptimus0", help="embedding file prefix; 'smoke' after a smoke test")
args = ap.parse_args()

EMB = ROOT / "data" / "interim" / "embeddings"
X = {k: np.load(EMB / f"{args.model}_fold{k}.npy") for k in (1, 2, 3)}
y = {k: np.load(EMB / f"{args.model}_fold{k}_types.npy") for k in (1, 2, 3)}
classes = sorted(set(np.concatenate(list(y.values()))))
# C = how little we regularise (larger C = the classifier may fit the training fold more closely).
# v2: 10 and 100 added after the first run picked C=1.0, the top of the old grid, every time.
# A winner at the edge of the grid means the best value may lie outside it.
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def fit(train, C):
    # StandardScaler: puts every one of the 1,536 numbers on the same scale, which helps the
    # solver and makes one C value mean the same thing for every feature.
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    return clf.fit(X[train], y[train])


rounds = []
for train, tune, test in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
    tuning = {C: balanced_accuracy_score(y[tune], fit(train, C).predict(X[tune])) for C in C_GRID}
    best_C = max(tuning, key=tuning.get)
    if best_C in (C_GRID[0], C_GRID[-1]):
        print(f"  note: C={best_C} is at the edge of the grid; the best value may lie beyond it")
    pred = fit(train, best_C).predict(X[test])
    rounds.append(dict(
        train=train, tune=tune, test=test, C=best_C, tune_balanced_accuracy=tuning,
        accuracy=accuracy_score(y[test], pred),
        balanced_accuracy=balanced_accuracy_score(y[test], pred),
        macro_f1=f1_score(y[test], pred, average="macro", labels=classes, zero_division=0),
        per_type_f1=dict(zip(classes, f1_score(y[test], pred, average=None, labels=classes,
                                               zero_division=0).round(3).tolist())),
        confusion=confusion_matrix(y[test], pred, labels=classes).tolist(),
    ))
    r = rounds[-1]
    print(f"train {train}, tune {tune}, test {test}: C={best_C:<6} accuracy {r['accuracy']:.1%}  "
          f"balanced {r['balanced_accuracy']:.1%}  macro F1 {r['macro_f1']:.3f}")

mean = {m: float(np.mean([r[m] for r in rounds])) for m in ("accuracy", "balanced_accuracy", "macro_f1")}
print(f"\nmean of 3 rotations: accuracy {mean['accuracy']:.1%}  balanced {mean['balanced_accuracy']:.1%}  "
      f"macro F1 {mean['macro_f1']:.3f}")
print("(hackathon, organ labels on the organizers' slides: accuracy 89.6%; different data, so a guide only)")

print("\nF1 per tissue type, mean of 3 rotations, worst first")
per_type = {c: float(np.mean([r["per_type_f1"][c] for r in rounds])) for c in classes}
for c, f in sorted(per_type.items(), key=lambda kv: kv[1]):
    print(f"  {c:<16}{f:.3f}")

if args.model != "smoke":
    out = ROOT / "results" / "organ_probe.json"
    json.dump(dict(model=args.model, classes=classes, mean=mean, per_type_f1=per_type, rounds=rounds),
              open(out, "w"), indent=1)
    print(f"\nsaved {out.relative_to(ROOT)}")
