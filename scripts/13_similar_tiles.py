"""Step 13 (Track 2): similar-tile search. For a query tile, are its nearest neighbours the same tissue?

No training at all: we compare embeddings directly. Two vectors that point the same way
(high cosine similarity) mean the model sees the tiles as alike. This is the public
version of the hackathon's "similar slides" task, scored the same way: precision at 1, 5, 10.

Query and gallery come from DIFFERENT folds. Why: neighbouring tiles cut from the same
slide look almost identical. If they could match each other, the search would look great
without knowing anything about organs. Separate folds make that much less likely. We also
print the same-fold score, so you can see how much that shortcut is worth.

Chance line: a random ranking scores (on average) the share of the query's own tissue in the
gallery. PanNuke is unbalanced, so this is well above 1/19. Read every precision against it.

Ceiling ("best@10"): a perfect search can only find as many same-type tiles as the gallery
holds. With 6 uterus tiles in a gallery, a uterus query scores at most 0.6 at k=10. Read
precision at 10 against this ceiling as well as against chance.

Output: printed table, results/similar_tiles.json
"""
import argparse
import json

import numpy as np

from prss.slides.data import ROOT
from prss.slides.metrics import best_precision, chance_precision, nearest, precision_at_k

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hoptimus0", help="embedding file prefix; 'smoke' after a smoke test")
args = ap.parse_args()

EMB = ROOT / "data" / "interim" / "embeddings"
X = {k: np.load(EMB / f"{args.model}_fold{k}.npy") for k in (1, 2, 3)}
y = {k: np.load(EMB / f"{args.model}_fold{k}_types.npy") for k in (1, 2, 3)}
KS = (1, 5, 10)

rows = []
for q, g in [(3, 1), (1, 2), (2, 3)]:    # query fold, gallery fold
    p = precision_at_k(y[q], y[g], nearest(X[q], X[g], k=max(KS)), KS)
    best = best_precision(y[q], y[g], KS)
    rows.append(dict(query=q, gallery=g, chance=chance_precision(y[q], y[g]),
                     **{f"p@{k}": p[k] for k in KS}, **{f"best@{k}": best[k] for k in KS}))

same = []
for k in (1, 2, 3):
    p = precision_at_k(y[k], y[k], nearest(X[k], X[k], k=max(KS), exclude_self=True), KS)
    same.append({f"p@{kk}": p[kk] for kk in KS})

print(f"{'query -> gallery':<18}{'p@1':>7}{'p@5':>7}{'p@10':>7}{'chance':>8}{'best@10':>9}")
for r in rows:
    print(f"fold {r['query']} -> fold {r['gallery']:<6}" + "".join(f"{r[f'p@{k}']:>7.3f}" for k in KS)
          + f"{r['chance']:>8.3f}{r['best@10']:>9.3f}")
mean = {m: float(np.mean([r[m] for r in rows])) for m in [f"p@{k}" for k in KS] + ["chance"] + [f"best@{k}" for k in KS]}
print(f"{'mean, across folds':<18}" + "".join(f"{mean[f'p@{k}']:>7.3f}" for k in KS) + f"{mean['chance']:>8.3f}"
      + f"{mean['best@10']:>9.3f}")
same_mean = {m: float(np.mean([s[m] for s in same])) for m in same[0]}
print(f"{'same fold (leaky)':<18}" + "".join(f"{same_mean[f'p@{k}']:>7.3f}" for k in KS))
print("(hackathon, similar slides on the organizers' data: 0.881 / 0.893 / 0.887; different data, a guide only)")

if args.model != "smoke":
    out = ROOT / "results" / "similar_tiles.json"
    json.dump(dict(model=args.model, across_folds=rows, mean=mean, same_fold_mean=same_mean), open(out, "w"), indent=1)
    print(f"\nsaved {out.relative_to(ROOT)}")
