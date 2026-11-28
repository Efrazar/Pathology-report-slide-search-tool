"""Step 22 (Track 2b): does H-optimus-0 know which hospital a breast cancer tile came from?

Two tests on the embeddings from script 11 --dataset tcga:

1. Site probe. A logistic regression learns to predict the hospital (tissue source site) from
   a tile's embedding. Every patient's tiles stay on ONE side of each split (grouped
   cross-validation). Without that, the probe could learn "this exact tumour" and look as if
   it had learned the hospital. Scored per tile, and per patient (average the tile
   predictions, take the most likely site).
   Chance is about 1 in (number of sites). To check that number honestly we also re-run the
   probe with the site labels shuffled between patients: whatever that scores is what
   "no real signal" looks like with this much data.

2. Search check. For each tile, find its 10 most similar tiles from OTHER patients, and count
   how many come from the same hospital. If similar-slide search were blind to the lab, that
   share would match the hospital's share of the tiles (the chance line). Well above it means
   the search partly finds "same lab" instead of "similar tissue".
   We also report how often a tile's nearest neighbours are the same patient when that is
   allowed: a sense of how strongly one slide's look dominates.

What it can and cannot say: a high score shows the model can tell the sites apart. It does
not say why: staining, scanner, slide age and patient mix all differ by site.

Output: printed tables, results/site_bias.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prss.slides.data import ROOT, TCGA_TILES
from prss.slides.metrics import l2_normalize

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hoptimus0")
ap.add_argument("--k", type=int, default=10, help="neighbours for the search check")
ap.add_argument("--shuffles", type=int, default=5, help="label-shuffled re-runs for the chance check")
args = ap.parse_args()

X = np.load(ROOT / "data" / "interim" / "embeddings" / f"{args.model}_tcga.npy")
meta = pd.read_csv(TCGA_TILES / "meta.csv", dtype=str)
assert len(X) == len(meta), "embeddings and meta.csv disagree: re-run script 11 --dataset tcga"
site, patient = meta.site.to_numpy(), meta.patient.to_numpy()
sites = sorted(set(site))
pts_per_site = meta.groupby("site").patient.nunique()
n_splits = int(min(4, pts_per_site.min()))
print(f"{len(X)} tiles, {meta.patient.nunique()} patients, {len(sites)} sites; {n_splits}-fold grouped CV")


def probe(labels):
    """Out-of-fold site probabilities for every tile, never trained on the tile's own patient."""
    proba = np.zeros((len(X), len(sites)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in cv.split(X, labels, groups=patient):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        clf.fit(X[tr], labels[tr])
        proba[np.ix_(te, [sites.index(c) for c in clf.classes_])] = clf.predict_proba(X[te])
    return proba


def scores(labels, proba):
    tile_pred = np.array(sites)[proba.argmax(1)]
    df = pd.DataFrame(proba).assign(patient=patient, label=labels)
    by_pt = df.groupby("patient")
    pt_pred = np.array(sites)[by_pt[list(range(len(sites)))].mean().to_numpy().argmax(1)]
    pt_true = by_pt.label.first().to_numpy()
    return dict(tile_accuracy=accuracy_score(labels, tile_pred),
                tile_balanced_accuracy=balanced_accuracy_score(labels, tile_pred),
                patient_accuracy=accuracy_score(pt_true, pt_pred)), tile_pred


real, tile_pred = scores(site, probe(site))

# chance check: shuffle which site each PATIENT belongs to (tiles keep their patient)
rng = np.random.default_rng(0)
pt_site = meta.groupby("patient").site.first()
shuffled = []
for _ in range(args.shuffles):
    fake = dict(zip(pt_site.index, rng.permutation(pt_site.to_numpy())))
    fake_site = np.array([fake[p] for p in patient])
    shuffled.append(scores(fake_site, probe(fake_site))[0])
chance = {m: float(np.mean([s[m] for s in shuffled])) for m in real}

print(f"\nsite probe              {'real':>8}{'shuffled':>10}{'1/sites':>9}")
for m in real:
    print(f"  {m:<22}{real[m]:>8.1%}{chance[m]:>10.1%}{1 / len(sites):>9.1%}")

# search check: neighbours from other patients only
Z = l2_normalize(X)
sims = Z @ Z.T
same_patient = patient[:, None] == patient[None, :]
np.fill_diagonal(sims, -np.inf)
own = np.argsort(-sims, axis=1)[:, :args.k]
share_same_patient = float(same_patient[np.arange(len(X))[:, None], own].mean())
sims[same_patient] = -np.inf
nb = np.argsort(-sims, axis=1)[:, :args.k]
same_site = site[nb] == site[:, None]
other = ~same_patient
expected = np.array([(other[i] & (site == site[i])).sum() / other[i].sum() for i in range(len(X))])

print(f"\nsearch check, {args.k} nearest tiles from other patients")
print(f"  {'site':<8}{'same site':>10}{'chance':>9}")
per_site = {}
for c in sites:
    m = site == c
    per_site[c] = dict(same_site=float(same_site[m].mean()), chance=float(expected[m].mean()))
    print(f"  {c:<8}{per_site[c]['same_site']:>10.1%}{per_site[c]['chance']:>9.1%}")
print(f"  {'all':<8}{same_site.mean():>10.1%}{expected.mean():>9.1%}")
print(f"\nwhen the same patient is allowed, {share_same_patient:.1%} of the {args.k} nearest tiles are the same patient")

out = ROOT / "results" / "site_bias.json"
json.dump(dict(model=args.model, n_tiles=len(X), n_patients=int(meta.patient.nunique()), sites=sites,
               cv_splits=n_splits, probe=real, probe_shuffled_labels=chance,
               confusion=confusion_matrix(site, tile_pred, labels=sites).tolist(),
               search=dict(k=args.k, same_site=float(same_site.mean()), chance=float(expected.mean()),
                           per_site=per_site, same_patient_when_allowed=share_same_patient)),
          open(out, "w"), indent=1)
print(f"\nsaved {out.relative_to(ROOT)}")
