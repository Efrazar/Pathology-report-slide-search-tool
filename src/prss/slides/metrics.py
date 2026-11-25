"""Scores for similar-tile search. Kept apart from the scripts so the tests can check them.

precision at k, as in the hackathon: for each query, look at its k nearest neighbours and
count the share with the same label. Average that share over all queries.
"""
import numpy as np


def l2_normalize(x):
    """Scale every row to length 1. After this, a dot product IS the cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def nearest(query, gallery, k, exclude_self=False):
    """Indices of the k most similar gallery rows for every query row (cosine similarity).

    exclude_self: set True when query and gallery are the same set, so a tile does not find
    itself as its own best match (that would give precision at 1 of 100% for free).
    """
    sims = l2_normalize(query) @ l2_normalize(gallery).T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    # argpartition finds the top k without sorting everything; then we sort just those k.
    top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def precision_at_k(query_labels, gallery_labels, neighbours, ks=(1, 5, 10)):
    """{k: mean share of the first k neighbours whose label matches the query's label}."""
    hits = np.asarray(gallery_labels)[neighbours] == np.asarray(query_labels)[:, None]
    return {k: float(hits[:, :k].mean()) for k in ks}


def chance_precision(query_labels, gallery_labels):
    """What precision at k a RANDOM ranking would score, on average.

    A random neighbour has label c with probability (share of c in the gallery). So the
    expected precision is the average, over queries, of that share for the query's own label.
    Why report it: PanNuke is unbalanced (breast alone has far more tiles than most types), so a
    precision that looks high may be closer to chance than it seems.
    """
    labels, counts = np.unique(np.asarray(gallery_labels), return_counts=True)
    share = dict(zip(labels, counts / counts.sum()))
    return float(np.mean([share.get(q, 0.0) for q in query_labels]))
