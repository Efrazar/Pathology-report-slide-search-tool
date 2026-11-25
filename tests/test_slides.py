"""Tests for the similar-tile scores. Run with:  pytest -q"""
import numpy as np

from prss.slides.metrics import chance_precision, nearest, precision_at_k


def test_nearest_finds_most_similar_first():
    gallery = np.array([[1, 0], [0, 1], [1, 0.1]])
    assert nearest(np.array([[1, 0.02]]), gallery, k=2).tolist() == [[0, 2]]


def test_nearest_can_skip_itself():
    x = np.array([[1, 0], [0.9, 0.1], [0, 1]])
    assert nearest(x, x, k=1, exclude_self=True)[:, 0].tolist() == [1, 0, 1]


def test_precision_at_k():
    neighbours = np.array([[0, 1], [2, 0]])
    p = precision_at_k(["a", "b"], ["a", "b", "b"], neighbours, ks=(1, 2))
    assert p == {1: 1.0, 2: 0.5}   # query a sees a, b; query b sees b, a: first always right, half of two


def test_chance_precision_counts_class_share():
    # gallery is 3/4 "a": a random neighbour of an "a" query matches 75% of the time, of a "b" 25%
    assert chance_precision(["a", "b"], ["a", "a", "a", "b"]) == 0.5
