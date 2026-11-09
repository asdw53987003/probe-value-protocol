import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from probe_value.inference import (auc, average_precision, cluster_weights, placement_values, summarize, verdict)
from probe_value.threshold import curve, fit_logistic_ramp, sufficiency_threshold


def test_placement_mean_equals_auc_with_ties():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(30, int), np.zeros(120, int)]
    s = np.round(rng.normal(y * 0.8, 1.0), 1)          # rounding creates ties
    psi = placement_values(s, y)
    assert abs(np.nanmean(psi[y == 1]) - roc_auc_score(y, s)) < 1e-12
    assert abs(auc(s, y) - roc_auc_score(y, s)) < 1e-12
    W = np.ones((2, len(y)))
    assert np.allclose(auc(s, y, weights=W), roc_auc_score(y, s))


def test_placement_within_coverage_group():
    y = np.array([1, 0, 0, 1, 0, 0])
    c = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.5, 0.4, 0.6, 0.9, 0.1, 0.95])
    psi = placement_values(s, y, c)
    assert psi[0] == 0.5 and psi[3] == 0.5


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.random(200) < 0.2
    s = rng.normal(y * 1.0, 1.0)
    assert abs(average_precision(s, y) - average_precision_score(y, s)) < 1e-12


def test_average_precision_tied_scores_do_not_depend_on_order():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1])
    ap = average_precision(s, y)
    assert abs(ap - average_precision_score(y, s)) < 1e-12
    o = np.array([1, 0, 4, 3, 2, 6, 5, 7])                       # same scores, ties in another order
    assert abs(average_precision(s[o], y[o]) - ap) < 1e-12


def test_weighted_average_precision_rows():
    rng = np.random.default_rng(2)
    y = rng.random(60) < 0.3
    s = np.round(rng.normal(y * 1.0, 1.0), 1)                    # rounding creates ties
    W = rng.poisson(1.0, (4, 60)).astype(float)
    ap = average_precision(s, y, W)
    for r in range(4):
        k = W[r] > 0
        assert abs(ap[r] - average_precision_score(y[k], s[k], sample_weight=W[r][k])) < 1e-12
    assert np.isnan(average_precision(s, y, np.where(y, 0.0, 1.0)[None, :])[0])


def test_cluster_weights_shared_within_cluster():
    cl = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
    W = cluster_weights(cl, 50, 1)
    assert W.shape == (50, 6)
    assert np.all(W[:, 0] == W[:, 1]) and np.all(W[:, 3] == W[:, 5])
    assert np.all(W[:, [0, 2, 3]].sum(1) == 3)


def test_verdicts():
    assert verdict(0.05, 0.02, 0.08, 0.01) == 'improvement'
    assert verdict(0.0, -0.005, 0.006, 0.01) == 'equivalent'
    assert verdict(-0.05, -0.08, -0.02, 0.01) == 'deterioration'
    assert verdict(0.02, -0.01, 0.05, 0.01) == 'inconclusive'
    s = summarize(0.05, np.linspace(0.02, 0.08, 101), 0.01)
    assert s['verdict'] == 'improvement'


def test_monotone_ramp_fit():
    x = np.arange(0, 10, 1 / 3)
    w = np.full(len(x), 50.0)
    p = 0.5 + 0.3 * (1 - np.exp(-x / 2))
    a = w * (p + np.random.default_rng(2).normal(0, 0.03, len(x)))
    knots = np.linspace(0, 10, 6)
    b = fit_logistic_ramp(x, w, a, knots, monotone=True)
    c = curve(b, knots, x)
    assert np.all(np.diff(c) >= -1e-12)
    assert abs(c[-1] - p[-1]) < 0.05


def _runs(gain_from, n_rep=3, seed=0):
    """Crashes whose PG score improves on G once the count exceeds `gain_from` (None: never)."""
    rng = np.random.default_rng(seed)
    n1, n0 = 200, 800
    y = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    base = rng.normal(0, 1, n1 + n0) + 0.5 * y
    runs = []
    for b in range(n_rep):
        for share in (1.0, 0.5, 0.25, 0.125):
            n = rng.uniform(0, 12, n1 + n0) * share
            boost = np.zeros(n1 + n0) if gain_from is None else 1.2 * y * (n > gain_from)
            g = base + rng.normal(0, 0.3, n1 + n0)
            runs.append(dict(share=share, replicate=b, n=n,
                             scores={'G': g, 'P': g + boost, 'PG': g + boost + rng.normal(0, 0.05, n1 + n0)}))
    nref = np.mean([r['n'] for r in runs if r['share'] == 1.0], axis=0)
    return runs, y, nref


def test_threshold_found_where_gain_starts():
    runs, y, nref = _runs(gain_from=4.0)
    W = cluster_weights(np.arange(len(y)) % 60, 100, 3)
    t = sufficiency_threshold(runs, y, np.zeros(len(y), int), np.ones(len(y), bool), W, nref, (0.01,), seed=1)
    assert t.threshold[0.01] is not None
    assert 2.0 <= t.threshold[0.01] <= 8.0
    assert t.curves['gain'].iloc[-1] > t.curves['gain'].iloc[0]


def test_no_threshold_without_gain():
    runs, y, nref = _runs(gain_from=None)
    W = cluster_weights(np.arange(len(y)) % 60, 100, 3)
    t = sufficiency_threshold(runs, y, np.zeros(len(y), int), np.ones(len(y), bool), W, nref, (0.01,), seed=1)
    assert t.threshold[0.01] is None
    assert len(t.notes) == 1
