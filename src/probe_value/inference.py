# -*- coding: utf-8 -*-
"""Placement values, AUC, PR-AUC and the crash-date cluster bootstrap (Sections IV-A and IV-H).

For crash i, the placement value psi_i is the share of controls of the same coverage group (and the same
penetration level) scored below it, with ties counted as one half; the mean placement value over the crashes of a
group equals the AUC of that group. Intervals come from resampling clusters (crash dates by default, all matched
groups of a date drawn together) with the fitted models and out-of-fold scores held fixed, so they are conditional on
the fitted models.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

Z90 = 1.6448536269514722


def placement_values(scores: np.ndarray, y: np.ndarray, coverage: Optional[np.ndarray] = None,
                     control_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Placement value of every crash (NaN for controls).

    control_weights: optional (R, n) weights of the samples (e.g. bootstrap weights); the result is then (R, n) with
    weighted shares of controls below each crash."""
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    c = np.zeros(len(s), int) if coverage is None else np.asarray(coverage)
    W = np.ones((1, len(s))) if control_weights is None else np.atleast_2d(control_weights).astype(float)
    out = np.full((W.shape[0], len(s)), np.nan)
    for cc in np.unique(c):
        i1 = np.flatnonzero((c == cc) & (y == 1))
        i0 = np.flatnonzero((c == cc) & (y == 0))
        if not len(i1) or not len(i0):
            continue
        o = np.argsort(s[i0], kind='stable')
        s0 = s[i0][o]
        cw = np.cumsum(W[:, i0[o]], axis=1)
        tot = cw[:, -1:]
        lo = np.searchsorted(s0, s[i1], 'left')
        hi = np.searchsorted(s0, s[i1], 'right')
        below = np.where(lo > 0, cw[:, np.maximum(lo - 1, 0)], 0.0)
        upto = np.where(hi > 0, cw[:, np.maximum(hi - 1, 0)], 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, i1] = 0.5 * (below + upto) / tot
    return out[0] if control_weights is None else out


def auc(scores: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None):
    """AUC of the samples in `mask` (all crashes against all controls). With (R, n) weights, returns (R,) values."""
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    s = np.asarray(scores, float)[m]
    yy = np.asarray(y, int)[m]
    if weights is None:
        psi = placement_values(s, yy)
        return float(np.nanmean(psi[yy == 1]))
    W = np.atleast_2d(weights)[:, m]
    psi = placement_values(s, yy, control_weights=W)
    w1 = W[:, yy == 1]
    return np.nansum(psi[:, yy == 1] * w1, axis=1) / w1.sum(axis=1)


def average_precision(scores: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None):
    """Average precision (PR-AUC) by `sklearn.metrics.average_precision_score`, which treats tied scores as one
    threshold. With (R, n) weights, the weighted average precision of each row (NaN for a row without weighted
    crashes)."""
    s = np.asarray(scores, float)
    yy = np.asarray(y).astype(int)
    if weights is None:
        return float(average_precision_score(yy, s))
    W = np.atleast_2d(np.asarray(weights, float))
    out = np.full(W.shape[0], np.nan)
    for r, w in enumerate(W):
        keep = w > 0
        if (w[keep] * yy[keep]).sum() > 0:
            out[r] = average_precision_score(yy[keep], s[keep], sample_weight=w[keep])
    return out


def cluster_weights(cluster: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) bootstrap weights: clusters drawn with replacement; each sample takes its cluster's count."""
    u, inv = np.unique(np.asarray(cluster), return_inverse=True)
    rng = np.random.default_rng(seed)
    C = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u)), size=n_boot).astype(float)
    return C[:, inv]


def crash_date_clusters(samples: pd.DataFrame) -> np.ndarray:
    """Cluster of each sample = calendar date of its matched group's crash."""
    return pd.to_datetime(samples['crash_time']).dt.normalize().to_numpy()


def verdict(est: float, lo: float, hi: float, margin: float) -> str:
    """'improvement' if the lower bound exceeds zero and the estimate reaches the margin; 'equivalent' if the interval
    lies within plus and minus the margin; 'deterioration' the mirror of improvement; otherwise 'inconclusive'."""
    if lo > 0 and est >= margin:
        return 'improvement'
    if lo > -margin and hi < margin:
        return 'equivalent'
    if hi < 0 and est <= -margin:
        return 'deterioration'
    return 'inconclusive'


def summarize(est: float, boot: np.ndarray, margin: float, level: float = 0.90) -> dict:
    a = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(boot, [a, 1.0 - a])
    return dict(est=float(est), lo=float(lo), hi=float(hi), se=float(np.nanstd(boot, ddof=1)),
                verdict=verdict(est, lo, hi, margin))


def mean_auc_over_replicates(score_list, y, mask=None, weights=None):
    """AUC averaged over training replicates (point estimate, or (R,) bootstrap values with weights)."""
    vals = [auc(s, y, mask, weights) for s in score_list]
    return np.mean(vals, axis=0)
