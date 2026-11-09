# -*- coding: utf-8 -*-
"""Step 3 of the protocol: fusion-gain curves, simultaneous band and the sufficiency threshold (Section IV-E).

Each conditional mean in Eq. (1), E[psi^k | n_bar, c] for k in {PG, G, P}, is a logistic curve in a monotone
piecewise-linear (ramp) basis of the probe count n_bar, fitted to the placement values of all crashes, thinning
levels and replicates (each replicate weighted 1/B). The PG and P curves are constrained to be non-decreasing; the G
curve is not, since gantry inputs do not change with thinning. The fusion gain is Delta_c(n) = E[psi^PG] - E[psi^G].

The observed count range runs from the 5th to the 75th percentile of the reference counts of the crashes in the
group. A 90% simultaneous band over this range is formed from the cluster bootstrap by the maximum standardized
deviation; the training variance, estimated by comparing bootstrap draws that use all replicates with draws that
use one random replicate, is divided by the number of replicates and added to each bootstrap variance.
Eq. (2): n*_c = min{ n in [n_0.05, n_0.75] : lower band limit >= epsilon }.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .inference import placement_values

N_KNOTS = 8
BIN = 1.0 / 3.0                 # counts are binned to 1/3 vehicle


def ramp_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    B = np.zeros((len(x), len(knots) - 1))
    for j in range(1, len(knots)):
        B[:, j - 1] = np.clip((x - knots[j - 1]) / (knots[j] - knots[j - 1]), 0, 1)
    return B


def fit_logistic_ramp(x: np.ndarray, w: np.ndarray, a: np.ndarray, knots: np.ndarray, monotone: bool = True,
                      ridge: float = 1e-3, init: Optional[np.ndarray] = None) -> np.ndarray:
    """Fractional-response logistic fit of a (weighted successes) out of w at x, in a ramp basis; non-negative ramp
    coefficients make the curve non-decreasing."""
    Bm = ramp_basis(x, knots)
    m = w > 0
    Bm, w, a = Bm[m], w[m], a[m]
    p0 = np.clip(a.sum() / w.sum(), 1e-3, 1 - 1e-3)

    def nll(beta):
        eta = beta[0] + Bm @ beta[1:]
        r = a - w / (1 + np.exp(-eta))
        f = -(a * eta - w * np.logaddexp(0, eta)).sum() / w.sum() + ridge * np.sum(beta[1:] ** 2)
        g = np.r_[-r.sum(), -Bm.T @ r] / w.sum() + np.r_[0, 2 * ridge * beta[1:]]
        return f, g
    x0 = init if init is not None else np.r_[np.log(p0 / (1 - p0)), np.zeros(Bm.shape[1])]
    bounds = [(None, None)] + [((0, None) if monotone else (None, None))] * Bm.shape[1]
    return minimize(nll, x0, jac=True, method='L-BFGS-B', bounds=bounds).x


def curve(beta: np.ndarray, knots: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(beta[0] + ramp_basis(grid, knots) @ beta[1:])))


@dataclass
class ThresholdResult:
    label: str
    crashes: int
    controls: int
    n_p05: float
    n_p75: float
    c90: float
    threshold: Dict[float, Optional[float]]            # Eq. (2) with the simultaneous band, per epsilon
    threshold_pointwise: Dict[float, Optional[float]]
    sufficient_range: Dict[float, Optional[List[float]]]
    falls_back: Dict[float, bool]
    curves: pd.DataFrame                               # n, gain, band, pointwise band, A_PG, A_G, A_P, in_range
    notes: list = field(default_factory=list)


def _band(G0, Gr, supp, sd_use):
    sd = Gr.std(axis=0, ddof=1)
    sds = np.where(sd > 1e-9, sd, 1e-9)
    T = np.max(np.abs(Gr[:, supp] - G0[supp]) / sds[supp], axis=1)
    c90 = float(np.quantile(T, 0.90))
    return G0 - c90 * sd_use, G0 + c90 * sd_use, c90


def _first(grid, ok):
    return float(grid[np.flatnonzero(ok)[0]]) if ok.any() else None


def sufficiency_threshold(runs: Sequence[dict], y: np.ndarray, coverage: np.ndarray, mask: np.ndarray,
                          boot_weights: np.ndarray, reference_counts: np.ndarray, epsilons: Sequence[float] = (0.01,),
                          seed: int = 0, label: str = '') -> ThresholdResult:
    """Fusion-gain curve, 90% simultaneous band and threshold for the crashes and controls in `mask`.

    runs              one dict per thinning level x replicate: {'replicate': b, 'n': (n,) probe counts per sample,
                      'scores': {'PG': (n,), 'G': (n,), 'P': (n,)}} (out-of-fold scores of retrained models).
    boot_weights      (R, n) cluster-bootstrap weights of the samples (e.g. inference.cluster_weights on crash dates).
    reference_counts  (n,) probe counts at the reference level (mean over replicates), for the observed count range.
    """
    y = np.asarray(y, int)
    mask = np.asarray(mask, bool)
    i1 = np.flatnonzero(mask & (y == 1))
    i0 = np.flatnonzero(mask & (y == 0))
    reps = sorted({r['replicate'] for r in runs})
    B = len(reps)
    nref = np.asarray(reference_counts, float)[i1]
    p5, p75, p90 = (float(np.quantile(nref, q)) for q in (0.05, 0.75, 0.90))
    top = np.floor(p90 / BIN) * BIN
    nb = int(round(top / BIN)) + 1
    centers = np.arange(nb) * BIN
    allc = np.concatenate([np.minimum(np.asarray(r['n'], float)[i1], top) for r in runs])
    knots = np.unique(np.r_[0.0, np.quantile(allc, np.linspace(0.02, 0.98, N_KNOTS - 1)), top + 1e-6])
    if len(knots) < 4:
        knots = np.linspace(0, top + 1e-6, N_KNOTS)
    grid = np.round(np.arange(0, top + 1e-9, BIN), 6)
    supp = (grid >= p5) & (grid <= p75)
    W = np.asarray(boot_weights, float)
    R = W.shape[0]
    rng = np.random.default_rng(seed)
    rb = np.asarray(reps)[rng.integers(0, B, R)]               # one random replicate per draw (single-model spread)
    c = np.asarray(coverage)
    w1A = np.vstack([np.ones((1, len(i1))), W[:, i1]])
    w0A = np.vstack([np.ones((1, len(i0))), W[:, i0]])
    WA = np.zeros((R + 1, nb))
    VA = {k: np.zeros((R + 1, nb)) for k in ('PG', 'G', 'P')}
    WB = np.zeros((R, nb))
    VB = {k: np.zeros((R, nb)) for k in ('PG', 'G', 'P')}
    for r in runs:
        nn = np.asarray(r['n'], float)[i1]
        bi = np.minimum(np.round(nn / BIN).astype(int), nb - 1)
        oh = np.zeros((len(i1), nb))
        oh[np.arange(len(i1)), bi] = 1.0
        WA += (w1A @ oh) / B
        rr = np.flatnonzero(rb == r['replicate'])
        if len(rr):
            WB[rr] += w1A[rr + 1] @ oh
        for k in ('PG', 'G', 'P'):
            s = np.asarray(r['scores'][k], float)
            sc = np.r_[s[i1], s[i0]]
            yy = np.r_[np.ones(len(i1), int), np.zeros(len(i0), int)]
            Wc = np.hstack([w1A, w0A])
            v = placement_values(sc, yy, np.r_[c[i1], c[i0]], Wc)[:, :len(i1)]
            VA[k] += ((w1A * v) @ oh) / B
            if len(rr):
                VB[k][rr] += (w1A[rr + 1] * v[rr + 1]) @ oh
    CA = {k: np.zeros((R + 1, len(grid))) for k in VA}
    CB = {k: np.zeros((R, len(grid))) for k in VA}
    for k in VA:
        mono = k != 'G'
        b0 = fit_logistic_ramp(centers, WA[0], VA[k][0], knots, mono)
        CA[k][0] = curve(b0, knots, grid)
        for i in range(R):
            CA[k][i + 1] = curve(fit_logistic_ramp(centers, WA[i + 1], VA[k][i + 1], knots, mono, init=b0), knots, grid)
            CB[k][i] = curve(fit_logistic_ramp(centers, WB[i], VB[k][i], knots, mono, init=b0), knots, grid)
    G0 = CA['PG'][0] - CA['G'][0]
    GA = CA['PG'][1:] - CA['G'][1:]
    GB = CB['PG'] - CB['G']
    sdA, sdB = GA.std(axis=0, ddof=1), GB.std(axis=0, ddof=1)
    var_train = np.maximum(sdB ** 2 - sdA ** 2, 0.0)
    sd_main = np.sqrt(sdA ** 2 + var_train / B)
    lo, hi, c90 = _band(G0, GA, supp, sd_main)
    lo_pw, hi_pw = np.quantile(GA, 0.05, axis=0), np.quantile(GA, 0.95, axis=0)
    thr, thr_pw, rng_ok, fb = {}, {}, {}, {}
    for e in epsilons:
        ok = supp & (lo >= e)
        thr[e] = _first(grid, ok)
        thr_pw[e] = _first(grid, supp & (lo_pw >= e))
        if ok.any():
            idx = np.flatnonzero(ok)
            rng_ok[e] = [float(grid[idx[0]]), float(grid[idx[-1]])]
            fb[e] = bool(np.any(supp & (grid >= grid[idx[0]]) & (lo < e)))
        else:
            rng_ok[e], fb[e] = None, False
    curves = pd.DataFrame(dict(n=grid, gain=G0, lo=lo, hi=hi, lo_pointwise=lo_pw, hi_pointwise=hi_pw,
                               A_PG=CA['PG'][0], A_G=CA['G'][0], A_P=CA['P'][0], in_range=supp.astype(int)))
    notes = []
    for e in epsilons:
        if thr[e] is None:
            notes.append('epsilon %g: no count in the observed range qualifies; no threshold is reported' % e)
        elif fb[e]:
            notes.append('epsilon %g: the lower band limit falls below the margin again within the range' % e)
    return ThresholdResult(label=label, crashes=len(i1), controls=len(i0), n_p05=p5, n_p75=p75, c90=c90,
                           threshold=thr, threshold_pointwise=thr_pw, sufficient_range=rng_ok, falls_back=fb,
                           curves=curves, notes=notes)
