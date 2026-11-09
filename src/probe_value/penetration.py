# -*- coding: utf-8 -*-
"""Required penetration (Section IV-G).

Eq. (3): E[n_uh] = p_uh q_uh (tau + l_u / s_uh) phi_u, the expected probe vehicles per slice of unit u in hour h, with
p q the probe flow observed at the unit's gantry (veh/min), tau the slice length (5 min), l_u the unit length, s_uh the
hourly speed and phi_u the share of probe passes with at least three recorded seconds that keep at least three
delivered points. The bracket counts the vehicles present when a slice starts plus those entering during it.

mu_req is the smallest expected slice count at which a share (0.8) of input windows reach the threshold n*, with the
three-slice sum taken as negative binomial. Eq. (4): p_req = p_bar mu_req / E[n_uh | p_bar], with p_bar the
network-average penetration of the reference data; the probe fleet is assumed to scale in proportion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

SLICE_MIN = 5.0


def expected_slice_count(probe_flow_per_min, speed_mps, unit_length_m, phi, slice_min: float = SLICE_MIN):
    """Eq. (3) in vehicles per slice."""
    travel_min = np.asarray(unit_length_m, float) / np.asarray(speed_mps, float) / 60.0
    return np.asarray(probe_flow_per_min, float) * (slice_min + travel_min) * np.asarray(phi, float)


def nb_size_moments(three_slice_sums: np.ndarray, expected: np.ndarray) -> float:
    """Method-of-moments size theta of the negative binomial for three-slice sums S with means 3 E[n]:
    Var(S) = m + m^2 / theta."""
    S = np.asarray(three_slice_sums, float)
    m = 3.0 * np.asarray(expected, float)
    excess = np.sum((S - m) ** 2 - m)
    return float(np.sum(m ** 2) / excess) if excess > 0 else np.inf


def prob_reaching(mu: float, n_star: float, theta: float) -> float:
    """P(mean count over three slices >= n_star) when the three-slice sum is NB with mean 3 mu and size theta."""
    k = int(np.ceil(3.0 * n_star - 1e-4))          # thresholds lie on a 1/3 grid rounded to 6 decimals
    m = 3.0 * mu
    if not np.isfinite(theta):
        return float(stats.poisson.sf(k - 1, m))
    return float(stats.nbinom.sf(k - 1, theta, theta / (theta + m)))


def mu_required(n_star: float, theta: float, share: float = 0.8) -> float:
    """Smallest expected slice count at which `share` of input windows reach n_star."""
    f = lambda mu: prob_reaching(mu, n_star, theta) - share          # noqa: E731
    hi = max(1.0, n_star)
    while f(hi) < 0:
        hi *= 2.0
    return float(optimize.brentq(f, 1e-6, hi, xtol=1e-6))


def required_penetration(mu_req: float, expected_at_reference, p_ref: float) -> np.ndarray:
    """Eq. (4)."""
    return p_ref * mu_req / np.asarray(expected_at_reference, float)


def calibration_ratio(observed_counts, predicted_counts) -> float:
    """Single ratio of observed to predicted slice counts on a scoring panel."""
    return float(np.nansum(observed_counts) / np.nansum(predicted_counts))


def requirement_table(units_hours: pd.DataFrame, n_star: float, theta: float, p_ref: float, ratio: float = 1.0,
                      share: float = 0.8) -> pd.DataFrame:
    """Required penetration of every unit and hour from columns probe_flow_per_min, speed_mps, unit_length_m, phi
    (reference fleet, averaged by hour of day)."""
    e = expected_slice_count(units_hours['probe_flow_per_min'], units_hours['speed_mps'], units_hours['unit_length_m'],
                             units_hours['phi']) * ratio
    mu = mu_required(n_star, theta, share)
    out = units_hours.copy()
    out['expected_count'] = e
    out['mu_req'] = mu
    out['p_req'] = required_penetration(mu, e, p_ref)
    return out
