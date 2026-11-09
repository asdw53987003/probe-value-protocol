# -*- coding: utf-8 -*-
"""Step 1 of the protocol: controlled degradation of the delivered probe data, one dimension at a time, while the
crash and its traffic state stay fixed (Section IV-D).

Thinning      a vehicle is kept if its uniform random number falls below a retention share. The same numbers are used
              at every level, so the levels are nested and paired and the fleet composition is preserved.
Daily profile a retention share per hour of day (e.g. the hourly shares of the reference fleet, flat, or between).
Deletion      a share of the delivered vehicle-seconds of each input window (both units, three slices) is deleted;
              only the arrangement varies: random seconds; bursts of exponentially distributed length; bursts that
              start more often in slow traffic or tunnels; or bursts placed in one given slice.
Latency       probe and gantry inputs delayed together by whole minutes (negative values advance them); a delay of
              the probe feed alone is also available.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .features import N_SLICES, SLICE_S


def vehicle_uniforms(vehicles: Sequence, seed: int, replicate: int) -> pd.Series:
    """One uniform number per vehicle and replicate, shared by all retention levels of the replicate."""
    veh = pd.unique(np.asarray(vehicles))
    rng = np.random.default_rng([int(seed), int(replicate)])
    return pd.Series(rng.random(len(veh)), index=veh)


def thin_mask(vstats: pd.DataFrame, uniforms: pd.Series, share) -> np.ndarray:
    """Rows of vehicle-slice statistics (or records) kept at a retention share (scalar or one value per row)."""
    u = uniforms.reindex(vstats['vehicle'].to_numpy()).to_numpy(float)
    return u < np.asarray(share, float)


def profile_share(sample_hour: pd.Series, vstats: pd.DataFrame, hourly_share: pd.Series) -> np.ndarray:
    """Retention share of each vehicle-slice row from the hour of its sample and a share per hour of day (0-23)."""
    h = sample_hour.reindex(vstats['sample_id'].to_numpy()).to_numpy(int)
    return hourly_share.reindex(h).to_numpy(float)


def _bursts(t: np.ndarray, block: np.ndarray, quota: int, rng, mean_len: float, start_w: Optional[np.ndarray],
            alive: np.ndarray) -> np.ndarray:
    """Delete bursts within blocks (one vehicle's record sequence) until `quota` records are removed.
    A burst starts at a delivered record (drawn with weights `start_w`) and removes the records of the same block
    with time in [t0, t0 + L), L ~ exponential(mean_len)."""
    keep = np.ones(len(t), bool)
    removed = 0
    w = np.ones(len(t)) if start_w is None else start_w.astype(float)
    guard = 0
    while removed < quota and guard < 100000:
        guard += 1
        p = w * (keep & alive)
        if p.sum() <= 0:
            break
        j = rng.choice(len(t), p=p / p.sum())
        L = max(1, int(round(rng.exponential(mean_len))))
        k = j
        while k < len(t) and block[k] == block[j] and t[k] < t[j] + L and removed < quota:
            if keep[k] and alive[k]:
                keep[k] = False
                removed += 1
            k += 1
    return keep


def delete_records(records: pd.DataFrame, share: float, arrangement: str, seed: int, mean_burst: float = 76.0,
                   slow_speed: float = 20.0, slow_weight: float = 3.0, slice_index: Optional[int] = None,
                   window_end: float = 0.0) -> pd.DataFrame:
    """Delete `share` of the delivered records of each sample's input window.

    arrangement: 'random' (seconds at random), 'burst' (exponential bursts, mean `mean_burst` s), 'state' (bursts
    that start `slow_weight` times as often below `slow_speed` m/s or in tunnel units; needs a 'tunnel' column, 0/1),
    'slice' (the quota of the whole window placed as bursts inside slice `slice_index`, 0 = first, 2 = last; the
    paper uses bursts with a mean of 240 s for this contrast).
    Records outside the input window are kept."""
    if arrangement not in ('random', 'burst', 'state', 'slice'):
        raise ValueError('unknown arrangement %s' % arrangement)
    rng = np.random.default_rng(seed)
    r = records.sort_values(['sample_id', 'role', 'vehicle', 't'], kind='stable').reset_index(drop=True)
    lo = window_end - N_SLICES * SLICE_S
    inwin = ((r['t'] >= lo) & (r['t'] < window_end)).to_numpy()
    keep = np.ones(len(r), bool)
    block_all = pd.factorize(pd.MultiIndex.from_arrays([r['sample_id'], r['role'], r['vehicle']]))[0]
    for sid, idx in r.groupby('sample_id', sort=False).indices.items():
        idx = np.asarray(idx)
        win = inwin[idx]
        quota = int(round(share * win.sum()))
        if quota <= 0:
            continue
        if arrangement == 'random':
            cand = idx[win]
            keep[rng.choice(cand, quota, replace=False)] = False
            continue
        t = r['t'].to_numpy(float)[idx]
        alive = win.copy()
        start_w = None
        if arrangement == 'state':
            slow = r['speed'].to_numpy(float)[idx] < slow_speed
            if 'tunnel' in r.columns:
                slow |= r['tunnel'].to_numpy(int)[idx] == 1
            start_w = np.where(slow, slow_weight, 1.0)
        if arrangement == 'slice':
            s0 = lo + slice_index * SLICE_S
            alive = win & (t >= s0) & (t < s0 + SLICE_S)
            quota = min(quota, int(alive.sum()))
        k = _bursts(t, block_all[idx], quota, rng, mean_burst, start_w, alive)
        keep[idx] = k
    return r.loc[keep].reset_index(drop=True)


def delay(records: pd.DataFrame, minutes: float, gantry: Optional[pd.DataFrame] = None):
    """Delay a feed by whole minutes: its records appear 60 x minutes s later, so a decision at the same time sees
    the window [-900 - 60 m, -60 m). Negative minutes advance the feed (the window ends closer to the crash).
    Returns the shifted records (and the shifted gantry rows when given)."""
    r = records.copy()
    r['t'] = r['t'] + 60 * minutes
    if gantry is None:
        return r
    g = gantry.copy()
    g['t'] = g['t'] + 60 * minutes
    return r, g
