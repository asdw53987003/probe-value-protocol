# -*- coding: utf-8 -*-
"""Detection at an operating point set from training data only, and event-level warning (Section IV-H).

Alarm threshold: within the training set of each outer fold, the learner is cross-fitted on inner folds formed by
matched group, and a quantile (0.9) of the inner out-of-fold control scores is the alarm threshold of that fold.

Event level: every unit is scored each minute; an alarm follows k consecutive decisions above the alarm threshold, and
alarms on one unit within 15 min form one event. A crash counts as detected if an alarm falls 15 to 5 min before it
(a second reading also counts alarms in the last 4 min). False alarms are events away from all crashes, expressed per
100 km x day. Detection at another model's false-alarm rate is read by log-linear interpolation between alarm
thresholds of one family.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .folds import fit_model
from .learners import inner_folds


def inner_alarm_threshold(learner, X: pd.DataFrame, samples: pd.DataFrame, train: np.ndarray, seed: int = 0,
                          k_inner: int = 4, quantile: float = 0.9) -> float:
    """Alarm threshold of one outer fold from inner cross-fitting on its training samples."""
    Xt = X.loc[train]
    st = samples.loc[train]
    y = st['label'].to_numpy(int)
    g = st['group_id'].to_numpy()
    inn = inner_folds(g, k_inner)
    s = np.full(len(st), np.nan)
    for k in range(k_inner):
        tr, te = inn != k, inn == k
        m = fit_model(learner, Xt.loc[tr], y[tr], g[tr], seed)
        s[te] = m.predict(Xt.loc[te])
    return float(np.quantile(s[y == 0], quantile))


def sample_detection(scores: np.ndarray, y: np.ndarray, folds: np.ndarray, thresholds: dict,
                     mask: Optional[np.ndarray] = None) -> float:
    """Share of crashes whose score exceeds the alarm threshold of their fold."""
    m = (np.asarray(y) == 1) if mask is None else (np.asarray(mask, bool) & (np.asarray(y) == 1))
    th = np.array([thresholds[f] for f in np.asarray(folds)[m]])
    return float(np.mean(np.asarray(scores)[m] > th))


def alarm_minutes(timeline: pd.DataFrame, threshold, k: int = 1) -> pd.DataFrame:
    """Minutes at which an alarm is issued: the k-th of k consecutive decisions above the threshold.

    timeline: columns unit, minute (integer), score; one row per unit and minute. `threshold` is a scalar or a
    column name holding a threshold per row."""
    tl = timeline.sort_values(['unit', 'minute']).reset_index(drop=True)
    th = tl[threshold].to_numpy(float) if isinstance(threshold, str) else float(threshold)
    above = (tl['score'].to_numpy(float) > th).astype(int)
    unit = tl['unit'].to_numpy()
    minute = tl['minute'].to_numpy(int)
    run = np.zeros(len(tl), int)
    for i in range(len(tl)):
        cont = i > 0 and unit[i] == unit[i - 1] and minute[i] == minute[i - 1] + 1
        run[i] = above[i] * ((run[i - 1] + 1) if cont else 1)
    return tl.loc[run >= k, ['unit', 'minute']].reset_index(drop=True)


def events(alarms: pd.DataFrame, merge_minutes: int = 15) -> pd.DataFrame:
    """Alarms on one unit within `merge_minutes` of the previous alarm form one event (start, end)."""
    rows = []
    for u, g in alarms.sort_values(['unit', 'minute']).groupby('unit'):
        m = g['minute'].to_numpy()
        start = prev = m[0]
        for x in m[1:]:
            if x - prev > merge_minutes:
                rows.append((u, start, prev))
                start = x
            prev = x
        rows.append((u, start, prev))
    return pd.DataFrame(rows, columns=['unit', 'start', 'end'])


def detected(alarms: pd.DataFrame, crashes: pd.DataFrame, lead: Sequence[int] = (15, 5)) -> np.ndarray:
    """For each crash (columns unit, minute), whether an alarm on its unit falls lead[0] to lead[1] min before it."""
    out = np.zeros(len(crashes), bool)
    by_unit = {u: g['minute'].to_numpy() for u, g in alarms.groupby('unit')}
    for i, (u, t) in enumerate(zip(crashes['unit'], crashes['minute'])):
        m = by_unit.get(u)
        if m is not None:
            out[i] = np.any((m >= t - lead[0]) & (m <= t - lead[1]))
    return out


def false_alarm_rate(ev: pd.DataFrame, crashes: pd.DataFrame, km_days: float, exclude: Iterable = (-90, 30)) -> float:
    """False-alarm events per 100 km x day: events whose start lies outside [crash + exclude[0], crash + exclude[1]]
    minutes of every crash on the same unit. The default window (90 min before to 30 min after) follows the exclusion
    rule of the control windows; set it to the crash range used in an application."""
    lo, hi = exclude
    n = 0
    cr = {u: g['minute'].to_numpy() for u, g in crashes.groupby('unit')}
    for u, s in zip(ev['unit'], ev['start']):
        c = cr.get(u)
        if c is None or not np.any((s >= c + lo) & (s <= c + hi)):
            n += 1
    return 100.0 * n / km_days


def detection_at_rate(fa_rates: Sequence[float], detections: Sequence[float], target: float) -> float:
    """Detection at a target false-alarm rate, interpolated linearly in log false-alarm rate between the alarm
    thresholds of one family (sorted internally)."""
    fa = np.asarray(fa_rates, float)
    d = np.asarray(detections, float)
    ok = fa > 0
    o = np.argsort(fa[ok])
    return float(np.interp(np.log(target), np.log(fa[ok][o]), d[ok][o]))
