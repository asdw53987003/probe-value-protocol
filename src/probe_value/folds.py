# -*- coding: utf-8 -*-
"""Step 2 of the protocol: folds and out-of-fold scoring (Section IV-H).

Five folds are formed from consecutive blocks of crash dates, and each matched group stays in the fold of its crash.
Every sample is scored by a model trained without its fold. Training samples within 7 days of a test crash are removed
to limit temporal leakage (the controls of a test crash do not enter this buffer), and no training sample shares its
unit and an overlapping input window with a test sample.

`samples` needs the columns sample_id (index), group_id (matched group), label (1 crash, 0 control), time (decision
time, datetime64), crash_time (time of the group's crash), unit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def date_block_folds(samples: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    """Fold of each sample: crash dates sorted and cut into consecutive blocks with similar numbers of crashes."""
    crashes = samples[samples['label'] == 1]
    day = pd.to_datetime(crashes['crash_time']).dt.normalize()
    per_day = day.value_counts().sort_index()
    cum = per_day.cumsum() / per_day.sum()
    fold_of_day = pd.Series(np.minimum((cum.to_numpy() - 1e-12) * n_folds, n_folds - 1).astype(int), index=cum.index)
    fold_of_group = pd.Series(fold_of_day.reindex(day).to_numpy(), index=crashes['group_id'].to_numpy())
    return pd.Series(fold_of_group.reindex(samples['group_id']).to_numpy(), index=samples.index, name='fold')


def training_mask(samples: pd.DataFrame, folds: pd.Series, f: int, buffer_days: float = 7.0,
                  window_minutes: float = 15.0) -> np.ndarray:
    """Training samples of outer fold f after the temporal buffer and the unit-window overlap rule."""
    fo = folds.reindex(samples.index).to_numpy()
    train = fo != f
    t = pd.to_datetime(samples['time']).to_numpy('datetime64[s]').astype(np.int64)
    test_crash_t = np.unique(pd.to_datetime(samples.loc[fo == f, 'crash_time']).to_numpy('datetime64[s]')
                             .astype(np.int64))
    if len(test_crash_t):
        pos = np.searchsorted(test_crash_t, t)
        lo = np.abs(t - test_crash_t[np.clip(pos - 1, 0, len(test_crash_t) - 1)])
        hi = np.abs(t - test_crash_t[np.clip(pos, 0, len(test_crash_t) - 1)])
        near = np.minimum(lo, hi) < buffer_days * 86400
        train &= ~near
    test = samples.loc[fo == f, ['unit']].assign(t=t[fo == f])
    if len(test):
        tr_idx = np.flatnonzero(train)
        trd = pd.DataFrame({'unit': samples['unit'].to_numpy()[tr_idx], 't': t[tr_idx], 'i': tr_idx})
        m = trd.merge(test, on='unit', suffixes=('', '_test'))
        clash = m.loc[np.abs(m['t'] - m['t_test']) < window_minutes * 60, 'i'].unique()
        train[clash] = False
    return train


def fit_model(learner, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int):
    if getattr(learner, 'needs_groups', False):
        return learner.fit(X, y, groups, seed)
    return learner.fit(X, y, seed)


def out_of_fold(learner, X: pd.DataFrame, samples: pd.DataFrame, folds: pd.Series, seed: int = 0,
                X_test: Optional[pd.DataFrame] = None, train_masks: Optional[dict] = None, keep_models: bool = False):
    """Out-of-fold scores: each fold scored by a model trained on the remaining folds (after the buffers).

    `X_test` replaces the inputs of the test folds (fixed-model checks: reference models, degraded test inputs)."""
    y = samples['label'].to_numpy(int)
    g = samples['group_id'].to_numpy()
    fo = folds.reindex(samples.index).to_numpy()
    scores = np.full(len(samples), np.nan)
    models = {}
    for f in np.unique(fo):
        tr = train_masks[f] if train_masks is not None else training_mask(samples, folds, f)
        te = fo == f
        m = fit_model(learner, X.loc[tr], y[tr], g[tr], seed)
        scores[te] = m.predict((X if X_test is None else X_test).loc[te])
        if keep_models:
            models[f] = m
    return (scores, models) if keep_models else scores
