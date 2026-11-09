# -*- coding: utf-8 -*-
"""Learner interface. Any learner that scores input slices can be used: it needs `fit(X, y, seed)` returning an object
with `predict(X)` that gives one score per row (higher = more crash-like). Missing features (empty slices) must be
handled by the learner.

Provided:
- LogisticLearner: logistic regression with mean imputation and missing-value indicators (scikit-learn).
- LightGBMLearner: gradient-boosted trees that handle missing values natively (optional dependency `lightgbm`).
  `lightgbm_settings(group)` returns the settings of the paper for G, P and PG.
- NestedFusion: per outer fold, chooses among candidate (learner, columns) pairs by the AUC of the pooled inner
  out-of-fold scores, with inner folds formed by matched group; with LightGBM the candidates scale the split gains of the probe features by 1, 0.5 or 0.2,
  or fall back to the Gantry-only model.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class LogisticLearner:
    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y: np.ndarray, seed: int = 0):
        Xa = np.asarray(X, float)
        mean = np.nanmean(Xa, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        miss_cols = np.isnan(Xa).any(axis=0)
        Z = self._design(Xa, mean, miss_cols)
        mu, sd = Z.mean(0), Z.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        m = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=seed)
        m.fit((Z - mu) / sd, np.asarray(y, int))
        return _FittedLogistic(m, mean, miss_cols, mu, sd, self._design)

    @staticmethod
    def _design(Xa, mean, miss_cols):
        miss = np.isnan(Xa)
        filled = np.where(miss, mean[None, :], Xa)
        return np.column_stack([filled, miss[:, miss_cols].astype(float)])


class _FittedLogistic:
    def __init__(self, m, mean, miss_cols, mu, sd, design):
        self.m, self.mean, self.miss_cols, self.mu, self.sd, self.design = m, mean, miss_cols, mu, sd, design

    def predict(self, X):
        Z = self.design(np.asarray(X, float), self.mean, self.miss_cols)
        return self.m.decision_function((Z - self.mu) / self.sd)


def lightgbm_settings(group: str) -> dict:
    """LightGBM settings by model group (binary objective, 150 trees)."""
    base = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1)
    tuned = {'P': dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7),
             'G': dict(num_leaves=15, min_data_in_leaf=50, feature_fraction=1.0),
             'PG': dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7)}
    base.update(tuned.get(group, {}))
    return base


class LightGBMLearner:
    def __init__(self, params: Optional[dict] = None, n_trees: int = 150, feature_contri: Optional[Sequence] = None):
        self.params = dict(params or lightgbm_settings('PG'))
        self.n_trees = n_trees
        self.feature_contri = feature_contri

    def fit(self, X: pd.DataFrame, y: np.ndarray, seed: int = 0):
        import lightgbm as lgb
        pr = dict(self.params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        if self.feature_contri is not None:
            pr['feature_contri'] = list(self.feature_contri)
        return lgb.train(pr, lgb.Dataset(np.asarray(X, float), np.asarray(y, int)), num_boost_round=self.n_trees)


def inner_folds(group_ids: Sequence, k: int) -> np.ndarray:
    """Inner folds by matched group: the sorted groups are dealt to k folds in turn."""
    g = np.asarray(group_ids)
    ug = np.unique(g)
    pos = {u: i % k for i, u in enumerate(ug)}
    return np.array([pos[x] for x in g])


class NestedFusion:
    """Chooses, on the training data of each outer fold, among candidates (name, learner, columns) by the AUC of the
    pooled out-of-fold scores of `k_inner` inner folds formed by matched group, then refits the chosen candidate on
    all training data."""

    needs_groups = True

    def __init__(self, candidates: List[Tuple[str, object, List[str]]], k_inner: int = 3):
        self.candidates = candidates
        self.k_inner = k_inner
        self.choices: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 0):
        inn = inner_folds(groups, self.k_inner)
        best, best_auc = None, -np.inf
        for name, learner, cols in self.candidates:
            s = np.full(len(y), np.nan)
            for k in range(self.k_inner):
                tr, te = inn != k, inn == k
                m = learner.fit(X.loc[tr, cols], y[tr], seed)
                s[te] = m.predict(X.loc[te, cols])
            auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
            if auc > best_auc:
                best, best_auc = (name, learner, cols), auc
        name, learner, cols = best
        self.choices.append(name)
        m = learner.fit(X[cols], y, seed)
        return _Restricted(m, cols, name)


class _Restricted:
    def __init__(self, model, cols, name):
        self.model, self.cols, self.name = model, cols, name

    def predict(self, X):
        return self.model.predict(X[self.cols])


def lightgbm_fusion_candidates(columns: Sequence[str], probe_cols: Sequence[str], gantry_context_cols: Sequence[str],
                               weights: Sequence[float] = (1.0, 0.5, 0.2)) -> list:
    """Candidates of the nested selection: probe split gains scaled by each weight, or the Gantry-only fallback."""
    probe = set(probe_cols)
    out = []
    for w in weights:
        fc = [w if c in probe else 1.0 for c in columns]
        out.append(('omega=%g' % w, LightGBMLearner(lightgbm_settings('PG'), feature_contri=fc), list(columns)))
    out.append(('fallback', LightGBMLearner(lightgbm_settings('G')), list(gantry_context_cols)))
    return out
