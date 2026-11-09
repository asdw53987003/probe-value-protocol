# -*- coding: utf-8 -*-
"""Algorithm 1: evaluation of probe data for real-time crash prediction.

    for each scenario (thinning, deletion, delay), level and replicate:
        degrade the probe records (or delay both feeds) for all samples with fixed random numbers
        compute the slice features of every sample
        for each model group G, P and PG: retrain on the training folds and score the test fold
        compute the placement value of each crash within its coverage group
    fit Delta_c(n) to the placement values of all thinning levels and replicates, Eq. (1)
    resample crash dates and form the 90% simultaneous band of Delta_c(n)
    n*_c <- smallest n whose lower band limit reaches epsilon, Eq. (2)
    for the reference and the attribution scenarios, and each of the eight channel subsets:
        retrain the subset on reference or degraded features and compute its AUC
    compute the Shapley value of each channel from the subset AUCs
    convert n*_0 into the required penetration of each unit and hour, Eqs. (3)-(4)

Inputs (see features.py and folds.py for the columns)
    records  delivered 1 Hz probe records around each sample's decision time
    gantry   gantry minute rows of each sample
    samples  indexed by sample_id: group_id, label, time, crash_time, unit, coverage (c: 1 if a same-direction
             gantry lies in the precursor zone of the crash, else 0; controls take the value of their crash),
             optionally night (0/1), and the context feature columns
    learner_factory(group) -> learner for model group 'C', 'G', 'P' or 'PG'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import degrade as D
from .features import gantry_features, model_matrix, probe_columns, slice_features, vehicle_slice_stats
from .folds import date_block_folds, out_of_fold, training_mask
from .inference import (average_precision, cluster_weights, crash_date_clusters, mean_auc_over_replicates,
                        summarize)
from .shapley import CHANNEL_BITS, shapley_values, subset_channels
from .threshold import ThresholdResult, sufficiency_threshold

AUC_MARGIN = 0.01


@dataclass
class Scenario:
    """A degradation of the delivered data. kind: 'delete' (arrangement, share, mean_burst, slice_index) or
    'delay' (minutes; probe_only=True delays the probe feed alone)."""
    name: str
    kind: str
    params: dict = field(default_factory=dict)


@dataclass
class ProtocolResult:
    reference: pd.DataFrame                 # AUC, PR-AUC and fusion gain at the reference, by coverage group
    thinning: pd.DataFrame                  # AUC and gain by retention share
    thresholds: Dict[str, ThresholdResult]
    scenarios: pd.DataFrame                 # Probe-only AUC and fusion gain under each degradation scenario
    channels: pd.DataFrame                  # Shapley values of the channels (reference value and losses)
    runs: List[dict]
    folds: pd.Series

    def summary(self) -> str:
        f = lambda v: '%.3f' % v                                    # noqa: E731
        out = ['Evaluation protocol (Algorithm 1)', '', 'Reference data:', self.reference.to_string(float_format=f),
               '', 'Thinning (retrained models):', self.thinning.to_string(float_format=f), '']
        for k, t in self.thresholds.items():
            thr = ', '.join('eps %g: %s' % (e, 'none' if v is None else '%.2f' % v) for e, v in t.threshold.items())
            out.append('Sufficiency threshold %s (%d crashes; observed range %.2f-%.2f vehicles per slice): %s'
                       % (k, t.crashes, t.n_p05, t.n_p75, thr))
            for nt in t.notes:
                out.append('  note: ' + nt)
        if len(self.scenarios):
            out += ['', 'Degradation scenarios:', self.scenarios.to_string(float_format=f)]
        if len(self.channels):
            out += ['', 'Channel values (Shapley, AUC scale):', self.channels.to_string(float_format=f)]
        return '\n'.join(out)


def _features(records, gfeat, vstats=None, keep=None):
    vs = vehicle_slice_stats(records) if vstats is None else vstats
    return slice_features(vs, gfeat.index, keep)


def _score(learner_factory, group, pf, gfeat, ctx, samples, folds, masks, seed, channels=('state', 'lon', 'lat')):
    X = model_matrix(group, pf, gfeat, ctx, channels)
    return out_of_fold(learner_factory(group), X, samples, folds, seed, train_masks=masks)


def run_protocol(records: pd.DataFrame, gantry: pd.DataFrame, samples: pd.DataFrame, context_columns: Sequence[str],
                 learner_factory: Callable[[str], object], shares: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 1 / 16),
                 replicates: int = 5, scenarios: Sequence[Scenario] = (), scenario_replicates: int = 3,
                 attribution: Sequence[str] = ('reference',), n_boot: int = 1000, epsilons: Sequence[float] = (0.01,),
                 seed: int = 1, by_daynight: bool = False, n_folds: int = 5, cluster: str = 'date') -> ProtocolResult:
    """Run the four steps of the protocol and return the readings.

    shares       retention shares of the thinning experiment (1 = reference data).
    attribution  'reference' and/or names of scenarios (and 'share=<x>' for a thinning level) whose losses are divided
                 among the channels; every channel subset or mixed input is retrained.
    cluster      resampling unit of the bootstrap: 'date' (crash dates, all matched groups of a date drawn together)
                 or 'group' (matched groups).
    """
    if scenario_replicates > replicates:
        raise ValueError('scenario_replicates (%d) cannot exceed replicates (%d): scenarios reuse the Gantry-only '
                         'scores and reference runs of the first replicates' % (scenario_replicates, replicates))
    samples = samples.copy()
    y = samples['label'].to_numpy(int)
    c = samples['coverage'].to_numpy(int)
    folds = date_block_folds(samples, n_folds)
    masks = {f: training_mask(samples, folds, f) for f in np.unique(folds)}
    ctx = samples[list(context_columns)].astype(float)
    gfeat = gantry_features(gantry, samples.index)
    vstats = vehicle_slice_stats(records)
    if cluster not in ('date', 'group'):
        raise ValueError("cluster must be 'date' or 'group'")
    cl = crash_date_clusters(samples) if cluster == 'date' else samples['group_id'].to_numpy()
    W = cluster_weights(cl, n_boot, seed + 1)
    groups = {'all': np.ones(len(y), bool), 'c0': c == 0, 'c1': c == 1}

    # steps 1-2: thinning, retrained at every level and replicate
    runs = []
    scores_G = {}
    for b in range(replicates):
        scores_G[b] = _score(learner_factory, 'G', None, gfeat, ctx, samples, folds, masks, seed + b)
        U = D.vehicle_uniforms(vstats['vehicle'], seed, b)
        for share in shares:
            keep = D.thin_mask(vstats, U, share)
            pf = slice_features(vstats, samples.index, keep)
            sc = {'G': scores_G[b]}
            for grp in ('P', 'PG'):
                sc[grp] = _score(learner_factory, grp, pf, gfeat, ctx, samples, folds, masks, seed + b)
            runs.append(dict(share=share, replicate=b, n=pf['n_bar'].to_numpy(), scores=sc))

    def block(rs, mask):
        out = {}
        for grp in ('G', 'P', 'PG'):
            sl = [r['scores'][grp] for r in rs]
            out[grp] = (mean_auc_over_replicates(sl, y, mask), mean_auc_over_replicates(sl, y, mask, W))
        return out

    rows = []
    ref_runs = [r for r in runs if r['share'] == 1.0]
    for lab, m in groups.items():
        if not np.any(m & (y == 1)):
            continue
        A = block(ref_runs, m)
        row = dict(subset=lab, crashes=int(np.sum(m & (y == 1))))
        for grp in ('G', 'P', 'PG'):
            row['AUC_' + grp] = A[grp][0]
            row['PRAUC_' + grp] = float(np.mean([average_precision(r['scores'][grp][m], y[m]) for r in ref_runs]))
        s = summarize(A['PG'][0] - A['G'][0], A['PG'][1] - A['G'][1], AUC_MARGIN)
        row.update(gain=s['est'], gain_lo=s['lo'], gain_hi=s['hi'], gain_verdict=s['verdict'])
        rows.append(row)
    reference = pd.DataFrame(rows).set_index('subset')

    trows = []
    for share in shares:
        rs = [r for r in runs if r['share'] == share]
        for lab in ('c0', 'c1'):
            m = groups[lab]
            if not np.any(m & (y == 1)):
                continue
            A = block(rs, m)
            trows.append(dict(share=share, subset=lab, n_bar=float(np.mean([r['n'][m & (y == 1)].mean() for r in rs])),
                              AUC_P=A['P'][0], AUC_PG=A['PG'][0], AUC_G=A['G'][0], gain=A['PG'][0] - A['G'][0]))
    thinning = pd.DataFrame(trows)

    # step 3: fusion-gain curves, simultaneous band and thresholds
    nref = np.mean([r['n'] for r in ref_runs], axis=0)
    thresholds = {}
    subsets = [('c0', c == 0), ('c1', c == 1)]
    if by_daynight and 'night' in samples.columns:
        nt = samples['night'].to_numpy(int)
        subsets += [('c0_day', (c == 0) & (nt == 0)), ('c0_night', (c == 0) & (nt == 1))]
    for lab, m in subsets:
        if np.sum(m & (y == 1)) >= 10:
            thresholds[lab] = sufficiency_threshold(runs, y, c, m, W, nref, epsilons, seed=seed + 2, label=lab)

    # degradation scenarios: Probe-only AUC and fusion gain against the reference
    srows = []
    deg_feats = {}
    for sc in scenarios:
        vals = {'P': [], 'PG': [], 'G': []}
        for b in range(scenario_replicates):
            rec = records
            gan = gantry
            if sc.kind == 'delete':
                rec = D.delete_records(records, seed=seed * 1000 + b, **sc.params)
            elif sc.kind == 'delay':
                if sc.params.get('probe_only'):
                    rec = D.delay(records, sc.params['minutes'])
                else:
                    rec, gan = D.delay(records, sc.params['minutes'], gantry)
            else:
                raise ValueError('unknown scenario kind %s' % sc.kind)
            gf = gantry_features(gan, samples.index) if gan is not gantry else gfeat
            pf = _features(rec, gf)
            deg_feats.setdefault(sc.name, pf)
            vals['G'].append(scores_G[b] if gf is gfeat else
                             _score(learner_factory, 'G', None, gf, ctx, samples, folds, masks, seed + b))
            for grp in ('P', 'PG'):
                vals[grp].append(_score(learner_factory, grp, pf, gf, ctx, samples, folds, masks, seed + b))
        for lab in ('all', 'c0', 'c1'):
            m = groups[lab]
            if not np.any(m & (y == 1)):
                continue
            ref = block([r for r in ref_runs if r['replicate'] < scenario_replicates], m)
            dP = mean_auc_over_replicates(vals['P'], y, m) - ref['P'][0]
            dPb = mean_auc_over_replicates(vals['P'], y, m, W) - ref['P'][1]
            gd = mean_auc_over_replicates(vals['PG'], y, m) - mean_auc_over_replicates(vals['G'], y, m)
            gdb = mean_auc_over_replicates(vals['PG'], y, m, W) - mean_auc_over_replicates(vals['G'], y, m, W)
            g0, g0b = ref['PG'][0] - ref['G'][0], ref['PG'][1] - ref['G'][1]
            sP = summarize(dP, dPb, AUC_MARGIN)
            sG = summarize(gd - g0, gdb - g0b, AUC_MARGIN)
            srows.append(dict(scenario=sc.name, subset=lab, dAUC_P=sP['est'], dAUC_P_lo=sP['lo'], dAUC_P_hi=sP['hi'],
                              gain=gd, dgain=sG['est'], dgain_lo=sG['lo'], dgain_hi=sG['hi']))
    scen_tab = pd.DataFrame(srows)

    # step 4: channel attribution with retrained subsets
    crow = []
    pf_ref = slice_features(vstats, samples.index)
    for target in attribution:
        if target == 'reference':
            for base_grp, grp in (('C', 'P'), ('G', 'PG')):
                v_pt, v_bt = {}, {}
                for S in range(8):
                    chans = subset_channels(S)
                    sl = []
                    for b in range(scenario_replicates):
                        if S == 0:
                            sl.append(_score(learner_factory, base_grp, pf_ref, gfeat, ctx, samples, folds, masks,
                                             seed + b))
                        else:
                            sl.append(_score(learner_factory, grp, pf_ref, gfeat, ctx, samples, folds, masks,
                                             seed + b, chans))
                    v_pt[S] = mean_auc_over_replicates(sl, y)
                    v_bt[S] = mean_auc_over_replicates(sl, y, None, W)
                phi_pt = shapley_values(v_pt)
                phi_bt = shapley_values(v_bt)
                for ch in CHANNEL_BITS:
                    s = summarize(phi_pt[ch], phi_bt[ch], AUC_MARGIN)
                    crow.append(dict(target='reference', model=grp, channel=ch, value=s['est'], lo=s['lo'],
                                     hi=s['hi'], total=v_pt[7] - v_pt[0]))
        else:
            if target.startswith('share='):
                share = float(target.split('=')[1])
                U = D.vehicle_uniforms(vstats['vehicle'], seed, 0)
                pf_deg = slice_features(vstats, samples.index, D.thin_mask(vstats, U, share))
            elif target in deg_feats:
                pf_deg = deg_feats[target]
            else:
                raise ValueError('attribution target %s is neither a scenario nor share=<x>' % target)
            v_pt, v_bt = {}, {}
            for S in range(8):
                mixed = pf_ref.copy()
                for ch in subset_channels(S):
                    cols = probe_columns([ch])
                    mixed[cols] = pf_deg[cols].to_numpy()
                sl = [_score(learner_factory, 'P', mixed, gfeat, ctx, samples, folds, masks, seed + b)
                      for b in range(scenario_replicates)]
                v_pt[S] = -mean_auc_over_replicates(sl, y)
                v_bt[S] = -mean_auc_over_replicates(sl, y, None, W)
            phi_pt, phi_bt = shapley_values(v_pt), shapley_values(v_bt)
            for ch in CHANNEL_BITS:
                s = summarize(phi_pt[ch], phi_bt[ch], AUC_MARGIN)
                crow.append(dict(target=target, model='P (loss)', channel=ch, value=s['est'], lo=s['lo'], hi=s['hi'],
                                 total=v_pt[7] - v_pt[0]))
    channels = pd.DataFrame(crow)
    return ProtocolResult(reference=reference, thinning=thinning, thresholds=thresholds, scenarios=scen_tab,
                          channels=channels, runs=runs, folds=folds)
