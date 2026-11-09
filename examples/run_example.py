# -*- coding: utf-8 -*-
"""Run the evaluation protocol (Algorithm 1) on the example data in examples/data/.

Small settings keep the run short: five retention shares, three training replicates, 200 bootstrap draws, a
logistic-regression learner, five degradation scenarios with two replicates, and channel attribution at the
reference and at a quarter of the probe vehicles. It also shows the alarm threshold set from training data only and
the required penetration for an example panel of unit-hours. Writes examples/example_output.txt, or the file given
with `--out`.

Use `--lightgbm` to run the same protocol with LightGBM (needs the lightgbm package; slower).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from probe_value import LightGBMLearner, LogisticLearner, Scenario, lightgbm_settings, run_protocol  # noqa: E402
from probe_value import alarms, penetration  # noqa: E402
from probe_value.features import gantry_features, model_matrix, slice_features, vehicle_slice_stats  # noqa: E402
from probe_value.folds import date_block_folds, out_of_fold, training_mask  # noqa: E402

DATA = os.path.join(HERE, 'data')
CONTEXT = ['TOD', 'NIGHT', 'DAYTYPE', 'RAIN', 'ROUTE', 'CURV', 'GRADE', 'TUNNEL', 'GCOV']


def load():
    if not os.path.exists(os.path.join(DATA, 'records.csv.gz')):
        raise SystemExit('example data not found in examples/data/')
    rec = pd.read_csv(os.path.join(DATA, 'records.csv.gz'))
    gan = pd.read_csv(os.path.join(DATA, 'gantry.csv.gz'))
    smp = pd.read_csv(os.path.join(DATA, 'samples.csv'), index_col=0, parse_dates=['time', 'crash_time'])
    return rec, gan, smp


def main(use_lightgbm: bool = False, out_path: str = None):
    rec, gan, smp = load()
    if use_lightgbm:
        factory = lambda grp: LightGBMLearner(lightgbm_settings('PG' if grp == 'C' else grp))   # noqa: E731
    else:
        factory = lambda grp: LogisticLearner(C=0.1)                                           # noqa: E731
    scen = [Scenario('random 20%', 'delete', dict(share=0.2, arrangement='random')),
            Scenario('bursts 76 s 20%', 'delete', dict(share=0.2, arrangement='burst', mean_burst=76)),
            Scenario('last slice 20%', 'delete', dict(share=0.2, arrangement='slice', slice_index=2, mean_burst=240)),
            Scenario('first slice 20%', 'delete', dict(share=0.2, arrangement='slice', slice_index=0, mean_burst=240)),
            Scenario('delay 2 min', 'delay', dict(minutes=2))]
    res = run_protocol(rec, gan, smp, CONTEXT, factory, shares=(1.0, 0.5, 0.25, 0.125, 1 / 16), replicates=3,
                       scenarios=scen, scenario_replicates=2, attribution=('reference', 'share=0.25'), n_boot=200,
                       epsilons=(0.01,), seed=1)
    out = [res.summary(), '']

    # alarm thresholds from training data only (inner folds), Probe + Gantry at the reference
    folds = date_block_folds(smp)
    vs = vehicle_slice_stats(rec)
    X = model_matrix('PG', slice_features(vs, smp.index), gantry_features(gan, smp.index), smp[CONTEXT].astype(float))
    masks = {f: training_mask(smp, folds, f) for f in np.unique(folds)}
    s = out_of_fold(factory('PG'), X, smp, folds, seed=1, train_masks=masks)
    thr = {f: alarms.inner_alarm_threshold(factory('PG'), X, smp, masks[f], seed=1) for f in masks}
    det = alarms.sample_detection(s, smp['label'].to_numpy(), folds.to_numpy(), thr)
    out += ['Probe + Gantry detection at alarm thresholds from inner folds (90th percentile of control scores): %.3f'
            % det, '']

    # required penetration for an example panel of unit-hours, threshold of the uncovered group
    t0 = res.thresholds.get('c0')
    n_star = t0.threshold.get(0.01) if t0 is not None else None
    if n_star is not None:
        rng = np.random.default_rng(5)
        panel = pd.DataFrame({'unit': np.repeat(['u%02d' % i for i in range(6)], 24), 'hour': np.tile(np.arange(24), 6)})
        day = (panel['hour'] >= 6) & (panel['hour'] < 20)
        panel['probe_flow_per_min'] = np.where(day, 2.4, 0.8) * rng.uniform(0.6, 1.4, len(panel))
        panel['speed_mps'] = rng.uniform(20, 28, len(panel))
        panel['unit_length_m'] = 500.0
        panel['phi'] = 0.95
        tab = penetration.requirement_table(panel, n_star, theta=5.0, p_ref=0.08)
        summ = tab.groupby(day.map({True: 'day', False: 'night'}))['p_req'].median()
        out += ['Required penetration for n* = %.2f (median over example unit-hours): %s'
                % (n_star, ', '.join('%s %.3f' % kv for kv in summ.items())), '']
    text = '\n'.join(out)
    if out_path is None:
        out_path = os.path.join(HERE, 'example_output%s.txt' % ('_lightgbm' if use_lightgbm else ''))
    open(out_path, 'w', encoding='utf-8').write(text + '\n')
    print(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lightgbm', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    main(a.lightgbm, a.out)
