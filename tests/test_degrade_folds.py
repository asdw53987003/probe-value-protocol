import numpy as np
import pandas as pd

from probe_value import degrade as D
from probe_value.folds import date_block_folds, training_mask


def _records(n_samples=3, n_veh=4, length=60):
    rows = []
    for s in range(n_samples):
        for v in range(n_veh):
            for t in range(-900 + 30 * v, -900 + 30 * v + length):
                rows.append(dict(sample_id='s%d' % s, role='T', vehicle='s%d-v%d' % (s, v), t=t,
                                 speed=15.0 if t < -800 else 25.0, tunnel=0))
    return pd.DataFrame(rows)


def test_thinning_is_nested_and_paired():
    r = _records()
    U = D.vehicle_uniforms(r['vehicle'], 1, 0)
    k1 = D.thin_mask(r, U, 0.5)
    k2 = D.thin_mask(r, U, 0.25)
    assert np.all(k1 | ~k2)                          # kept at 0.25 implies kept at 0.5
    U2 = D.vehicle_uniforms(r['vehicle'], 1, 0)
    assert U.equals(U2)
    assert not U.equals(D.vehicle_uniforms(r['vehicle'], 1, 1))


def test_random_deletion_quota_per_sample():
    r = _records()
    out = D.delete_records(r, 0.2, 'random', seed=3)
    for s, g in r.groupby('sample_id'):
        assert len(g) - (out['sample_id'] == s).sum() == round(0.2 * len(g))


def test_bursts_and_slice_placement():
    r = _records(n_samples=2, n_veh=6, length=700)
    out = D.delete_records(r, 0.2, 'burst', seed=1, mean_burst=76)
    assert len(r) - len(out) == sum(round(0.2 * len(g)) for _, g in r.groupby('sample_id'))
    last = D.delete_records(r, 0.1, 'slice', seed=1, slice_index=2, mean_burst=240)
    removed = r.merge(last, how='left', indicator=True)
    removed = removed[removed['_merge'] == 'left_only']
    assert len(removed) > 0 and removed['t'].min() >= -300
    st = D.delete_records(r, 0.2, 'state', seed=1)
    assert len(st) < len(r)


def test_delay_shifts_both_feeds():
    r = _records(1, 1, 5)
    g = pd.DataFrame({'sample_id': 's0', 't': [-60], 'flow': 1})
    r2, g2 = D.delay(r, 2, g)
    assert (r2['t'] - r['t']).eq(120).all() and g2['t'].iloc[0] == 60


def _samples():
    rows = []
    days = pd.date_range('2025-01-01', periods=50, freq='7D')
    for i, d in enumerate(days):
        tc = d + pd.Timedelta(hours=10)
        for j in range(3):
            tt = tc + pd.Timedelta(weeks=(0 if j == 0 else (j if i < 25 else -j)))
            rows.append(dict(sample_id='g%d-%d' % (i, j), group_id='g%d' % i, label=int(j == 0), unit='u%d' % (i % 5),
                             time=tt - pd.Timedelta(minutes=5), crash_time=tc))
    return pd.DataFrame(rows).set_index('sample_id')


def test_date_block_folds_keep_groups_and_order():
    s = _samples()
    f = date_block_folds(s, 5)
    assert f.groupby(s['group_id']).nunique().max() == 1
    crash_fold = f[s['label'] == 1]
    order = s.loc[s['label'] == 1, 'crash_time'].sort_values().index
    assert crash_fold[order].is_monotonic_increasing
    assert crash_fold.value_counts().min() >= 9


def test_training_mask_buffer_and_overlap():
    s = _samples()
    f = date_block_folds(s, 5)
    tr = training_mask(s, f, 2)
    fo = f.to_numpy()
    assert not np.any(tr & (fo == 2))
    test_crashes = pd.to_datetime(s.loc[fo == 2, 'crash_time']).unique()
    t = pd.to_datetime(s['time'])
    for tc in test_crashes:
        near = (t - tc).abs() < pd.Timedelta(days=7)
        assert not np.any(tr & near.to_numpy())
