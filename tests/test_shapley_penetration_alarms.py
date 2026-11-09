import numpy as np
import pandas as pd
from scipy import stats

from probe_value import alarms, penetration
from probe_value.learners import LogisticLearner, NestedFusion, inner_folds
from probe_value.shapley import orders_check, shapley_values


def test_shapley_efficiency_and_orders():
    v = {0: 0.0, 1: 0.10, 2: 0.05, 4: 0.02, 3: 0.12, 5: 0.11, 6: 0.06, 7: 0.13}
    phi = shapley_values(v)
    assert abs(sum(phi.values()) - (v[7] - v[0])) < 1e-12
    assert abs(phi['state'] - orders_check(v)) < 1e-12
    vb = {k: np.full(5, val) for k, val in v.items()}
    assert np.allclose(shapley_values(vb)['lon'], phi['lon'])


def test_shapley_shared_information_split():
    # state and lon carry the same information: each gets half, lat nothing
    v = {0: 0.0, 1: 0.1, 2: 0.1, 4: 0.0, 3: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    phi = shapley_values(v)
    assert abs(phi['state'] - 0.05) < 1e-12 and abs(phi['lon'] - 0.05) < 1e-12 and abs(phi['lat']) < 1e-12


def test_expected_slice_count_eq3():
    e = penetration.expected_slice_count(2.0, 25.0, 1500.0, 0.9)
    assert abs(e - 2.0 * (5 + 1.0) * 0.9) < 1e-12


def test_mu_required_and_penetration():
    mu = penetration.mu_required(6.0, theta=8.0, share=0.8)
    assert abs(penetration.prob_reaching(mu, 6.0, 8.0) - 0.8) < 1e-4
    assert penetration.mu_required(6.0, theta=np.inf) < mu       # Poisson needs less than an overdispersed sum
    p = penetration.required_penetration(mu, np.array([mu, 2 * mu]), 0.08)
    assert np.allclose(p, [0.08, 0.04])
    m = np.full(2000, 4.0)
    S = stats.nbinom.rvs(5.0, 5.0 / (5.0 + 12.0), size=2000, random_state=1)
    assert abs(penetration.nb_size_moments(S, m) - 5.0) < 1.0


def test_alarm_minutes_events_and_detection():
    tl = pd.DataFrame({'unit': 'u', 'minute': np.arange(0, 60), 'score': 0.0})
    tl.loc[tl['minute'].isin([10, 11, 12, 40, 41, 50]), 'score'] = 1.0
    a1 = alarms.alarm_minutes(tl, 0.5, k=1)
    a2 = alarms.alarm_minutes(tl, 0.5, k=2)
    assert list(a1['minute']) == [10, 11, 12, 40, 41, 50]
    assert list(a2['minute']) == [11, 12, 41]
    ev = alarms.events(a1, 15)
    assert len(ev) == 2 and list(ev['start']) == [10, 40]
    crashes = pd.DataFrame({'unit': ['u', 'u'], 'minute': [22, 44]})
    assert list(alarms.detected(a1, crashes, (15, 5))) == [True, False]
    assert list(alarms.detected(a1, crashes, (15, 1))) == [True, True]
    fa = alarms.false_alarm_rate(ev, pd.DataFrame({'unit': ['u'], 'minute': [5]}), km_days=10.0)
    assert abs(fa - 100.0 * 1 / 10.0) < 1e-12
    assert abs(alarms.detection_at_rate([10, 100], [0.2, 0.6], 31.6227766) - 0.4) < 1e-6


def test_inner_alarm_threshold_and_nested_fusion():
    rng = np.random.default_rng(0)
    n = 300
    groups = np.repeat(np.arange(60), 5)
    y = np.tile([1, 0, 0, 0, 0], 60)
    X = pd.DataFrame({'a': rng.normal(y, 1.0), 'b': rng.normal(0, 1, n)})
    samples = pd.DataFrame({'label': y, 'group_id': groups})
    th = alarms.inner_alarm_threshold(LogisticLearner(), X, samples, np.ones(n, bool), seed=1)
    assert np.isfinite(th)
    assert set(np.unique(inner_folds(groups, 4))) == {0, 1, 2, 3}
    nf = NestedFusion([('both', LogisticLearner(), ['a', 'b']), ('b only', LogisticLearner(), ['b'])])
    m = nf.fit(X, y, groups, 1)
    assert m.name == 'both' and len(m.predict(X)) == n


def test_threshold_on_rounded_grid_needs_no_extra_vehicle():
    exact = penetration.mu_required(14 / 3, theta=8.0)
    grid = penetration.mu_required(round(14 / 3, 6), theta=8.0)
    assert abs(exact - grid) < 1e-6
    assert penetration.prob_reaching(5.0, 4.666667, 8.0) == penetration.prob_reaching(5.0, 14 / 3, 8.0)
