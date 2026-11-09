import numpy as np
import pandas as pd

from probe_value.features import gantry_features, model_matrix, probe_columns, slice_features, vehicle_slice_stats


def _veh(sample, role, veh, t, speed, yaw=None, ay=None, ax=None, kappa=0.0):
    n = len(t)
    return pd.DataFrame({'sample_id': sample, 'role': role, 'vehicle': veh, 't': t, 'speed': speed,
                         'yaw_rate': np.zeros(n) if yaw is None else yaw, 'ay': np.zeros(n) if ay is None else ay,
                         'ax': np.zeros(n) if ax is None else ax, 'curvature': kappa})


def test_longitudinal_events_and_gap_rule():
    t = np.arange(-300, -280)
    v = np.full(20, 25.0)
    v[5:8] = [22.5, 20.0, 17.5]                       # one run of three harsh decelerations
    v[12] = 26.6                                     # one rapid acceleration (+1.6 m/s^2)
    r = _veh('s', 'T', 'a', t, v)
    st = vehicle_slice_stats(r)
    assert st['ndec'].sum() == 1
    assert st['nacc'].sum() == 2                     # 17.5 -> 25 after the braking run, and 25 -> 26.6
    # a 70 s gap is not differenced
    r2 = _veh('s', 'T', 'b', [-500, -499, -429, -428], [25, 25, 15, 15])
    st2 = vehicle_slice_stats(r2)
    assert st2['ndec'].sum() == 0


def test_vehicle_needs_three_records_and_window():
    a = _veh('s', 'T', 'a', [-850, -849, -848], [20, 20, 20])
    b = _veh('s', 'T', 'b', [-850, -849], [30, 30])
    c = _veh('s', 'T', 'c', [-905, -904, -903, 0, 1], [10] * 5)  # outside the window [-900, 0)
    st = vehicle_slice_stats(pd.concat([a, b, c]))
    f = slice_features(st, ['s'])
    assert f.loc['s', 'n_T0'] == 1
    assert f.loc['s', 'MSP_T0'] == 20
    assert np.isnan(f.loc['s', 'VSD_T0'])
    assert f.loc['s', 'n_T1'] == 0 and np.isnan(f.loc['s', 'MSP_T1'])
    assert abs(f.loc['s', 'n_bar'] - 1 / 3) < 1e-12


def test_state_channel_two_vehicles():
    a = _veh('s', 'U', 'a', np.arange(-100, -90), np.full(10, 20.0))
    b = _veh('s', 'U', 'b', np.arange(-100, -90), np.full(10, 24.0))
    f = slice_features(vehicle_slice_stats(pd.concat([a, b])), ['s'])
    assert f.loc['s', 'MSP_U2'] == 22.0
    assert abs(f.loc['s', 'VSD_U2'] - np.std([20, 24], ddof=1)) < 1e-12


def test_lateral_channel_curvature_correction_and_alr():
    t = np.arange(-200, -180)
    v = np.full(20, 25.0)
    kap = 1 / 1000.0
    yaw = np.degrees(v * kap).copy()
    ay = v ** 2 * kap * np.ones(20)
    yaw[5:7] += 7.0                                  # residual yaw >= 6 deg/s over two consecutive seconds
    yaw[12] += 7.0                                   # a single second does not count
    r = _veh('s', 'T', 'a', t, v, yaw=yaw, ay=ay, kappa=kap)
    st = vehicle_slice_stats(r)
    assert st['alr'].sum() == 1
    ax = np.zeros(20)
    ax[4:7] = 2.0                                    # strong longitudinal acceleration suppresses the manoeuvre
    st2 = vehicle_slice_stats(_veh('s', 'T', 'a', t, v, yaw=yaw, ay=ay, ax=ax, kappa=kap))
    assert st2['alr'].sum() == 0
    # straight-line records give zero residual yaw dispersion
    r3 = _veh('s', 'T', 'b', t, v, yaw=np.degrees(v * kap), ay=ay, kappa=kap)
    f = slice_features(vehicle_slice_stats(r3), ['s'])
    assert abs(f.loc['s', 'YRD_T2']) < 1e-9
    # records below 30 km/h are excluded from the lateral channel
    r4 = _veh('s', 'T', 'c', t, np.full(20, 5.0), yaw=np.full(20, 10.0))
    assert vehicle_slice_stats(r4)['lat_n'].sum() == 0


def test_rates_per_100_vehicle_km():
    t = np.arange(-600, -500)                        # 100 s at 20 m/s, one harsh deceleration
    v = np.full(100, 20.0)
    v[50] = 17.0
    v[51:] = 17.0
    f = slice_features(vehicle_slice_stats(_veh('s', 'T', 'a', t, v)), ['s'])
    km = vehicle_slice_stats(_veh('s', 'T', 'a', t, v))['dist'].sum() / 1000
    assert abs(f.loc['s', 'HDR_T1'] - 100 / km) < 1e-9


def test_gantry_features_and_matrix():
    g = pd.DataFrame({'sample_id': 's', 'position': ['up', 'up', 'down', 'down'], 't': [-60, -120, -60, -120],
                      'flow': [10.0, 30.0, 5.0, 5.0], 'speed': [20.0, 24.0, 25.0, 25.0], 'speed_sd': [1.0, 2.0, 1, 1],
                      'heavy': [0.3, 0.5, 0.2, 0.2], 'valid': [1, 1, 1, 1]})
    gf = gantry_features(g, ['s'])
    assert abs(gf.loc['s', 'G_SPD_up2'] - (10 * 20 + 30 * 24) / 40) < 1e-12
    assert abs(gf.loc['s', 'G_DIFF_2'] - (23.0 - 25.0)) < 1e-12
    assert np.isnan(gf.loc['s', 'G_SPD_in2'])
    pf = slice_features(vehicle_slice_stats(_veh('s', 'T', 'a', np.arange(-50, -40), np.full(10, 20.0))), ['s'])
    ctx = pd.DataFrame({'TOD': [8.0]}, index=pd.Index(['s'], name='sample_id'))
    assert list(model_matrix('C', None, None, ctx).columns) == ['TOD']
    assert len(model_matrix('P', pf, gf, ctx, ['lon']).columns) == len(probe_columns(['lon'])) + 1
    assert 'G_SPD_up0' in model_matrix('PG', pf, gf, ctx).columns
