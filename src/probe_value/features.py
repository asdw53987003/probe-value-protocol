# -*- coding: utf-8 -*-
"""Slice features of the three probe observation channels and of the gantry detectors (Section IV-C, Table II).

Probe records (one row per delivered 1 Hz record) need the columns
    sample_id, role ('T' target unit, 'U' adjacent upstream unit), vehicle, t (s, integer, relative to the decision
    time; negative before it), speed (m/s), yaw_rate (deg/s, IMU), ay (m/s^2, IMU lateral), ax (m/s^2, IMU
    longitudinal), curvature (1/m, design curvature of the centreline at the record's chainage).
A decision uses three 5-min input slices that end at the decision time (t in [-900, 0)).

A probe vehicle counts in a slice if it has at least three delivered records in the unit during the slice; only such
vehicles enter the features.

State channel         MSP mean of vehicle mean speeds; VSD between-vehicle SD of mean speed (>= 2 vehicles).
Longitudinal channel  accelerations differenced from GNSS speed between consecutive delivered records, across gaps of
                      up to 60 s: ASD time-weighted SD of acceleration; HDR harsh decelerations (<= -2.0 m/s^2) and
                      HAR rapid accelerations (>= 1.5 m/s^2) per 100 vehicle-km, consecutive exceedances counting
                      once; D95 95th percentile across vehicles of the peak deceleration (>= 2 vehicles).
Lateral channel       IMU records corrected for road curvature, speed above 30 km/h: residual yaw rate
                      yaw - v kappa (deg/s) and residual lateral acceleration ay - v^2 kappa. YRD SD of the residual yaw
                      rate over the records of the slice (>= 3 records); ALR abrupt lateral manoeuvres per 100
                      vehicle-km, a manoeuvre being |residual yaw| >= 6 deg/s over two or more consecutive delivered
                      seconds while the 3-s mean of the measured longitudinal acceleration stays within +-0.56 m/s^2;
                      Y95 95th percentile across vehicles of the peak |residual lateral acceleration| (>= 2 vehicles).
Gantry features       per slice and position (nearest upstream, inside, nearest downstream): flow-weighted speed,
                      mean within-minute speed SD, mean flow, mean heavy-vehicle share over valid minutes, and the
                      upstream-minus-downstream speed difference.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SLICE_S = 300
N_SLICES = 3
MIN_RECORDS = 3
MAX_DIFF_GAP = 60.0
HARSH_DEC = -2.0
RAPID_ACC = 1.5
V_MIN_LAT = 30.0 / 3.6
ALR_YAW = 6.0
ALR_AX = 0.56
ROLES = ('T', 'U')

CHANNELS: Dict[str, tuple] = {'state': ('MSP', 'VSD'),
                              'lon': ('ASD', 'HDR', 'HAR', 'D95'),
                              'lat': ('YRD', 'ALR', 'Y95')}
PROBE_FEATURES = tuple(f for ch in CHANNELS.values() for f in ch)
GANTRY_STATS = ('SPD', 'SSD', 'FLW', 'HVS')
GANTRY_POSITIONS = ('up', 'in', 'down')
RECORD_COLUMNS = ('sample_id', 'role', 'vehicle', 't', 'speed', 'yaw_rate', 'ay', 'ax', 'curvature')


def _window(records: pd.DataFrame, window_end: float) -> pd.DataFrame:
    lo = window_end - N_SLICES * SLICE_S
    return records[(records['t'] >= lo) & (records['t'] < window_end)]


def vehicle_slice_stats(records: pd.DataFrame, window_end: float = 0.0) -> pd.DataFrame:
    """Sufficient statistics per (sample, role, slice, vehicle) from the delivered records of the input window.

    Columns: npts, sv (sum of speeds), dist (m, exposure from half-gap time weights capped at 30 s), sdt, sa, saa
    (time-weighted sums of acceleration), amin, ndec, nacc, lat_n, lat_sw, lat_sww (residual yaw), alr, aymax."""
    missing = set(RECORD_COLUMNS) - set(records.columns)
    if missing:
        raise ValueError('records lack columns %s' % sorted(missing))
    r = _window(records, window_end).sort_values(['sample_id', 'role', 'vehicle', 't'], kind='stable')
    if r.empty:
        return pd.DataFrame(columns=['sample_id', 'role', 'slice', 'vehicle'])
    sid = r['sample_id'].to_numpy()
    role = r['role'].to_numpy()
    veh = r['vehicle'].to_numpy()
    t = r['t'].to_numpy(float)
    v = r['speed'].to_numpy(float)
    key = pd.factorize(pd.MultiIndex.from_arrays([sid, role, veh]))[0]
    same = np.r_[False, key[1:] == key[:-1]]
    nxt = np.r_[same[1:], False]
    dt = np.where(same, t - np.r_[np.nan, t[:-1]], np.nan)
    pair = same & (dt > 0) & (dt <= MAX_DIFF_GAP)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = np.where(pair, (v - np.r_[np.nan, v[:-1]]) / dt, np.nan)
    dtn = np.r_[dt[1:], np.nan]
    wp = np.where(same, np.minimum(np.nan_to_num(dt, nan=0.0), MAX_DIFF_GAP) / 2.0, 0.5)
    wn = np.where(nxt, np.minimum(np.nan_to_num(dtn, nan=0.0), MAX_DIFF_GAP) / 2.0, 0.5)
    dist = v * (wp + wn)
    isdec = pair & (a <= HARSH_DEC)
    isacc = pair & (a >= RAPID_ACC)
    prev_dec = np.r_[False, isdec[:-1]] & same
    prev_acc = np.r_[False, isacc[:-1]] & same
    sl = ((t - (window_end - N_SLICES * SLICE_S)) // SLICE_S).astype(int)
    # lateral channel
    kap = r['curvature'].to_numpy(float)
    fast = v > V_MIN_LAT
    wres = r['yaw_rate'].to_numpy(float) - np.degrees(v * kap)
    ayres = r['ay'].to_numpy(float) - v ** 2 * kap
    ax = r['ax'].to_numpy(float)
    ax1 = np.r_[np.nan, ax[:-1]]
    ax2 = np.r_[np.nan, np.nan, ax[:-2]]
    t2 = np.r_[np.nan, np.nan, t[:-2]]
    same2 = np.r_[False, same[:-1]] & same
    use1 = same & (dt <= 2)
    use2 = same2 & ((t - t2) <= 2)
    num = ax + np.where(use1, ax1, 0.0) + np.where(use2, ax2, 0.0)
    den = 1.0 + use1 + use2
    ax3 = num / den
    cond = fast & (np.abs(wres) >= ALR_YAW) & (np.abs(ax3) <= ALR_AX)
    cont = same & (dt == 1)
    c1 = np.r_[False, cond[:-1]]
    c2 = np.r_[False, False, cond[:-2]]
    cont1 = np.r_[False, cont[:-1]]
    alr = cond & c1 & cont & ~(c2 & cont1)
    df = pd.DataFrame({'sample_id': sid, 'role': role, 'slice': sl, 'vehicle': veh, 'npts': 1.0, 'sv': v,
                       'dist': dist, 'sdt': np.where(pair, dt, 0.0), 'sa': np.where(pair, a * dt, 0.0),
                       'saa': np.where(pair, a * a * dt, 0.0), 'amin': np.where(pair, a, np.inf),
                       'ndec': (isdec & ~prev_dec).astype(float), 'nacc': (isacc & ~prev_acc).astype(float),
                       'lat_n': fast.astype(float), 'lat_sw': np.where(fast, wres, 0.0),
                       'lat_sww': np.where(fast, wres ** 2, 0.0), 'alr': alr.astype(float),
                       'aymax': np.where(fast, np.abs(ayres), -np.inf)})
    agg = {c: 'sum' for c in ('npts', 'sv', 'dist', 'sdt', 'sa', 'saa', 'ndec', 'nacc', 'lat_n', 'lat_sw', 'lat_sww',
                              'alr')}
    agg.update(amin='min', aymax='max')
    out = df.groupby(['sample_id', 'role', 'slice', 'vehicle'], sort=False).agg(agg).reset_index()
    out.loc[~np.isfinite(out['amin']), 'amin'] = np.nan
    out.loc[~np.isfinite(out['aymax']), 'aymax'] = np.nan
    return out


def _p95(x: pd.Series) -> float:
    x = x.dropna()
    return float(np.quantile(x, 0.95)) if len(x) >= 2 else np.nan


def slice_features(vstats: pd.DataFrame, samples: Iterable, keep: Optional[np.ndarray] = None,
                   min_records: int = MIN_RECORDS) -> pd.DataFrame:
    """Features of every sample x role x slice from vehicle-slice statistics, wide format.

    `keep` (boolean, aligned with `vstats`) removes vehicles, e.g. by thinning. Returns one row per sample with
    columns '<feature>_<role><slice>' and 'n_<role><slice>' (probe vehicles per slice), plus 'n_bar', the mean count
    over the three slices of the target unit. Empty slices give missing features."""
    s = vstats if keep is None else vstats[np.asarray(keep, bool)]
    s = s[s['npts'] >= min_records].copy()
    s['vm'] = s['sv'] / s['npts']
    s['peakdec'] = -s['amin']
    g = s.groupby(['sample_id', 'role', 'slice'])
    t = pd.DataFrame({'n': g.size(), 'MSP': g['vm'].mean(), 'VSD': g['vm'].std(ddof=1)})
    sums = g[['dist', 'sdt', 'sa', 'saa', 'ndec', 'nacc', 'lat_n', 'lat_sw', 'lat_sww', 'alr']].sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = sums['sa'] / sums['sdt']
        t['ASD'] = np.sqrt(np.maximum(sums['saa'] / sums['sdt'] - mu ** 2, 0.0)).where(sums['sdt'] > 0)
        km = sums['dist'] / 1000.0
        t['HDR'] = (100.0 * sums['ndec'] / km).where(km > 0)
        t['HAR'] = (100.0 * sums['nacc'] / km).where(km > 0)
        mw = sums['lat_sw'] / sums['lat_n']
        t['YRD'] = np.sqrt(np.maximum(sums['lat_sww'] / sums['lat_n'] - mw ** 2, 0.0)).where(sums['lat_n'] >= 3)
        t['ALR'] = (100.0 * sums['alr'] / km).where(km > 0)
    t['D95'] = g['peakdec'].apply(_p95)
    t['Y95'] = g['aymax'].apply(_p95)
    t.loc[t['n'] < 2, 'VSD'] = np.nan
    t = t.reset_index()
    t['col'] = t['role'].astype(str) + t['slice'].astype(str)
    wide = t.pivot(index='sample_id', columns='col', values=['n'] + list(PROBE_FEATURES))
    wide.columns = ['%s_%s' % (f, c) for f, c in wide.columns]
    cols = ['%s_%s%d' % (f, ro, k) for f in ['n'] + list(PROBE_FEATURES) for ro in ROLES for k in range(N_SLICES)]
    wide = wide.reindex(index=pd.Index(list(samples), name='sample_id'), columns=cols)
    ncols = [c for c in cols if c.startswith('n_')]
    wide[ncols] = wide[ncols].fillna(0.0)
    wide['n_bar'] = wide[['n_T%d' % k for k in range(N_SLICES)]].mean(axis=1)
    return wide


def gantry_features(gantry: pd.DataFrame, samples: Iterable, window_end: float = 0.0) -> pd.DataFrame:
    """Gantry features per sample from minute rows with columns sample_id, position ('up', 'in', 'down'), t (s,
    minute start relative to the decision time), flow (veh/min), speed (m/s), speed_sd (m/s), heavy (share) and
    valid (bool). Returns columns 'G_<stat>_<position><slice>' and 'G_DIFF_<slice>' (upstream minus downstream)."""
    g = _window(gantry, window_end)
    g = g[g['valid'].astype(bool)].copy()
    g['slice'] = ((g['t'] - (window_end - N_SLICES * SLICE_S)) // SLICE_S).astype(int)
    g['fv'] = g['flow'] * g['speed']
    a = g.groupby(['sample_id', 'position', 'slice']).agg(fv=('fv', 'sum'), f=('flow', 'sum'), SSD=('speed_sd', 'mean'),
                                                          FLW=('flow', 'mean'), HVS=('heavy', 'mean'))
    with np.errstate(invalid='ignore', divide='ignore'):
        a['SPD'] = (a['fv'] / a['f']).where(a['f'] > 0)
    a = a[list(GANTRY_STATS)].reset_index()
    a['col'] = a['position'].astype(str) + a['slice'].astype(str)
    wide = a.pivot(index='sample_id', columns='col', values=list(GANTRY_STATS))
    wide.columns = ['G_%s_%s' % (s, c) for s, c in wide.columns]
    cols = ['G_%s_%s%d' % (s, p, k) for s in GANTRY_STATS for p in GANTRY_POSITIONS for k in range(N_SLICES)]
    wide = wide.reindex(index=pd.Index(list(samples), name='sample_id'), columns=cols)
    for k in range(N_SLICES):
        wide['G_DIFF_%d' % k] = wide['G_SPD_up%d' % k] - wide['G_SPD_down%d' % k]
    return wide


def probe_columns(channels: Sequence[str] = ('state', 'lon', 'lat')) -> list:
    return ['%s_%s%d' % (f, ro, k) for ch in channels for f in CHANNELS[ch] for ro in ROLES for k in range(N_SLICES)]


def gantry_columns(gfeat: pd.DataFrame) -> list:
    return [c for c in gfeat.columns if c.startswith('G_')]


def model_matrix(group: str, probe: Optional[pd.DataFrame], gantry: Optional[pd.DataFrame], context: pd.DataFrame,
                 channels: Sequence[str] = ('state', 'lon', 'lat')) -> pd.DataFrame:
    """Feature matrix of a model group: 'C' context only, 'G' gantry, 'P' probe (the given channels), 'PG' both.
    Context features are shared by all groups; probe counts and empty-slice indicators are not included."""
    parts = []
    if group in ('P', 'PG') and channels:
        parts.append(probe[probe_columns(channels)])
    if group in ('G', 'PG'):
        parts.append(gantry[gantry_columns(gantry)])
    parts.append(context)
    return pd.concat([p.reindex(context.index) for p in parts], axis=1)
