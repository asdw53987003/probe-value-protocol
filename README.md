# probe-value-protocol

Reference implementation of the **evaluation protocol** for probe vehicle data in real-time crash prediction
(Algorithm 1 of the paper *Incremental Value and Quality Constraints of Probe Vehicle Data for Real-Time Freeway Crash
Prediction: An Empirical Evaluation*).

The protocol compares models with and without probe data on the same crashes while the probe data are degraded in a
controlled way. It needs only delivered probe records, detector data and a crash sample with matched controls, and it
works with any learner that scores input slices:

1. **Degrade** the probe data of every sample in one dimension at a time, with the crash and its traffic state fixed:
   thin probe vehicles (nested, paired retention levels), delete delivered seconds in different arrangements, or delay
   the inputs.
2. **Retrain** the model groups Gantry-only (G), Probe-only (P) and Probe + Gantry (PG) at every level or scenario
   and score every sample with a model trained without its fold.
3. **Read the sufficiency threshold**: fit the fusion gain as a function of probe vehicles per slice to the placement
   values of all levels and replicates, form a simultaneous band by resampling crash dates, and take the smallest
   count whose lower band limit reaches a margin; convert it into the penetration each unit needs in each hour.
4. **Attribute** the gain at the reference, and the loss under degradation, to the state, longitudinal and lateral
   observation channels by retraining every channel subset (Shapley values).

## Data in this repository

The original probe, gantry, and crash data cannot be publicly released under data-sharing agreements. The datasets in
`examples/data/` are synthetically generated for demonstrating the evaluation protocol and are not the empirical data
used in the study.

## Installation

Python 3.9 or later.

```
pip install -r requirements.txt          # numpy, pandas, scipy, scikit-learn; lightgbm is optional
pip install -e .                         # optional
```

## Quick start

```
python examples/run_example.py           # logistic-regression learner, about half a minute
python examples/run_example.py --lightgbm
python -m pytest
```

The examples and tests put `src/` on the path themselves; for your own scripts install the package first
(`pip install -e .`).

```python
import pandas as pd
from probe_value import LogisticLearner, Scenario, run_protocol

records = pd.read_csv('records.csv')                  # delivered 1 Hz probe records around each decision time
gantry = pd.read_csv('gantry.csv')                    # gantry minute data of each sample
samples = pd.read_csv('samples.csv', index_col=0, parse_dates=['time', 'crash_time'])
context = ['TOD', 'NIGHT', 'DAYTYPE', 'RAIN', 'ROUTE', 'CURV', 'GRADE', 'TUNNEL', 'GCOV']

res = run_protocol(records, gantry, samples, context, lambda group: LogisticLearner(),
                   shares=(1, 1/2, 1/4, 1/8, 1/16, 1/32), replicates=50,
                   scenarios=[Scenario('random 20%', 'delete', dict(share=0.2, arrangement='random')),
                              Scenario('delay 2 min', 'delay', dict(minutes=2))],
                   attribution=('reference', 'share=0.25'), n_boot=1000)
print(res.summary())
```

## Algorithm 1 and where each line is implemented

| Line | What it does | Module / function |
|---|---|---|
| 1–2 | Degrade the probe records (thinning, deletion arrangements, delay) with fixed random numbers | `degrade.vehicle_uniforms`, `thin_mask`, `profile_share`, `delete_records`, `delay` |
| 3 | Slice features of every sample (state, longitudinal and lateral channels; gantry features) | `features.vehicle_slice_stats`, `slice_features`, `gantry_features`, `model_matrix` |
| 4–5 | Retrain G, P and PG on the training folds and score the test fold | `folds.date_block_folds`, `training_mask`, `out_of_fold`; learners in `learners` |
| 6 | Placement value of each crash within its coverage group | `inference.placement_values`, `auc`, `average_precision` |
| 7 | Fit Δc(n̄), Eq. (1) | `threshold.fit_logistic_ramp`, `sufficiency_threshold` |
| 8 | Resample crash dates; 90% simultaneous band | `inference.cluster_weights`, `crash_date_clusters`; `threshold.sufficiency_threshold` |
| 9 | Threshold n̄*c, Eq. (2) | `threshold.sufficiency_threshold` |
| 10–12 | Retrain the eight channel subsets (reference or mixed degraded inputs); Shapley values | `protocol.run_protocol` (`attribution`), `shapley.shapley_values` |
| 13 | Required penetration of each unit and hour, Eqs. (3)–(4) | `penetration.expected_slice_count`, `mu_required`, `required_penetration`, `requirement_table` |

`protocol.run_protocol` runs lines 1–12 end to end. Further pieces of the implementation section of the paper:
`alarms.inner_alarm_threshold` (alarm thresholds set from training data only, by inner cross-fitting over matched
groups), `alarms.alarm_minutes`, `events`, `detected`, `false_alarm_rate` and `detection_at_rate` (event-level
warning), `inference.verdict` (improvement / equivalent / deterioration / inconclusive against a margin), and
`learners.NestedFusion` with `lightgbm_fusion_candidates` (the nested selection of Probe + Gantry). The bootstrap
resamples crash dates by default; `run_protocol(..., cluster='group')` resamples matched groups instead.

## Input format

Records are expected to be assigned to units already and timed relative to each decision.

`records` (one row per delivered 1 Hz record): `sample_id`, `role` (`T` target unit, `U` adjacent upstream unit),
`vehicle`, `t` (s relative to the decision time, negative before it), `speed` (m/s), `yaw_rate` (°/s), `ay` and `ax`
(m/s², IMU lateral and longitudinal), `curvature` (1/m, design curvature at the record's chainage, signed so that
speed × curvature is the yaw rate the curve requires, in the sign convention of `yaw_rate`); `tunnel` (0/1) is needed
for the state-dependent deletion. Records outside [−900, 0) s are used only by the delay scenarios.

`gantry` (one row per sample, gantry position and minute): `sample_id`, `position` (`up`, `in`, `down`), `t` (s,
minute start relative to the decision time), `flow` (veh/min), `speed` (m/s), `speed_sd` (m/s), `heavy` (share),
`valid` (0/1).

`samples` (indexed by `sample_id`): `group_id` (matched group), `label` (1 crash, 0 control), `time` (decision time),
`crash_time` (time of the group's crash), `unit`, `coverage` (1 if a same-direction gantry lies in the precursor zone of
the crash; controls take the value of their crash), optionally `night`, and the context feature columns.

## Definitions used

- Each decision uses three 5-min input slices that end at the decision time; a probe vehicle counts in a slice if it
  has at least three delivered records in the unit during the slice; n̄ is the mean count over the three slices of the
  target unit.
- Accelerations are differenced from GNSS speed between consecutive delivered records across gaps of up to 60 s;
  harsh decelerations are ≤ −2.0 m/s², rapid accelerations ≥ 1.5 m/s², consecutive exceedances counting once.
- The lateral channel uses IMU yaw rate and lateral acceleration corrected for road curvature at the current speed,
  with records below 30 km/h excluded; an abrupt lateral manoeuvre is a residual yaw rate of at least 6°/s over two or
  more consecutive delivered seconds while the 3-s mean longitudinal acceleration stays within ±0.56 m/s².
- Missing values: VSD, D95 and Y95 need at least two vehicles in the slice, YRD at least three records; empty slices
  give missing features, which the learner must handle. Exposure (vehicle-km) weights each record by half the gaps to
  its neighbours, each capped at 30 s.
- Thinning keeps a vehicle if its uniform random number falls below the retention share; the same numbers are used at
  every level of a replicate.
- Folds are consecutive blocks of crash dates; matched groups stay together; training samples within 7 days of a test
  crash, and samples on the same unit with an overlapping input window, are removed from training.
- The curves are evaluated over the observed count range, from the 5th to the 75th percentile of the reference counts of
  the crashes in the group; intervals are conditional on the fitted models.

## Conditions and limits

- The fusion-gain curve is the gain expected over training replicates; a threshold read from it refers to models
  retrained on data of a given quality, not to one deployed model.
- Shapley values divide the gain in AUC among the channels; they are not effects of the recorded behaviour on crash
  occurrence.
- Eq. (4) assumes that the probe fleet would be scaled in proportion.
- The example settings are small so that the run is short; the paper uses many more replicates and bootstrap draws.
  Library versions can move the last digits of `examples/example_output.txt`; the test allows for this.

## Repository layout

```
src/probe_value/   features, degradation, folds and scoring, learners, inference, thresholds, Shapley, penetration,
                   alarms, protocol (Algorithm 1)
examples/          example data and an end-to-end run with expected output
tests/             unit tests (pytest)
```

## Licence

Released under the MIT License (`LICENSE`).
