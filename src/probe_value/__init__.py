# -*- coding: utf-8 -*-
"""probe_value: a learner-independent protocol for evaluating probe vehicle data in real-time crash prediction.

Reference implementation of Algorithm 1 of "Incremental Value and Quality Constraints of Probe Vehicle Data for
Real-Time Freeway Crash Prediction: An Empirical Evaluation".
"""
from . import alarms, degrade, features, folds, inference, learners, penetration, shapley, threshold
from .features import gantry_features, model_matrix, slice_features, vehicle_slice_stats
from .folds import date_block_folds, out_of_fold, training_mask
from .inference import auc, average_precision, cluster_weights, placement_values, summarize, verdict
from .learners import LightGBMLearner, LogisticLearner, NestedFusion, lightgbm_settings
from .protocol import ProtocolResult, Scenario, run_protocol
from .shapley import shapley_values
from .threshold import ThresholdResult, sufficiency_threshold

__version__ = '0.1.0'
