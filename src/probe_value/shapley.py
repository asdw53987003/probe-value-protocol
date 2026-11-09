# -*- coding: utf-8 -*-
"""Step 4 of the protocol: attribution of the gain (or of the loss under degradation) to observation channels
(Section IV-F).

The value of a channel is its Shapley value, its gain in AUC averaged over the orders in which the three probe
channels can be added, with each of the eight channel subsets retrained on its own features. Because every subset is
retrained, information that two channels share is divided between them. The values divide the gain in AUC among the
channels; they are not effects of the recorded behaviour on crash occurrence.

Subsets are coded as bit masks: state = 1, longitudinal = 2, lateral = 4 (S = 0 is the base).
"""
from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import Dict, Sequence

import numpy as np

CHANNEL_BITS = {'state': 1, 'lon': 2, 'lat': 4}


def subset_channels(S: int, bits: Dict[str, int] = CHANNEL_BITS) -> list:
    return [ch for ch, b in bits.items() if S & b]


def shapley_weights(n_players: int = 3) -> Dict[int, float]:
    """Weight of a coalition of size k in the Shapley value: k! (n - k - 1)! / n!."""
    return {k: factorial(k) * factorial(n_players - k - 1) / factorial(n_players) for k in range(n_players)}


def shapley_values(v: Dict[int, np.ndarray], bits: Dict[str, int] = CHANNEL_BITS) -> Dict[str, np.ndarray]:
    """Shapley value of each channel from coalition values v[S] (scalars or arrays, e.g. bootstrap vectors);
    v[0] is the value of the base. The values add up to v[full] - v[0]."""
    n = len(bits)
    w = shapley_weights(n)
    full = sum(bits.values())
    out = {}
    for ch, b in bits.items():
        tot = 0.0
        for S in range(full + 1):
            if S & b:
                continue
            k = bin(S).count('1')
            tot = tot + w[k] * (np.asarray(v[S | b]) - np.asarray(v[S]))
        out[ch] = tot
    return out


def mixed_inputs(reference: dict, degraded: dict, S: int, channel_columns: Dict[str, Sequence[str]]) -> dict:
    """Feature table in which the channels of S take their degraded columns and the others their reference columns
    (used to divide a loss under degradation; every mixed input is retrained)."""
    out = dict(reference)
    for ch, b in CHANNEL_BITS.items():
        if S & b:
            for col in channel_columns[ch]:
                out[col] = degraded[col]
    return out


def all_subsets(bits: Dict[str, int] = CHANNEL_BITS) -> list:
    full = sum(bits.values())
    return list(range(full + 1))


def orders_check(v: Dict[int, float], bits: Dict[str, int] = CHANNEL_BITS) -> float:
    """Shapley value of the first channel computed by enumerating orders (for testing the weights)."""
    names = list(bits)
    ch = names[0]
    vals = []
    for perm in permutations(names):
        S = 0
        for p in perm:
            if p == ch:
                vals.append(v[S | bits[ch]] - v[S])
                break
            S |= bits[p]
    return float(np.mean(vals))


__all__ = ['CHANNEL_BITS', 'subset_channels', 'shapley_weights', 'shapley_values', 'mixed_inputs', 'all_subsets',
           'orders_check']
