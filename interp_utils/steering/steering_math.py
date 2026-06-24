"""Turn a trained linear probe into an additive embedding-space steering vector.

Derivation
----------
A probe predicts standardized targets from standardized features:

    x_std = (x - feature_mean) / feature_std
    y_std_pred = x_std @ w_k + bias_k          # w_k = W[:-1, k], single target dim k
    y_raw_pred = y_std_pred * target_std[k] + target_mean[k]

We want a raw-embedding perturbation `v_raw` such that `x + v_raw` shifts the
probe's own raw prediction for dim `k` by exactly `delta_raw`, moving along
the probe's own readout gradient (the direction of steepest change in
y_std_pred per unit of raw-space movement is `feature_std * w_k`, since
x_std's gradient w.r.t. x is `1 / feature_std`):

    direction_raw = feature_std * w_k
    x' = x + alpha * direction_raw
    x'_std = x_std + alpha * w_k
    y_std_pred' = y_std_pred + alpha * (w_k @ w_k)
    y_raw_pred' = y_raw_pred + alpha * (w_k @ w_k) * target_std[k]

Solving y_raw_pred' - y_raw_pred == delta_raw for alpha:

    alpha = delta_raw / ((w_k @ w_k) * target_std[k])
    v_raw = alpha * direction_raw

This is independent of x (the probe is linear), so v_raw can be precomputed
once per (probe, target_dim_index, delta_raw) and added to any embedding at
that probe's layer.
"""

from __future__ import annotations

import numpy as np

from .probe_io import ProbeSpec


_DEGENERATE_GRADIENT_EPS = 1e-12


def steering_vector(probe: ProbeSpec, target_dim_index: int, delta_raw: float) -> np.ndarray:
    """Return v_raw (D,): the raw-embedding offset that shifts `probe`'s own
    raw prediction for target dim `target_dim_index` by exactly `delta_raw`.
    """
    num_target_dims = probe.weight.shape[1]
    if not 0 <= target_dim_index < num_target_dims:
        raise ValueError(
            f"target_dim_index {target_dim_index} out of range for probe "
            f"'{probe.probe_name}' with {num_target_dims} target dim(s)."
        )

    w_k = probe.weight[:-1, target_dim_index]
    gradient_norm_sq = float(w_k @ w_k)
    if gradient_norm_sq < _DEGENERATE_GRADIENT_EPS:
        raise ValueError(
            f"Probe '{probe.probe_name}' has a near-zero readout gradient "
            f"for target dim {target_dim_index} (||w_k||^2={gradient_norm_sq:.3g}). "
            "This layer's probe doesn't predict this target well enough to "
            "steer along -- pick a different layer."
        )

    direction_raw = probe.feature_std * w_k
    alpha = delta_raw / (gradient_norm_sq * probe.target_std[target_dim_index])
    return alpha * direction_raw


def predicted_raw_target(probe: ProbeSpec, x_raw: np.ndarray) -> np.ndarray:
    """Forward pass of the probe: x_raw (..., D) -> y_raw (..., K).

    Used to sanity-check steering_vector's effect empirically: applying
    v_raw to x_raw and re-running this should shift y_raw[..., k] by
    exactly the requested delta_raw.
    """
    x_std = (x_raw - probe.feature_mean) / probe.feature_std
    y_std_pred = x_std @ probe.weight[:-1, :] + probe.weight[-1, :]
    return y_std_pred * probe.target_std + probe.target_mean
