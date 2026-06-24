"""Turn a trained linear probe into an additive embedding-space steering vector.

Derivation
----------
A probe predicts standardized targets from standardized features:

    x_std = (x - feature_mean) / feature_std
    y_std_pred = x_std @ w_k + bias_k          # w_k = W[:-1, k], single target dim k
    y_raw_pred = y_std_pred * target_std[k] + target_mean[k]

For a single target dim `k`, the raw-embedding perturbation `v_k` that
shifts the probe's raw prediction for dim `k` by exactly `delta_raw[k]`,
moving along the probe's own readout gradient, is:

    direction_raw = feature_std * w_k
    alpha = delta_raw[k] / ((w_k @ w_k) * target_std[k])
    v_k = alpha * direction_raw

For a multi-dim target (K>1, e.g. block_position), `steering_vector` sums
these per-dimension contributions independently: `v_raw = sum_k v_k`.

Why independent-and-summed rather than a joint pseudoinverse solve: solving
`activation_delta_std @ W[:-1,:] = target_delta_std` jointly (the textbook
minimum-norm solution via `pinv(W[:-1,:])`) exactly satisfies multi-dim
constraints (e.g. "move x, pin y to exactly zero predicted change"), and
is mathematically identical to the per-dimension formula above when K=1
(verified numerically: cosine sim 1.0, diff at float64 epsilon). But for
K>1 its failure mode is harder to protect against: it blows up when the
target dimensions' weight columns are *collinear* (e.g. block_x and
block_y's probe directions correlated in a small ViT-tiny embedding), not
just when an individual column is degenerate -- a small-singular-value
problem in the joint system that's easy to miss and amplifies noise.
Independent-and-summed only fails when one *specific* column's own
`||w_k||^2` is near zero, which is what `_DEGENERATE_GRADIENT_EPS` below
guards against directly. The cost: for K>1 it does NOT exactly pin
unrequested dimensions to zero (there's some cross-talk leakage
proportional to how correlated the columns are) -- check this for a given
probe with `predicted_raw_target` before/after applying v_raw if it
matters for your experiment.
"""

from __future__ import annotations

import numpy as np

from .probe_io import ProbeSpec


_DEGENERATE_GRADIENT_EPS = 1e-12


def steering_vector(probe: ProbeSpec, delta_raw: np.ndarray | float) -> np.ndarray:
    """Return v_raw (D,): the raw-embedding offset that shifts `probe`'s own
    raw prediction by `delta_raw` (K,), one target dim at a time,
    independently, then summed (see module docstring for why). For a 1-D
    target (e.g. block_angle) this is exact; for K>1 it's an approximation
    that trades exact joint-constraint satisfaction for robustness against
    collinear target dimensions.
    """
    delta_raw = np.atleast_1d(np.asarray(delta_raw, dtype=np.float64))
    probe_weights = probe.weight[:-1, :]  # (D, K)
    num_target_dims = probe_weights.shape[1]

    if delta_raw.shape[0] != num_target_dims:
        raise ValueError(
            f"delta_raw has {delta_raw.shape[0]} element(s) but probe "
            f"'{probe.probe_name}' has {num_target_dims} target dim(s)."
        )

    gradient_norm_sq = np.sum(probe_weights**2, axis=0)  # (K,), ||w_k||^2 per dim
    nonzero_dims = delta_raw != 0
    degenerate = nonzero_dims & (gradient_norm_sq < _DEGENERATE_GRADIENT_EPS)
    if np.any(degenerate):
        raise ValueError(
            f"Probe '{probe.probe_name}' has a near-zero readout gradient for "
            f"target dim(s) {np.nonzero(degenerate)[0].tolist()} with nonzero "
            "requested delta. This layer's probe doesn't predict that target "
            "dim well enough to steer along -- pick a different layer."
        )

    v_raw = np.zeros(probe_weights.shape[0])
    for k in np.nonzero(nonzero_dims)[0]:
        alpha = delta_raw[k] / (gradient_norm_sq[k] * probe.target_std[k])
        v_raw += alpha * probe.feature_std * probe_weights[:, k]
    return v_raw


def predicted_raw_target(probe: ProbeSpec, x_raw: np.ndarray) -> np.ndarray:
    """Forward pass of the probe: x_raw (..., D) -> y_raw (..., K).

    Used to sanity-check steering_vector's effect empirically: applying
    v_raw to x_raw and re-running this should shift y_raw by approximately
    the requested delta_raw (exactly, for K=1; with some cross-talk in
    unrequested dims for K>1 -- see module docstring).
    """
    x_std = (x_raw - probe.feature_mean) / probe.feature_std
    y_std_pred = x_std @ probe.weight[:-1, :] + probe.weight[-1, :]
    return y_std_pred * probe.target_std + probe.target_mean
