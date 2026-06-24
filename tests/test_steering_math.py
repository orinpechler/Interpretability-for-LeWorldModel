"""Smoke test for interp_utils/steering/steering_math.py.

No torch, no model, no GPU. Run directly: python tests/test_steering_math.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from interp_utils.steering.probe_io import ProbeSpec
from interp_utils.steering.steering_math import predicted_raw_target, steering_vector


def make_probe(d: int = 16, k: int = 1, seed: int = 0) -> ProbeSpec:
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(d + 1, k))
    return ProbeSpec(
        target="block_angle",
        target_columns=[4],
        target_names=["block_angle"],
        probe_name="layer_07",
        layer_index=7,
        seed=0,
        weight=weight,
        feature_mean=rng.normal(size=d),
        feature_std=np.abs(rng.normal(size=d)) + 0.5,
        target_mean=rng.normal(size=k),
        target_std=np.abs(rng.normal(size=k)) + 0.5,
    )


def test_steering_vector_round_trip():
    probe = make_probe(d=16, k=1)
    rng = np.random.default_rng(123)
    x_raw = rng.normal(size=16) * probe.feature_std + probe.feature_mean

    delta_raw = 15.0
    v_raw = steering_vector(probe, target_dim_index=0, delta_raw=delta_raw)

    y_before = predicted_raw_target(probe, x_raw)
    y_after = predicted_raw_target(probe, x_raw + v_raw)

    achieved_delta = float(y_after[0] - y_before[0])
    assert abs(achieved_delta - delta_raw) < 1e-6, (
        f"expected delta {delta_raw}, got {achieved_delta}"
    )


def test_steering_vector_independent_of_x():
    probe = make_probe(d=8, k=2)
    v1 = steering_vector(probe, target_dim_index=1, delta_raw=3.0)
    v2 = steering_vector(probe, target_dim_index=1, delta_raw=3.0)
    assert np.allclose(v1, v2)


def test_degenerate_gradient_raises():
    probe = make_probe(d=4, k=1)
    probe = ProbeSpec(**{**probe.__dict__, "weight": np.zeros_like(probe.weight)})
    try:
        steering_vector(probe, target_dim_index=0, delta_raw=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for degenerate (all-zero) probe gradient")


def test_target_dim_out_of_range_raises():
    probe = make_probe(d=4, k=1)
    try:
        steering_vector(probe, target_dim_index=5, delta_raw=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range target_dim_index")


if __name__ == "__main__":
    test_steering_vector_round_trip()
    test_steering_vector_independent_of_x()
    test_degenerate_gradient_raises()
    test_target_dim_out_of_range_raises()
    print("test_steering_math.py: all checks passed")
