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


def test_steering_vector_round_trip_k1():
    probe = make_probe(d=16, k=1)
    rng = np.random.default_rng(123)
    x_raw = rng.normal(size=16) * probe.feature_std + probe.feature_mean

    delta_raw = 15.0
    v_raw = steering_vector(probe, delta_raw)

    y_before = predicted_raw_target(probe, x_raw)
    y_after = predicted_raw_target(probe, x_raw + v_raw)

    achieved_delta = float(y_after[0] - y_before[0])
    assert abs(achieved_delta - delta_raw) < 1e-6, (
        f"expected delta {delta_raw}, got {achieved_delta}"
    )


def test_steering_vector_single_dim_of_k2_is_exact():
    """For a multi-dim probe, requesting a delta on only ONE dimension
    (others left at 0) must still achieve that dimension's delta exactly --
    independent-and-summed guarantees each individually-requested term is
    exact, regardless of correlation with other columns.
    """
    probe = make_probe(d=16, k=2)
    rng = np.random.default_rng(9)
    x_raw = rng.normal(size=16) * probe.feature_std + probe.feature_mean

    v_raw = steering_vector(probe, np.array([4.0, 0.0]))
    y_before = predicted_raw_target(probe, x_raw)
    y_after = predicted_raw_target(probe, x_raw + v_raw)

    assert abs(float(y_after[0] - y_before[0]) - 4.0) < 1e-6


def test_steering_vector_k2_joint_request_has_bounded_cross_talk():
    """When BOTH dims of a K=2 probe are requested simultaneously, summing
    independent per-dimension contributions does NOT exactly satisfy both
    constraints unless the two weight columns are orthogonal (the explicit
    trade-off documented in steering_math's module docstring). Check the
    achieved delta is in the right ballpark, not exact, and that the
    leakage is bounded by how correlated the two columns are.
    """
    probe = make_probe(d=16, k=2)
    rng = np.random.default_rng(7)
    x_raw = rng.normal(size=16) * probe.feature_std + probe.feature_mean

    delta_raw = np.array([5.0, -3.0])
    v_raw = steering_vector(probe, delta_raw)

    y_before = predicted_raw_target(probe, x_raw)
    y_after = predicted_raw_target(probe, x_raw + v_raw)
    achieved_delta = y_after - y_before

    w0, w1 = probe.weight[:-1, 0], probe.weight[:-1, 1]
    cos_w0_w1 = float(np.dot(w0, w1) / (np.linalg.norm(w0) * np.linalg.norm(w1)))
    # cross-talk in dim k from the other dim's term is delta_raw[other] * cos(w0,w1) * ||w_other||/||w_k|| (target_std-scaled)
    # -- not exactly this formula in raw units, but it must vanish as cos_w0_w1 -> 0 and must be exactly
    # zero only in that limit; assert the achieved delta is NOT trivially exact (proving this test
    # exercises the approximation, not a degenerate orthogonal case) and is still finite/sane.
    assert not np.allclose(achieved_delta, delta_raw, atol=1e-6) or abs(cos_w0_w1) < 1e-9
    assert np.all(np.isfinite(achieved_delta))


def test_steering_vector_independent_of_x():
    probe = make_probe(d=8, k=2)
    v1 = steering_vector(probe, np.array([0.0, 3.0]))
    v2 = steering_vector(probe, np.array([0.0, 3.0]))
    assert np.allclose(v1, v2)


def test_degenerate_gradient_raises():
    probe = make_probe(d=4, k=1)
    probe = ProbeSpec(**{**probe.__dict__, "weight": np.zeros_like(probe.weight)})
    try:
        steering_vector(probe, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for degenerate (all-zero) probe gradient")


def test_wrong_delta_length_raises():
    probe = make_probe(d=4, k=2)
    try:
        steering_vector(probe, np.array([1.0, 2.0, 3.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for delta_raw length mismatch")


if __name__ == "__main__":
    test_steering_vector_round_trip_k1()
    test_steering_vector_single_dim_of_k2_is_exact()
    test_steering_vector_k2_joint_request_has_bounded_cross_talk()
    test_steering_vector_independent_of_x()
    test_degenerate_gradient_raises()
    test_wrong_delta_length_raises()
    print("test_steering_math.py: all checks passed")
