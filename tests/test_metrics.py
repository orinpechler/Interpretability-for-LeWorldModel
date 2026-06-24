"""Smoke test for interp_utils/steering/metrics.py.

No torch, no model, no GPU. Run directly: python tests/test_metrics.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from interp_utils.steering.metrics import (
    cosine_similarity,
    magnitude_diff,
    mae,
    mse,
    summarize_trajectory,
    write_metrics_table,
)


def test_known_offset():
    ref = np.zeros((3, 4))
    pred = ref + 2.0  # constant offset of 2 in every dim

    assert np.allclose(mse(pred, ref), 4.0)
    assert np.allclose(mae(pred, ref), 2.0)
    assert np.allclose(magnitude_diff(pred, ref), np.linalg.norm(pred, axis=-1))


def test_cosine_similarity_identical_vectors():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 8))
    cos = cosine_similarity(v, v)
    assert np.allclose(cos, 1.0, atol=1e-6)


def test_cosine_similarity_orthogonal():
    pred = np.array([[1.0, 0.0]])
    ref = np.array([[0.0, 1.0]])
    cos = cosine_similarity(pred, ref)
    assert np.allclose(cos, 0.0, atol=1e-9)


def test_cosine_similarity_zero_vector_is_nan():
    pred = np.array([[0.0, 0.0]])
    ref = np.array([[1.0, 0.0]])
    cos = cosine_similarity(pred, ref)
    assert np.isnan(cos[0])


def test_summarize_trajectory_t1_edge_case():
    pred = np.array([[1.0, 2.0, 3.0]])
    ref = np.array([[1.0, 2.0, 4.0]])
    result = summarize_trajectory(pred, ref)
    assert len(result.mse_t) == 1
    assert result.mse_mean == result.mse_last == result.mse_t[0]


def test_summarize_trajectory_matches_hand_computed():
    # mse/mae average the squared/abs error over D=2, so a [2,0]-vs-[1,0]
    # row contributes ((2-1)^2 + 0^2)/2 = 0.5, not 1.0. pred/ref chosen
    # parallel (not zero) so cosine similarity is well-defined (1.0).
    pred = np.array([[2.0, 0.0], [0.0, 3.0]])
    ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = summarize_trajectory(pred, ref)
    assert result.mse_t == [0.5, 2.0]
    assert result.mae_t == [0.5, 1.0]
    assert result.mse_mean == 1.25
    assert result.mse_last == 2.0
    assert np.allclose(result.cosine_sim_t, [1.0, 1.0])
    assert result.magnitude_diff_t == [1.0, 2.0]


def test_shape_mismatch_raises():
    try:
        summarize_trajectory(np.zeros((2, 3)), np.zeros((2, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_write_metrics_table_round_trip():
    pred = np.array([[2.0, 0.0], [0.0, 3.0]])
    ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics = summarize_trajectory(pred, ref)
    rows = [{"run_id": "test_run", **asdict(metrics)}]

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        write_metrics_table(output_dir, rows)
        assert (output_dir / "metrics_by_run.json").exists()
        assert (output_dir / "metrics_by_run.csv").exists()


if __name__ == "__main__":
    test_known_offset()
    test_cosine_similarity_identical_vectors()
    test_cosine_similarity_orthogonal()
    test_cosine_similarity_zero_vector_is_nan()
    test_summarize_trajectory_t1_edge_case()
    test_summarize_trajectory_matches_hand_computed()
    test_shape_mismatch_raises()
    test_write_metrics_table_round_trip()
    print("test_metrics.py: all checks passed")
