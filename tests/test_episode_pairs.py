"""Smoke test for interp_utils/steering/episode_pairs.py.

No torch, no model, no GPU. Run directly: python tests/test_episode_pairs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from interp_utils.steering.episode_pairs import find_episode_pairs


def make_synthetic_states(num_episodes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    episode_ids = np.arange(num_episodes)
    # 5-dim state: agent_x, agent_y, block_x, block_y, block_angle
    states = rng.uniform(low=-1.0, high=1.0, size=(num_episodes, 5))
    states[:, 4] = rng.uniform(low=-180.0, high=180.0, size=num_episodes)  # block_angle in degrees
    return episode_ids, states


def test_finds_pairs_close_to_target_delta():
    episode_ids, states = make_synthetic_states(500)
    delta_target = np.array([20.0])  # degrees
    tolerance = 2.0

    pairs = find_episode_pairs(
        episode_ids,
        states,
        target_columns=[4],
        delta_target=delta_target,
        tolerance=tolerance,
        max_pairs=20,
    )

    assert len(pairs) > 0, "expected to find at least one pair in 500 random episodes"
    for pair in pairs:
        assert abs(pair.achieved_delta[0] - delta_target[0]) <= tolerance + 1e-9
        assert pair.src_episode != pair.ref_episode
        assert pair.delta_error >= 0.0

    # sorted ascending by delta_error
    errors = [p.delta_error for p in pairs]
    assert errors == sorted(errors)


def test_match_columns_filters_confounds():
    episode_ids, states = make_synthetic_states(500, seed=1)
    delta_target = np.array([20.0])
    tolerance = 2.0

    unconstrained = find_episode_pairs(
        episode_ids, states, target_columns=[4], delta_target=delta_target,
        tolerance=tolerance, max_pairs=None,
    )
    constrained = find_episode_pairs(
        episode_ids, states, target_columns=[4], delta_target=delta_target,
        tolerance=tolerance, max_pairs=None,
        match_columns=[0, 1], match_tolerance=0.01,
    )

    assert len(constrained) <= len(unconstrained)
    for pair in constrained:
        src = pair.src_initial_state[[0, 1]]
        ref = pair.ref_initial_state[[0, 1]]
        assert np.all(np.abs(src - ref) <= 0.01 + 1e-9)


def test_no_pairs_for_tiny_dataset():
    episode_ids, states = make_synthetic_states(1)
    pairs = find_episode_pairs(
        episode_ids, states, target_columns=[4], delta_target=np.array([20.0]), tolerance=2.0,
    )
    assert pairs == []


def test_scales_reasonably():
    # PushT "expert" datasets are O(10^2-10^3) episodes; 2000 is a generous
    # margin above the real scale we expect, not a hard ceiling.
    episode_ids, states = make_synthetic_states(2000, seed=2)
    start = time.time()
    pairs = find_episode_pairs(
        episode_ids, states, target_columns=[4], delta_target=np.array([20.0]),
        tolerance=2.0, max_pairs=50,
    )
    elapsed = time.time() - start
    assert elapsed < 10.0, f"expected well under 10s for 2000 episodes, took {elapsed:.2f}s"
    assert len(pairs) > 0


if __name__ == "__main__":
    test_finds_pairs_close_to_target_delta()
    test_match_columns_filters_confounds()
    test_no_pairs_for_tiny_dataset()
    test_scales_reasonably()
    print("test_episode_pairs.py: all checks passed")
