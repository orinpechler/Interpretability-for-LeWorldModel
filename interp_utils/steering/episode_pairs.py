"""Mine pairs of episodes whose initial states differ by a target delta.

Used to find a real "reference" trajectory for a steering experiment: given
a source episode and a desired delta (e.g. +15 degrees of block_angle), find
a different episode whose initial state actually differs from the source's
by approximately that delta, so its real future embeddings can serve as the
ground truth the steered model's predictions should move toward.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EpisodePair:
    src_episode: int
    ref_episode: int
    src_initial_state: np.ndarray  # (S,) full state vector
    ref_initial_state: np.ndarray  # (S,)
    achieved_delta: np.ndarray  # (K,) ref - src, restricted to target_columns
    delta_error: float  # ||achieved_delta - delta_target||


def load_initial_states(
    dataset_h5: h5py.File,
    num_episodes: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (episode_ids (E,), initial_state (E, S)).

    Uses `ep_offset` (one row index per episode, giving exactly where each
    episode's step_idx==0 frame sits in the flat per-frame arrays) to index
    directly into `state` -- NOT a scan over the full per-frame
    episode_idx/step_idx/state columns (those have 2M+ rows; ep_offset has
    one row per episode, e.g. ~18.7k for pusht_expert_train).

    If `num_episodes` is given and smaller than the total episode count,
    sample that many episodes uniformly at random (without replacement,
    `seed`-controlled) instead of every episode -- for fast iteration when
    you don't need the full episode set, e.g. interactive debugging.
    """
    ep_offset = np.asarray(dataset_h5["ep_offset"][:])
    total_episodes = len(ep_offset)
    episode_ids = np.arange(total_episodes)

    if num_episodes is not None and num_episodes < total_episodes:
        rng = np.random.default_rng(seed)
        episode_ids = rng.choice(total_episodes, size=num_episodes, replace=False)
        episode_ids.sort()

    rows = ep_offset[episode_ids]  # ep_offset is monotonic in episode_id, so rows stays sorted
    initial_state = np.asarray(dataset_h5["state"][rows])

    return episode_ids, initial_state


def find_episode_pairs(
    episode_ids: np.ndarray,
    initial_states: np.ndarray,
    target_columns: list[int],
    delta_target: np.ndarray,
    tolerance: float | np.ndarray,
    max_pairs: int | None = None,
    exclude_self: bool = True,
    match_columns: list[int] | None = None,
    match_tolerance: float | np.ndarray | None = None,
    max_neighbors: int = 50,
) -> list[EpisodePair]:
    """Find (src, ref) episode pairs whose initial_state[ref, target_columns]
    minus initial_state[src, target_columns] is within `tolerance` of
    `delta_target`.

    Uses a KD-tree nearest-neighbor query (O(E log E)) rather than an O(E^2)
    all-pairs scan: for each episode, query its target-subspace point shifted
    by delta_target, retrieve its `max_neighbors` nearest real episodes, then
    filter those by the actual per-dimension tolerance.

    `match_columns`/`match_tolerance` optionally require near-equality on
    OTHER state dimensions too (e.g. require agent position to roughly match
    when isolating a block_angle delta), to reduce confounds from mining
    real, non-independently-varying trajectories. Off by default.

    Returns pairs sorted by delta_error ascending (closest match first),
    truncated to `max_pairs` if given.
    """
    initial_states = np.asarray(initial_states, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    delta_target = np.atleast_1d(np.asarray(delta_target, dtype=np.float64))
    num_episodes = initial_states.shape[0]

    if num_episodes < 2:
        return []

    target_subspace = initial_states[:, target_columns]
    tol = np.broadcast_to(np.asarray(tolerance, dtype=np.float64), delta_target.shape)

    if match_columns:
        if match_tolerance is None:
            raise ValueError("match_tolerance is required when match_columns is set.")
        match_subspace = initial_states[:, match_columns]
        match_tol = np.broadcast_to(
            np.asarray(match_tolerance, dtype=np.float64), (len(match_columns),)
        )

    tree = cKDTree(target_subspace)
    query_points = target_subspace + delta_target
    k = min(max_neighbors, num_episodes)
    _, neighbor_idx = tree.query(query_points, k=k)
    if k == 1:
        neighbor_idx = neighbor_idx[:, None]

    pairs: list[EpisodePair] = []
    for src_idx in range(num_episodes):
        for ref_idx in neighbor_idx[src_idx]:
            ref_idx = int(ref_idx)
            if exclude_self and ref_idx == src_idx:
                continue

            achieved_delta = target_subspace[ref_idx] - target_subspace[src_idx]
            if np.any(np.abs(achieved_delta - delta_target) > tol):
                continue

            if match_columns:
                match_diff = np.abs(match_subspace[ref_idx] - match_subspace[src_idx])
                if np.any(match_diff > match_tol):
                    continue

            delta_error = float(np.linalg.norm(achieved_delta - delta_target))
            pairs.append(
                EpisodePair(
                    src_episode=int(episode_ids[src_idx]),
                    ref_episode=int(episode_ids[ref_idx]),
                    src_initial_state=initial_states[src_idx].copy(),
                    ref_initial_state=initial_states[ref_idx].copy(),
                    achieved_delta=achieved_delta,
                    delta_error=delta_error,
                )
            )

    pairs.sort(key=lambda p: p.delta_error)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs
