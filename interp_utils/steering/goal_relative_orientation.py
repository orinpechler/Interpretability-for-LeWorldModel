#!/usr/bin/env python
"""Does the embedding encode which way the block still needs to rotate to
reach its actual task goal?

Every other steering/probing script in this package uses SYNTHETIC,
single-variable perturbations (render a frame with only the target dim
changed). This script instead uses REAL task-relevant pairs: `eval.py`
confirms the env's goal is always the SAME episode's frame `goal_offset`
steps ahead (`goal_offset_steps: 25` in every config/eval/*.yaml) -- exactly
what `world.evaluate(..., goal_offset=...)` uses for the actual PushT
task. So `delta_h_real = h(goal) - h(current)` is the real, multi-variable
embedding change the model would actually need to "see" coming, not an
isolated synthetic one -- noisier (position/velocity/agent state change
too, not just orientation), but task-grounded.

Question: does sign(delta_angle) -- does the goal require clockwise or
counterclockwise rotation from here -- classify from delta_h_real?

I/O: same chunk-aligned block-sampling trick as orientation_classification.py
(encoder_cls_layers chunks are (1024, 12, 192)). Since goal_offset=25 is
tiny relative to the 1024-row chunk size, (s, s+offset) pairs land in the
same chunk -- cheap. Episode boundaries within a block are respected by
filtering on episode_idx (average PushT episode is ~125 frames, so each
1024-row block crosses several episode boundaries).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from interp_utils.steering.classifier_direction import evaluate_direction_agreement


def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference a - b, wrapped to (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def sample_goal_pairs(
    dataset_path: Path,
    embeddings_path: Path,
    layer_index: int | None,
    starts: np.ndarray,
    block_size: int,
    goal_offset: int,
    concat_layers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (delta_angles, delta_hs, h_currents) -- h_currents lets a
    caller test whether the CURRENT frame alone (no goal frame needed)
    already predicts which way the block will need to rotate, e.g. because
    PushT typically has a fixed target pose, so "which way to goal" can be
    a near-deterministic function of the current angle alone.

    If concat_layers is True, ignores layer_index and uses all 12 layers
    concatenated into one 2304-dim vector per frame instead (matches
    concat_layer_probe.py's "does combining layers help" question).
    """
    delta_angles, delta_hs, h_currents = [], [], []
    with h5py.File(dataset_path, "r") as fs, h5py.File(embeddings_path, "r") as fe:
        for start in starts:
            end = start + block_size
            angles = fs["state"][start:end, 4]
            episode_idx = fe["episode_idx"][start:end]
            if concat_layers:
                layers = np.asarray(fe["encoder_cls_layers"][start:end, :, :])
                embeddings = layers.reshape(layers.shape[0], -1)
            else:
                embeddings = fe["encoder_cls_layers"][start:end, layer_index, :]

            candidate_s = np.arange(block_size - goal_offset)
            same_episode = episode_idx[candidate_s] == episode_idx[candidate_s + goal_offset]
            valid_s = candidate_s[same_episode]
            if len(valid_s) == 0:
                continue

            delta_angle = circular_diff(angles[valid_s + goal_offset], angles[valid_s])
            delta_h = embeddings[valid_s + goal_offset] - embeddings[valid_s]

            delta_angles.append(delta_angle)
            delta_hs.append(delta_h)
            h_currents.append(embeddings[valid_s])

    return np.concatenate(delta_angles), np.concatenate(delta_hs), np.concatenate(h_currents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify which way the block needs to rotate to reach its real task goal, from delta_h."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("/scratch-shared/orinxAI/stable-wm-data/datasets/pusht_expert_train.h5")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("/scratch-shared/orinxAI/embeddings/pusht_encoder_cls_fp32.h5")
    )
    parser.add_argument("--layer-index", type=int, default=9)
    parser.add_argument(
        "--concat-layers",
        action="store_true",
        help="Use all 12 layers concatenated (2304-dim) instead of a single --layer-index.",
    )
    parser.add_argument("--goal-offset", type=int, default=25, help="Matches goal_offset_steps in config/eval/pusht.yaml.")
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feature",
        choices=["delta", "current"],
        default="delta",
        help="'delta' = h(goal)-h(current) (needs the goal frame). "
        "'current' = h(current) alone -- tests whether the current frame's CLS token already "
        "predicts which way the block needs to rotate, with no goal frame involved at all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as f:
        total_rows = f["state"].shape[0]

    rng = np.random.default_rng(args.seed)
    num_slots = total_rows // args.block_size
    starts = np.sort(rng.choice(num_slots, size=args.num_blocks, replace=False)) * args.block_size
    layer_desc = "concat_layers (all 12, 2304-dim)" if args.concat_layers else f"layer_index={args.layer_index}"
    print(f"Sampling {args.num_blocks} contiguous blocks of {args.block_size} rows from {total_rows} total rows ({layer_desc})")

    delta_angles, delta_hs, h_currents = sample_goal_pairs(
        args.dataset, args.embeddings, args.layer_index, starts, args.block_size, args.goal_offset,
        concat_layers=args.concat_layers,
    )
    print(f"Loaded {len(delta_angles)} real (current, goal) pairs, goal_offset={args.goal_offset}")
    print(
        f"delta_angle stats: mean={delta_angles.mean():.4f} std={delta_angles.std():.4f} "
        f"mean|delta_angle|={np.abs(delta_angles).mean():.4f} median|delta_angle|={np.median(np.abs(delta_angles)):.4f}"
    )

    labels = (delta_angles > 0).astype(int)
    print(f"label balance: {np.bincount(labels)} (positive=CCW-ish, negative=CW-ish per circular_diff convention)")

    features = delta_hs if args.feature == "delta" else h_currents
    print(f"Classifying from feature={args.feature!r} (shape={features.shape})")

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=args.test_frac, random_state=args.seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(features[train_idx], labels[train_idx])
    accuracy = clf.score(features[test_idx], labels[test_idx])

    print(
        f"\nRotation-direction-to-goal held-out accuracy: {accuracy:.4f} "
        f"(chance=0.5, n_test={len(test_idx)})"
    )

    if args.feature == "delta":
        w = clf.coef_[0]
        w_unit = w / np.linalg.norm(w)
        cosines = evaluate_direction_agreement(w_unit, delta_angles[test_idx], delta_hs[test_idx])
        print(
            f"Direction agreement (classifier decision direction vs. true delta_h, alpha-invariant): "
            f"cosine_sim mean={np.nanmean(cosines):.4f} std={np.nanstd(cosines):.4f}"
        )


if __name__ == "__main__":
    main()
