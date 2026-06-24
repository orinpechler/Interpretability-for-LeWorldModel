#!/usr/bin/env python
"""Is the block's CURRENT orientation (which way the T points) linearly
separable from a single frame's embedding?

Unlike every other script in this package, this needs no delta,
perturbation, or rendering at all -- it's a direct decode of the existing
recorded block_angle from the existing cached embedding (the simplest
possible reframing of the probing question: discrete quadrant
classification instead of continuous regression).

Quadrants are defined by splitting [0, 2*pi) into `num_quadrants` equal
slices (default 4); quadrant 0 and quadrant 2 are 180 degrees apart (e.g.
"pointing up" vs "pointing down"), the pair most likely to look visually
distinct.

I/O note: `encoder_cls_layers` is chunked (1024, 12, 192) -- a few thousand
SCATTERED rows would force decompressing nearly the entire ~21GB array
(same chunk-touching problem we hit with the `state` column, just much
bigger). Instead this samples a handful of CONTIGUOUS, chunk-aligned
blocks spread across the file: each PushT episode is only ~125 frames on
average, so a single 1024-row block already spans ~8 different episodes
and a wide range of angles, while only touching 1-2 chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def sample_contiguous_blocks(total_rows: int, num_blocks: int, block_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_slots = total_rows // block_size
    slot_indices = rng.choice(num_slots, size=num_blocks, replace=False)
    return np.sort(slot_indices) * block_size


def load_blocks(
    dataset_path: Path, embeddings_path: Path, layer_index: int, starts: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    angles_list, embeddings_list = [], []
    with h5py.File(dataset_path, "r") as fs, h5py.File(embeddings_path, "r") as fe:
        for start in starts:
            angles_list.append(fs["state"][start : start + block_size, 4])
            embeddings_list.append(fe["encoder_cls_layers"][start : start + block_size, layer_index, :])
    return np.concatenate(angles_list), np.concatenate(embeddings_list)


def quadrant_label(angles: np.ndarray, num_quadrants: int) -> np.ndarray:
    width = 2 * np.pi / num_quadrants
    return (np.floor(angles / width).astype(int)) % num_quadrants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether two orientation quadrants are linearly separable from the cached embedding."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("/scratch-shared/orinxAI/stable-wm-data/datasets/pusht_expert_train.h5")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("/scratch-shared/orinxAI/embeddings/pusht_encoder_cls_fp32.h5")
    )
    parser.add_argument("--layer-index", type=int, default=9)
    parser.add_argument("--num-quadrants", type=int, default=4)
    parser.add_argument("--quadrant-a", type=int, default=0)
    parser.add_argument("--quadrant-b", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=1024, help="Should match the embeddings file's chunk size.")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as f:
        total_rows = f["state"].shape[0]

    starts = sample_contiguous_blocks(total_rows, args.num_blocks, args.block_size, args.seed)
    print(f"Sampling {args.num_blocks} contiguous blocks of {args.block_size} rows from {total_rows} total rows")

    angles, embeddings = load_blocks(args.dataset, args.embeddings, args.layer_index, starts, args.block_size)
    print(f"Loaded {len(angles)} frames, embedding dim={embeddings.shape[1]}")

    labels = quadrant_label(angles, args.num_quadrants)
    print(f"Quadrant counts: {np.bincount(labels, minlength=args.num_quadrants)}")

    mask = (labels == args.quadrant_a) | (labels == args.quadrant_b)
    print(f"Quadrant {args.quadrant_a} vs {args.quadrant_b}: {mask.sum()} usable examples")

    binary_labels = (labels[mask] == args.quadrant_b).astype(int)
    features = embeddings[mask]

    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=args.test_frac, random_state=args.seed, stratify=binary_labels
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(features[train_idx], binary_labels[train_idx])
    accuracy = clf.score(features[test_idx], binary_labels[test_idx])

    print(
        f"\nQuadrant {args.quadrant_a} vs {args.quadrant_b} held-out accuracy: {accuracy:.4f} "
        f"(chance=0.5, n_test={len(test_idx)})"
    )


if __name__ == "__main__":
    main()
