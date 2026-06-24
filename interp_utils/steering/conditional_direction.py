#!/usr/bin/env python
"""Does the steering direction depend on the CURRENT state, not just the
requested delta? I.e. is v(Δ, h_base) a meaningfully better model than the
single global v(Δ) fit by delta_regression.py?

The PCA/implied-direction diagnostic showed each example's true steering
direction deviates from the population mean by 3-4x the mean's own size --
consistent with frame-specific context mattering. This script tests the
most direct, interpretable version of that hypothesis: bin examples by
their TRUE base state (e.g. the actual block_angle value at the start
frame, read from the dataset, not the embedding), fit a SEPARATE
delta-regression direction PER BIN using only that bin's data, and compare
against the single GLOBAL direction evaluated on the exact same held-out
examples. Same global train/test split throughout, so "local beats global"
or "local doesn't help" is a fair, apples-to-apples comparison, not an
artifact of different data splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from interp_utils.steering.delta_regression import fit_delta_regression
from interp_utils.steering.metrics import write_metrics_table
from interp_utils.steering.validate_direction_rendered import cosine_similarity


def load_base_state_feature(dataset_path: Path, frame_indices: np.ndarray, state_column: int) -> np.ndarray:
    """Read the TRUE base-state feature (e.g. block_angle) for each
    example's base frame -- the conditioning variable, read from the
    dataset directly, not derived from the embedding.

    Reads the WHOLE state column once (state is tiny overall, ~65MB total
    for 2.3M rows x 7 columns) and indexes in numpy, rather than scattered
    h5py fancy-indexing on a handful of thousand rows -- that pattern was
    the actual bottleneck behind episode_pairs.load_initial_states before
    it was fixed the same way; same fix applies here.
    """
    with h5py.File(dataset_path, "r") as f:
        full_column = np.asarray(f["state"][:, state_column])
    return full_column[frame_indices]


def bin_by_quantile(values: np.ndarray, num_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, num_bins + 1))
    bin_idx = np.digitize(values, edges[1:-1], right=False)
    return bin_idx


def evaluate_direction(coefficients: np.ndarray, deltas: np.ndarray, delta_hs: np.ndarray) -> np.ndarray:
    predicted = deltas[:, None] * coefficients[None, :]
    return np.array([cosine_similarity(predicted[i], delta_hs[i]) for i in range(len(deltas))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a per-bin (state-conditioned) steering direction against the global direction."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="delta_regression.py output .npz")
    parser.add_argument(
        "--state-dataset",
        type=Path,
        default=Path("/scratch-shared/orinxAI/stable-wm-data/datasets/pusht_expert_train.h5"),
        help="Raw HDF5 dataset to read the TRUE base-state conditioning feature from.",
    )
    parser.add_argument("--state-column", type=int, default=4, help="State column to condition on (default 4 = block_angle).")
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--min-bin-train-size", type=int, default=30, help="Skip bins with fewer than this many train examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset)
    deltas = data["deltas"]
    delta_hs = data["delta_hs"]
    frame_indices = data["frame_indices"]
    target = str(data["target"]) if "target" in data else args.dataset.stem
    print(f"Dataset: {args.dataset} (target={target!r}, n={len(deltas)})")

    base_feature = load_base_state_feature(args.state_dataset, frame_indices, args.state_column)
    print(f"Conditioning on state column {args.state_column}: range [{base_feature.min():.3f}, {base_feature.max():.3f}]")

    bin_labels = bin_by_quantile(base_feature, args.num_bins)

    # ONE global split, reused for everything -- bins are just subsets of these same indices.
    train_idx, test_idx = train_test_split(np.arange(len(deltas)), test_size=args.test_frac, random_state=args.seed)
    train_mask = np.zeros(len(deltas), dtype=bool)
    train_mask[train_idx] = True

    global_coefficients = fit_delta_regression(deltas[train_idx], delta_hs[train_idx])
    global_cosines_all_test = evaluate_direction(global_coefficients, deltas[test_idx], delta_hs[test_idx])
    print(
        f"\nGLOBAL direction, all held-out test examples: cosine_sim mean={np.nanmean(global_cosines_all_test):.4f} "
        f"(n_test={len(test_idx)})"
    )

    rows = []
    print(f"\nPer-bin comparison (same held-out test examples for both global and local):")
    for bin_id in range(args.num_bins):
        bin_train_idx = np.nonzero((bin_labels == bin_id) & train_mask)[0]
        bin_test_idx = np.nonzero((bin_labels == bin_id) & ~train_mask)[0]

        if len(bin_train_idx) < args.min_bin_train_size or len(bin_test_idx) < 5:
            print(f"  bin {bin_id}: too few examples (train={len(bin_train_idx)}, test={len(bin_test_idx)}), skipping")
            continue

        local_coefficients = fit_delta_regression(deltas[bin_train_idx], delta_hs[bin_train_idx])

        global_on_bin = evaluate_direction(global_coefficients, deltas[bin_test_idx], delta_hs[bin_test_idx])
        local_on_bin = evaluate_direction(local_coefficients, deltas[bin_test_idx], delta_hs[bin_test_idx])

        row = {
            "target": target,
            "bin": bin_id,
            "state_range": f"[{base_feature[bin_labels == bin_id].min():.3f}, {base_feature[bin_labels == bin_id].max():.3f}]",
            "n_train": len(bin_train_idx),
            "n_test": len(bin_test_idx),
            "global_cosine_sim_mean": float(np.nanmean(global_on_bin)),
            "local_cosine_sim_mean": float(np.nanmean(local_on_bin)),
            "improvement": float(np.nanmean(local_on_bin) - np.nanmean(global_on_bin)),
        }
        rows.append(row)
        print(
            f"  bin {bin_id} (state in {row['state_range']}, n_train={row['n_train']}, n_test={row['n_test']}): "
            f"global={row['global_cosine_sim_mean']:.4f}  local={row['local_cosine_sim_mean']:.4f}  "
            f"improvement={row['improvement']:+.4f}"
        )

    if rows:
        mean_improvement = np.mean([r["improvement"] for r in rows])
        print(f"\nmean improvement (local - global) across bins: {mean_improvement:+.4f}")

    if args.output_dir is not None and rows:
        write_metrics_table(args.output_dir, rows, filename_stem="conditional_direction")
        print(f"\nWrote {len(rows)} row(s) to {args.output_dir}/conditional_direction.csv/.json")


if __name__ == "__main__":
    main()
