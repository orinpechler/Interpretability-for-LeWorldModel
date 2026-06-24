#!/usr/bin/env python
"""A battery of varied BINARY separability tests on (delta, delta_h) pairs.

direction_classification.py showed accuracy-over-chance climbing with finer
granularity, but pooled everything into one sweep. This module asks a
sharper question per-experiment: which SPECIFIC binary distinctions are
separable in the empirical embedding delta, and which aren't? Finding both
kinds of result is the point -- a battery that only ever reports "separable"
isn't trustworthy; including a deliberate negative control (label-shuffling)
demonstrates the method can detect non-separability too, not just confirm
whatever we already expect.

All experiments reuse the SAME (delta, delta_h) pairs delta_regression.py
already produced -- pure numpy/sklearn, no model/GPU/rendering needed.

Experiments:
  sign_all                 -- moved positive vs negative (the full dataset)
  magnitude_above_below_median -- big vs small |delta|, sign-agnostic
  sign_small_deltas_only    -- sign, restricted to the smallest-magnitude third
  sign_large_deltas_only    -- sign, restricted to the largest-magnitude third
  sign_shuffled_control     -- sign labels randomly permuted (expect ~chance)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from interp_utils.steering.metrics import write_metrics_table


LabelFn = Callable[..., tuple[np.ndarray, np.ndarray]]


def label_sign(deltas: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Label = sign of delta. Mask = everything (no restriction)."""
    labels = (deltas > 0).astype(int)
    mask = np.ones(len(deltas), dtype=bool)
    return labels, mask


def label_magnitude_median(deltas: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Label = |delta| above/below the median, ignoring sign entirely."""
    abs_d = np.abs(deltas)
    median = np.median(abs_d)
    labels = (abs_d > median).astype(int)
    mask = np.ones(len(deltas), dtype=bool)
    return labels, mask


def label_sign_restricted(
    deltas: np.ndarray, magnitude_quantile_range: tuple[float, float], **_
) -> tuple[np.ndarray, np.ndarray]:
    """Label = sign of delta, but only keep examples whose |delta| falls
    within the given quantile range of the magnitude distribution -- tests
    whether sign is still separable within a narrow magnitude band, rather
    than relying on a wide range that mixes the easy (large) and hard
    (small) cases together.
    """
    abs_d = np.abs(deltas)
    lo_q, hi_q = magnitude_quantile_range
    lo, hi = np.quantile(abs_d, [lo_q, hi_q])
    mask = (abs_d >= lo) & (abs_d <= hi)
    labels = (deltas > 0).astype(int)
    return labels, mask


def label_shuffled_sign(deltas: np.ndarray, seed: int = 0, **_) -> tuple[np.ndarray, np.ndarray]:
    """Negative control: sign labels are computed normally, then randomly
    permuted, breaking any real association with delta_h. Accuracy should
    collapse to ~50% if the pipeline is sound -- a sanity check that high
    accuracy elsewhere isn't a methodological artifact.
    """
    labels = (deltas > 0).astype(int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labels)
    mask = np.ones(len(deltas), dtype=bool)
    return shuffled, mask


EXPERIMENTS: dict[str, tuple[LabelFn, dict]] = {
    "sign_all": (label_sign, {}),
    "magnitude_above_below_median": (label_magnitude_median, {}),
    "sign_small_deltas_only": (label_sign_restricted, {"magnitude_quantile_range": (0.0, 0.34)}),
    "sign_large_deltas_only": (label_sign_restricted, {"magnitude_quantile_range": (0.67, 1.0)}),
    "sign_shuffled_control": (label_shuffled_sign, {}),
}


def run_experiment(
    name: str,
    deltas: np.ndarray,
    delta_hs: np.ndarray,
    label_fn: LabelFn,
    kwargs: dict,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    labels, mask = label_fn(deltas, seed=seed, **kwargs)
    d_masked, dh_masked, labels_masked = deltas[mask], delta_hs[mask], labels[mask]

    train_idx, test_idx = train_test_split(
        np.arange(len(d_masked)), test_size=test_frac, random_state=seed, stratify=labels_masked
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(dh_masked[train_idx], labels_masked[train_idx])
    accuracy = float(clf.score(dh_masked[test_idx], labels_masked[test_idx]))

    return {
        "experiment": name,
        "accuracy": accuracy,
        "chance": 0.5,
        "accuracy_over_chance": accuracy / 0.5,
        "n": int(mask.sum()),
        "n_test": len(test_idx),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a battery of varied binary separability tests on a delta_regression dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a delta_regression.py output .npz (e.g. outputs/steering/delta_dataset_block_angle.npz).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Which experiments to run. Default: all of them.",
    )
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset)
    deltas = data["deltas"]
    delta_hs = data["delta_hs"]
    target = str(data["target"]) if "target" in data else args.dataset.stem

    print(f"Dataset: {args.dataset} (target={target!r}, n={len(deltas)})")
    rows = []
    for name in args.experiments:
        label_fn, kwargs = EXPERIMENTS[name]
        result = run_experiment(name, deltas, delta_hs, label_fn, kwargs, args.test_frac, args.seed)
        result["target"] = target
        rows.append(result)
        print(
            f"  {name:32s} accuracy={result['accuracy']:.4f}  "
            f"accuracy/chance={result['accuracy_over_chance']:.2f}x  "
            f"(n={result['n']}, n_test={result['n_test']})"
        )

    if args.output_dir is not None:
        write_metrics_table(args.output_dir, rows, filename_stem="binary_separability")
        print(f"\nWrote {len(rows)} row(s) to {args.output_dir}/binary_separability.csv/.json")


if __name__ == "__main__":
    main()
