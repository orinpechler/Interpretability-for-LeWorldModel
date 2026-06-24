#!/usr/bin/env python
"""How much DIRECTIONAL (not exact-magnitude) information about a
perturbation survives in the empirical embedding delta?

steering_vector/delta_regression both cap out around cosine_sim~0.3 for the
EXACT direction+magnitude of a perturbation (see validate_direction_rendered.py,
delta_regression.py), and the PCA/implied-direction diagnostic showed why:
per-example deviation from the mean direction is ~3x the mean's own norm --
high-dimensional, frame-specific noise dominates the shared signal.

But exact regression and discrete classification are different tasks:
classification only needs the decision boundary to be roughly right, which
can be far more robust to that kind of noise. This script tests exactly
that, directly on the (delta, delta_h) pairs delta_regression.py already
produced -- no model/GPU/rendering needed, pure numpy/sklearn:

  1. Bin the continuous deltas into N discrete classes by sign and (within
     each sign) magnitude quantile -- e.g. N=2 is plain left/right or
     pos/neg; N=4 adds a small/large split within each sign; etc.
  2. Fit a multinomial logistic regression on embedding deltas -> bin label,
     80/20 held out.
  3. Sweep N and report accuracy vs. chance (1/N) -- this is the curve that
     answers "how much directional granularity is actually recoverable."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from interp_utils.steering.metrics import write_metrics_table


def bin_deltas(deltas: np.ndarray, num_bins: int) -> np.ndarray:
    """Discretize signed deltas into `num_bins` classes: split by sign
    first, then by magnitude quantile within each sign (so label 0 = most
    negative/smallest-magnitude-negative bin ... num_bins-1 = largest
    positive bin). num_bins must be even so the split between signs is
    balanced.
    """
    if num_bins < 2 or num_bins % 2 != 0:
        raise ValueError(f"num_bins must be even and >= 2, got {num_bins}")

    bins_per_sign = num_bins // 2
    labels = np.zeros(len(deltas), dtype=int)

    for sign, base_label in ((-1, 0), (1, bins_per_sign)):
        mask = (deltas < 0) if sign == -1 else (deltas > 0)
        magnitudes = np.abs(deltas[mask])
        if bins_per_sign == 1:
            labels[mask] = base_label
            continue
        edges = np.quantile(magnitudes, np.linspace(0, 1, bins_per_sign + 1))
        bin_idx = np.digitize(magnitudes, edges[1:-1], right=False)
        labels[mask] = base_label + bin_idx

    return labels


def evaluate_classification(
    deltas: np.ndarray,
    delta_hs: np.ndarray,
    num_bins: int,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """Bin deltas into num_bins classes, fit logistic regression on a
    train split, report held-out accuracy vs. chance (1/num_bins).
    """
    labels = bin_deltas(deltas, num_bins)
    train_idx, test_idx = train_test_split(
        np.arange(len(deltas)), test_size=test_frac, random_state=seed, stratify=labels
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(delta_hs[train_idx], labels[train_idx])
    accuracy = float(clf.score(delta_hs[test_idx], labels[test_idx]))
    chance = 1.0 / num_bins

    return {
        "num_bins": num_bins,
        "accuracy": accuracy,
        "chance": chance,
        "accuracy_over_chance": accuracy / chance,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep directional-classification accuracy vs. bin granularity on a delta_regression dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a delta_regression.py output .npz (e.g. outputs/steering/delta_dataset_block_angle.npz).",
    )
    parser.add_argument(
        "--num-bins-list",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="Bin counts to sweep, each must be even.",
    )
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None, help="If given, write results + plot here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset)
    deltas = data["deltas"]
    delta_hs = data["delta_hs"]
    target = str(data["target"]) if "target" in data else args.dataset.stem

    print(f"Dataset: {args.dataset} (target={target!r}, n={len(deltas)})")
    rows = []
    for num_bins in args.num_bins_list:
        result = evaluate_classification(deltas, delta_hs, num_bins, args.test_frac, args.seed)
        result["target"] = target
        rows.append(result)
        print(
            f"  num_bins={num_bins:2d}  accuracy={result['accuracy']:.4f}  "
            f"chance={result['chance']:.4f}  accuracy/chance={result['accuracy_over_chance']:.2f}x  "
            f"(n_test={result['n_test']})"
        )

    if args.output_dir is not None:
        write_metrics_table(args.output_dir, rows, filename_stem="direction_classification")
        print(f"\nWrote {len(rows)} row(s) to {args.output_dir}/direction_classification.csv/.json")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            bins = [r["num_bins"] for r in rows]
            acc = [r["accuracy"] for r in rows]
            chance = [r["chance"] for r in rows]
            ax.plot(bins, acc, marker="o", label="held-out accuracy")
            ax.plot(bins, chance, marker="o", linestyle="--", label="chance (1/num_bins)")
            ax.set_xlabel("number of direction bins")
            ax.set_ylabel("accuracy")
            ax.set_title(f"Direction classification accuracy vs. granularity ({target})")
            ax.legend()
            fig.tight_layout()
            args.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output_dir / "direction_classification.png")
            plt.close(fig)
            print(f"Wrote plot to {args.output_dir}/direction_classification.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
