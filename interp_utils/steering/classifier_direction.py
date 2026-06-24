#!/usr/bin/env python
"""Use a trained left/right classifier's decision direction as a steering
vector, and check how well it agrees with the true empirical delta.

A third way to get a steering direction, alongside probe-inversion
(steering_math.steering_vector) and direct delta regression
(delta_regression.py): fit a binary classifier (rotated left vs right) on
TRAIN data, take its decision-boundary normal vector as the candidate
direction, then on HELD-OUT data ask two separate questions:

  1. Direction agreement: cosine_sim(sign(delta_i) * w_unit, delta_h_i).
     This is the primary metric and is INVARIANT to any positive scale
     applied to w_unit -- multiplying a vector by alpha>0 never changes its
     direction. So "ablating alpha to improve cosine_sim" is not
     meaningful; cosine_sim is reported once, not per-alpha.

  2. Magnitude agreement: ||alpha * sign(delta_i) * w_unit - delta_h_i||_2,
     averaged over held-out examples, swept over a grid of alpha AND
     compared to the closed-form optimal alpha (minimizing this exactly):

         alpha* = mean_i[ sign(delta_i) * (w_unit . delta_h_i) ]

     (derived by setting d/d(alpha) of the average squared L2 error to 0,
     using ||w_unit||=1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from interp_utils.steering.metrics import write_metrics_table


def fit_classifier_direction(
    delta_hs_train: np.ndarray, signs_train: np.ndarray
) -> tuple[np.ndarray, LogisticRegression]:
    """Fit logistic regression (delta_h -> sign label) and return the
    unit-normalized decision direction (oriented so that w_unit . delta_h
    is positive for the positive-sign class), plus the fitted classifier.
    """
    clf = LogisticRegression(max_iter=2000)
    clf.fit(delta_hs_train, signs_train)
    w = clf.coef_[0]
    w_unit = w / np.linalg.norm(w)
    return w_unit, clf


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def evaluate_direction_agreement(
    w_unit: np.ndarray, deltas_test: np.ndarray, delta_hs_test: np.ndarray
) -> np.ndarray:
    """Per-example cosine_sim(sign(delta_i) * w_unit, delta_h_i). Invariant
    to any positive scaling of w_unit -- this is the headline metric.
    """
    signs = np.sign(deltas_test)
    return np.array(
        [cosine_similarity(signs[i] * w_unit, delta_hs_test[i]) for i in range(len(deltas_test))]
    )


def optimal_alpha(w_unit: np.ndarray, deltas: np.ndarray, delta_hs: np.ndarray) -> float:
    """Closed-form alpha minimizing mean ||alpha*sign(delta_i)*w_unit - delta_h_i||^2."""
    signs = np.sign(deltas)
    projections = delta_hs @ w_unit  # (N,)
    return float(np.mean(signs * projections))


def l2_error_for_alpha(
    alpha: float, w_unit: np.ndarray, deltas: np.ndarray, delta_hs: np.ndarray
) -> float:
    signs = np.sign(deltas)
    predicted = alpha * signs[:, None] * w_unit[None, :]
    return float(np.mean(np.linalg.norm(predicted - delta_hs, axis=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a left/right classifier's decision direction as a steering vector; "
        "check direction agreement (cosine_sim) and magnitude agreement (L2 error vs alpha)."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Grid of alpha values to evaluate L2 error at. Default: a grid around the closed-form optimum.",
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

    signs = (deltas > 0).astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(deltas)), test_size=args.test_frac, random_state=args.seed, stratify=signs
    )

    w_unit, clf = fit_classifier_direction(delta_hs[train_idx], signs[train_idx])
    held_out_accuracy = float(clf.score(delta_hs[test_idx], signs[test_idx]))
    print(f"classifier held-out sign accuracy: {held_out_accuracy:.4f}")

    cosines = evaluate_direction_agreement(w_unit, deltas[test_idx], delta_hs[test_idx])
    print(
        f"direction agreement (alpha-invariant): cosine_sim mean={np.nanmean(cosines):.4f} "
        f"std={np.nanstd(cosines):.4f} (n_test={len(test_idx)})"
    )

    alpha_star = optimal_alpha(w_unit, deltas[test_idx], delta_hs[test_idx])
    print(f"closed-form optimal alpha (minimizes mean L2 error): {alpha_star:.4f}")

    if args.alphas is not None:
        alphas = np.array(args.alphas, dtype=float)
    else:
        # grid centered on the optimum, spanning 0 to ~2x it
        span = max(abs(alpha_star), 1e-6) * 2
        alphas = np.linspace(0, span, 9)
        alphas = np.unique(np.concatenate([alphas, [alpha_star]]))

    rows = []
    for alpha in alphas:
        l2 = l2_error_for_alpha(alpha, w_unit, deltas[test_idx], delta_hs[test_idx])
        rows.append({"target": target, "alpha": float(alpha), "mean_l2_error": l2})
        marker = "  <-- closed-form optimum" if np.isclose(alpha, alpha_star) else ""
        print(f"  alpha={alpha:8.4f}  mean_l2_error={l2:.4f}{marker}")

    if args.output_dir is not None:
        write_metrics_table(args.output_dir, rows, filename_stem="classifier_direction_alpha_sweep")
        summary_rows = [
            {
                "target": target,
                "held_out_sign_accuracy": held_out_accuracy,
                "cosine_sim_mean": float(np.nanmean(cosines)),
                "cosine_sim_std": float(np.nanstd(cosines)),
                "optimal_alpha": alpha_star,
                "n_test": len(test_idx),
            }
        ]
        write_metrics_table(args.output_dir, summary_rows, filename_stem="classifier_direction_summary")
        print(f"\nWrote results to {args.output_dir}/classifier_direction_{{alpha_sweep,summary}}.csv/.json")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([r["alpha"] for r in rows], [r["mean_l2_error"] for r in rows], marker="o")
            ax.axvline(alpha_star, color="red", linestyle="--", label=f"optimal alpha={alpha_star:.3f}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("mean L2 error")
            ax.set_title(f"Magnitude agreement vs. alpha ({target})\ncosine_sim={np.nanmean(cosines):.3f} (alpha-invariant)")
            ax.legend()
            fig.tight_layout()
            args.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output_dir / "classifier_direction_alpha_sweep.png")
            plt.close(fig)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
