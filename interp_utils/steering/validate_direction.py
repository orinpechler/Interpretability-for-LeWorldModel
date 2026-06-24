#!/usr/bin/env python
"""Check whether steering_math.steering_vector's direction (the probe's
minimum-norm / pseudoinverse-style readout gradient) actually matches how
real activations differ when the underlying state differs by that delta.

Why this matters: steering_vector() returns a single direction in a D-dim
embedding space (D=192 here) that shifts a K-dim linear probe's own
prediction by a requested delta (K=1 for block_angle). The probe's null
space is (D - K) dimensions -- huge -- and is completely unconstrained by
that derivation. The minimum-norm choice ("don't move in the null space")
is a modeling assumption, not a fact about the model: real activations
encoding "block rotated by delta" may differ from the current activation
by steering_vector's direction PLUS correlated variation in null-space
directions the probe never used (incidental features the probe happened to
ignore). If so, adding only steering_vector's direction can be off the data
manifold -- coherent to the probe's own readout, but not representative of
what a real "rotated by delta" activation looks like to the rest of the
network.

This script checks that directly and empirically, with NO model/torch/GPU
needed at all (runs on the login node): for mined (src, ref) episode pairs
whose initial states differ by ~delta in the target's columns (reusing
episode_pairs.find_episode_pairs), look up each episode's ALREADY-EXTRACTED
initial-frame embedding from the embeddings cache (no forward pass), take
the empirical difference (ref's embedding minus src's), and compare it via
cosine similarity against steering_vector()'s predicted direction for that
pair's actual achieved delta.

Rule of thumb (from the review that prompted this script): mean cosine
similarity >= ~0.7 means the minimum-norm assumption is roughly fine. Well
below that means the null space is dominating, and a direct delta
regression (fit Δfeature -> Δh directly, rather than inverting a h -> feature
probe) would likely track real activation differences more faithfully.

Usage (from the login node, no SLURM needed):

    python -m interp_utils.steering.validate_direction \\
        --probe-dir /scratch-shared/orinxAI/stable-wm-data/probes/block_angle \\
        --delta 0.3 --tolerance 0.02 --num-pairs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from interp_utils.steering.episode_pairs import EpisodePair, find_episode_pairs, load_initial_states
from interp_utils.steering.probe_io import ProbeSpec, best_layer_for_target, load_probe_dir
from interp_utils.steering.steering_math import predicted_raw_target, steering_vector


def embedding_space_for_probe(probe: ProbeSpec) -> str:
    """Which embeddings_h5 dataset matches the representation `probe` was
    fit on: per-layer CLS (pre-projection) for layer probes, or the
    post-projection embedding for the "projected_emb" probe.
    """
    return "projected_emb" if probe.layer_index is None else "encoder_cls_layers"


def read_initial_embedding(embeddings_h5: h5py.File, episode_id: int, probe: ProbeSpec) -> np.ndarray:
    """The already-extracted embedding at step_idx==0 for `episode_id`, in
    the exact space `probe` was fit on. No model forward pass -- this is a
    pure lookup into the embeddings cache.
    """
    episode_idx = np.asarray(embeddings_h5["episode_idx"][:])
    step_idx = np.asarray(embeddings_h5["step_idx"][:])
    mask = (episode_idx == episode_id) & (step_idx == 0)
    rows = np.nonzero(mask)[0]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one step_idx==0 frame for episode {episode_id} "
            f"in the embeddings cache, found {len(rows)}."
        )
    row = int(rows[0])

    if embedding_space_for_probe(probe) == "projected_emb":
        return np.asarray(embeddings_h5["projected_emb"][row])
    return np.asarray(embeddings_h5["encoder_cls_layers"][row, probe.layer_index])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def check_pair(embeddings_h5: h5py.File, pair: EpisodePair, probe: ProbeSpec, target_dim_index: int) -> dict:
    """Compare the empirical embedding delta for one mined pair against
    steering_vector's predicted direction for that pair's actual achieved
    delta (not the originally requested delta_target, since mining only
    matches it approximately within --tolerance).

    Runs TWO separate, complementary checks using predicted_raw_target:

    1. self_consistency_error: does adding the predicted vector to the REAL
       h_src actually move the probe's own readout by exactly
       achieved_delta_raw? Tests steering_vector's math against this
       probe's real weights and real data -- tests/test_steering_math.py
       only proves this algebraically on synthetic probes; this confirms
       the same identity holds end-to-end here. Large error => an
       implementation/wiring bug (mismatched embedding space, stale probe,
       NaNs), not a model-quality issue.

    2. probe_prediction_error: does the probe's OWN prediction on the two
       REAL embeddings (y_ref_pred - y_src_pred) agree with
       achieved_delta_raw (the GROUND-TRUTH state delta used to mine this
       pair, read from the dataset's `state` column, not from the probe at
       all)? Large error here means the probe itself doesn't track the
       true block_angle well for these specific frames -- a third failure
       mode, distinct from both the null-space question this script was
       built to check and from self_consistency_error above.
    """
    h_src = read_initial_embedding(embeddings_h5, pair.src_episode, probe)
    h_ref = read_initial_embedding(embeddings_h5, pair.ref_episode, probe)
    empirical_delta_h = h_ref - h_src

    achieved_delta_raw = float(pair.achieved_delta[target_dim_index])
    delta_vector = np.zeros(len(probe.target_columns))
    delta_vector[target_dim_index] = achieved_delta_raw
    predicted_delta_h = steering_vector(probe, delta_vector)

    y_src_pred = predicted_raw_target(probe, h_src)
    y_ref_pred = predicted_raw_target(probe, h_ref)
    probe_predicted_delta = float((y_ref_pred - y_src_pred)[target_dim_index])
    probe_prediction_error = float(probe_predicted_delta - achieved_delta_raw)

    y_after = predicted_raw_target(probe, h_src + predicted_delta_h)
    self_consistency_error = float(
        abs((y_after - y_src_pred)[target_dim_index] - achieved_delta_raw)
    )

    return {
        "src_episode": pair.src_episode,
        "ref_episode": pair.ref_episode,
        "achieved_delta": achieved_delta_raw,
        "empirical_norm": float(np.linalg.norm(empirical_delta_h)),
        "predicted_norm": float(np.linalg.norm(predicted_delta_h)),
        "cosine_sim": cosine_similarity(empirical_delta_h, predicted_delta_h),
        "self_consistency_error": self_consistency_error,
        "probe_predicted_delta": probe_predicted_delta,
        "probe_prediction_error": probe_prediction_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check steering_vector's direction against empirical activation deltas. CPU-only, no model needed."
    )
    parser.add_argument("--dataset", type=Path, default=Path("stable-wm-data/datasets/pusht_expert_train.h5"))
    parser.add_argument("--embeddings", type=Path, default=Path("stable-wm-data/embeddings/pusht_encoder_cls_fp32.h5"))
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        default="block_angle",
        choices=("agent_position", "block_position", "block_angle"),
    )
    parser.add_argument("--target-dim-index", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--layer", default=None, help="Probe name, e.g. layer_07 or projected_emb. Default: best layer for --target.")
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--num-pairs", type=int, default=10, help="Keep this small -- the point is a quick CPU sanity check, not a full sweep.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Mean cosine-sim threshold below which the null space is flagged as likely dominating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_probe_dir(args.probe_dir)
    layer_name = args.layer or best_layer_for_target(args.probe_dir / "metrics.csv", args.target)
    probe = bundle.get(probe_name=layer_name, seed=args.probe_seed)
    print(f"Probe: {layer_name!r} (target={args.target!r}, embedding space={embedding_space_for_probe(probe)!r}, D={probe.feature_mean.shape[0]})")

    with h5py.File(args.dataset, "r") as dataset_h5:
        episode_ids, initial_states = load_initial_states(dataset_h5, num_episodes=100)

    delta_target = np.zeros(len(probe.target_columns))
    delta_target[args.target_dim_index] = args.delta
    pairs = find_episode_pairs(
        episode_ids,
        initial_states,
        probe.target_columns,
        delta_target,
        args.tolerance,
        max_pairs=args.num_pairs,
    )
    print(f"Found {len(pairs)} pair(s) for delta={args.delta} (tolerance={args.tolerance})")
    if not pairs:
        raise SystemExit("No episode pairs found -- widen --tolerance or pick a different --delta.")

    rows = []
    with h5py.File(args.embeddings, "r") as embeddings_h5:
        for pair in pairs:
            row = check_pair(embeddings_h5, pair, probe, args.target_dim_index)
            rows.append(row)
            print(
                f"  src={row['src_episode']} ref={row['ref_episode']} "
                f"achieved_delta={row['achieved_delta']:.4f} "
                f"||empirical||={row['empirical_norm']:.3f} "
                f"||predicted||={row['predicted_norm']:.3f} "
                f"cosine_sim={row['cosine_sim']:.4f} "
                f"self_consistency_error={row['self_consistency_error']:.2e} "
                f"probe_predicted_delta={row['probe_predicted_delta']:.4f} "
                f"probe_prediction_error={row['probe_prediction_error']:.4f}"
            )

    self_consistency_errors = np.array([row["self_consistency_error"] for row in rows])
    max_self_consistency_error = float(np.max(self_consistency_errors))
    if max_self_consistency_error > 1e-6:
        print(
            f"\nWARNING: max self_consistency_error={max_self_consistency_error:.2e} > 1e-6 -- "
            "steering_vector's predicted vector does NOT reproduce the requested delta on this "
            "probe's own readout. This points at an implementation/wiring bug (wrong embedding "
            "space, stale probe, NaNs), not a model-quality issue -- fix this before trusting the "
            "cosine_sim numbers below."
        )

    probe_prediction_errors = np.array([row["probe_prediction_error"] for row in rows])
    print(
        f"probe_prediction_error vs requested={args.delta}: mean={np.mean(probe_prediction_errors):.4f}  "
        f"std={np.std(probe_prediction_errors):.4f}  "
        f"min={np.min(probe_prediction_errors):.4f}  max={np.max(probe_prediction_errors):.4f}  "
        "(large values here mean the probe itself doesn't track ground-truth block_angle well for "
        "these frames -- a model/probe-quality issue, separate from the null-space question below)"
    )

    cosine_sims = np.array([row["cosine_sim"] for row in rows])
    mean_cos = float(np.nanmean(cosine_sims))
    print()
    print(
        f"mean cosine_sim={mean_cos:.4f}  std={np.nanstd(cosine_sims):.4f}  "
        f"min={np.nanmin(cosine_sims):.4f}  max={np.nanmax(cosine_sims):.4f}  (n={len(rows)})"
    )

    if mean_cos >= args.threshold:
        print(
            f"mean cosine_sim >= {args.threshold}: the minimum-norm direction is reasonably "
            "aligned with empirical activation deltas -- steering_vector's pseudoinverse "
            "assumption looks OK for this (probe, delta)."
        )
    else:
        print(
            f"mean cosine_sim < {args.threshold}: the null space likely dominates the "
            "perturbation norm here. Consider fitting a direct delta regression "
            "(Δfeature -> Δh, regressing the empirical activation difference directly) "
            "instead of inverting the probe."
        )


if __name__ == "__main__":
    main()
