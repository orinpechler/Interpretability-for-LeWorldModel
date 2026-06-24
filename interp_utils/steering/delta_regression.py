#!/usr/bin/env python
"""Direct delta regression: fit Δfeature -> Δh empirically, instead of
inverting a h -> feature probe.

validate_direction_rendered.py confirmed, with the cross-episode confound
removed (real-vs-rendered cosine_sim ~0.999 sanity check passing), that
steering_vector's minimum-norm/independent-per-dimension direction is
essentially orthogonal to how real activations actually change when the
underlying state changes by a known amount -- regardless of probe accuracy
(block_angle, R=0.933: cosine_sim=-0.025; block_position, R=0.996:
cosine_sim=0.107). This is the null-space problem the original review
predicted: D=192 vs K=1-2 leaves a huge unconstrained orthogonal
complement, and the probe's own readout gradient doesn't capture the
correlated, incidental variation real data actually moves through.

This module builds the alternative the review suggested: collect many
(delta_raw, Δh) pairs from real (base_frame, delta) examples -- using the
SAME rendering methodology already validated (render the perturbed state,
encode it, subtract the base frame's CACHED embedding) -- and fit a direct
linear regression Δh = delta_raw * B (no intercept: delta=0 must give
Δh=0 by construction) via ordinary least squares. Unlike probe inversion,
this directly estimates the AVERAGE real embedding change for a given
delta, including whatever correlated/incidental structure real data has
that the probe's own readout gradient never captured.

Needs the model (torch) and a working PushT renderer (pygame), same as
validate_direction_rendered.py -- not CPU/login-node-quick like
validate_direction.py. Each example costs one render + one encode; use
jobs/steer_delta_regression.sh for a full run (100-200 examples).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

from interp_utils.steering.probe_io import ProbeSpec, best_layer_for_target, load_probe_dir
from interp_utils.steering.validate_direction_rendered import cosine_similarity, encode_at_probe_layer
from render_utils import make_pusht_env, render_pusht_state_vector


def embedding_space_for_probe(probe: ProbeSpec) -> str:
    return "projected_emb" if probe.layer_index is None else "encoder_cls_layers"


def read_embedding_at_row(embeddings_h5: h5py.File, row: int, probe: ProbeSpec) -> np.ndarray:
    """Direct row lookup into the embeddings cache. embeddings_h5's row i
    corresponds EXACTLY to dataset_h5's row i (see its row_mapping attr),
    so no episode_idx/step_idx matching is needed for an arbitrary frame --
    unlike validate_direction.py's read_initial_embedding, which only ever
    looks up step_idx==0 frames.
    """
    if embedding_space_for_probe(probe) == "projected_emb":
        return np.asarray(embeddings_h5["projected_emb"][row])
    return np.asarray(embeddings_h5["encoder_cls_layers"][row, probe.layer_index])


def sample_deltas(num_examples: int, delta_min: float, delta_max: float, rng: np.random.Generator) -> np.ndarray:
    """Sample `num_examples` deltas with magnitude in [delta_min, delta_max]
    and a random sign, so the regression sees both directions and a range
    of magnitudes -- not just the single operating point we already tested.
    """
    magnitudes = rng.uniform(delta_min, delta_max, size=num_examples)
    signs = rng.choice([-1.0, 1.0], size=num_examples)
    return magnitudes * signs


def build_delta_dataset(
    model,
    env,
    dataset_h5: h5py.File,
    embeddings_h5: h5py.File,
    probe: ProbeSpec,
    target_dim_index: int,
    frame_indices: np.ndarray,
    deltas: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """For each (frame_index, delta) pair, render the perturbed state and
    encode it; return delta_hs (N, D).

    Uses the base frame's CACHED embedding (no re-encode) as the "before"
    side -- validate_direction_rendered.py's rerender_vs_real check showed
    this introduces only small, delta-independent jitter (cosine_sim
    ~0.999, norm ~0.4), cheap to average out over many examples versus
    re-rendering every base frame too.
    """
    state_column = probe.target_columns[target_dim_index]
    delta_hs = np.zeros((len(frame_indices), probe.feature_mean.shape[0]), dtype=np.float64)

    for i, (frame_index, delta) in enumerate(zip(frame_indices, deltas)):
        frame_index = int(frame_index)
        state = np.asarray(dataset_h5["state"][frame_index], dtype=np.float32)
        synthetic_state = state.copy()
        synthetic_state[state_column] += delta

        synthetic_frame = render_pusht_state_vector(
            synthetic_state, env=env, reset=True, close=False, step_after_set=True
        )
        synthetic_pixels = torch.from_numpy(np.asarray(synthetic_frame)).unsqueeze(0)

        h_base = read_embedding_at_row(embeddings_h5, frame_index, probe)
        h_perturbed = encode_at_probe_layer(model, synthetic_pixels, probe, device)
        delta_hs[i] = h_perturbed - h_base

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  generated {i + 1}/{len(frame_indices)} examples", flush=True)

    return delta_hs


def fit_delta_regression(deltas: np.ndarray, delta_hs: np.ndarray) -> np.ndarray:
    """Ordinary least squares, NO intercept (delta=0 must give delta_h=0 by
    construction -- forcing through the origin is the physically correct
    choice here, not a mere convenience). Returns B (D,) such that the
    predicted delta_h for a new delta_raw is `delta_raw * B`.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    denom = float(deltas @ deltas)
    if denom < 1e-12:
        raise ValueError("Deltas have ~zero norm; cannot fit a regression through the origin.")
    return (deltas @ delta_hs) / denom


def delta_regression_vector(coefficients: np.ndarray, delta_raw: float) -> np.ndarray:
    """Apply the fitted regression to a new delta_raw -- drop-in analogue
    of steering_math.steering_vector, but using the empirically-fit
    direction instead of the probe's inverted readout gradient.
    """
    return delta_raw * coefficients


def evaluate_cosine_sim(deltas: np.ndarray, delta_hs: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    predicted = deltas[:, None] * coefficients[None, :]
    return np.array([cosine_similarity(predicted[i], delta_hs[i]) for i in range(len(deltas))])


def held_out_cosine_sim(
    deltas: np.ndarray, delta_hs: np.ndarray, train_frac: float = 0.8, seed: int = 0
) -> np.ndarray:
    """Fit on a random train_frac of the examples, report cosine_sim on the
    REMAINING held-out examples -- an honest generalization check, since
    in-sample fit numbers look artificially good for a least-squares fit.
    """
    n = len(deltas)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train_coefficients = fit_delta_regression(deltas[train_idx], delta_hs[train_idx])
    return evaluate_cosine_sim(deltas[test_idx], delta_hs[test_idx], train_coefficients)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a (delta, delta_h) dataset via rendering and fit a direct delta regression."
    )
    parser.add_argument("--dataset", type=Path, default=Path("stable-wm-data/datasets/pusht_expert_train.h5"))
    parser.add_argument("--embeddings", type=Path, default=Path("stable-wm-data/embeddings/pusht_encoder_cls_fp32.h5"))
    parser.add_argument("--policy", default="pusht/lewm", help="Passed to swm.wm.utils.load_pretrained, same as eval.py's cfg.policy.")
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        default="block_angle",
        choices=("agent_position", "block_position", "block_angle"),
    )
    parser.add_argument("--target-dim-index", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--layer", default=None, help="Probe name, e.g. layer_07. Default: best layer for --target.")
    parser.add_argument("--num-examples", type=int, default=150)
    parser.add_argument("--delta-min", type=float, default=0.05, help="Minimum |delta| magnitude to sample.")
    parser.add_argument("--delta-max", type=float, default=0.5, help="Maximum |delta| magnitude to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling frames/deltas and the held-out split.")
    parser.add_argument("--output", type=Path, default=Path("outputs/steering/delta_dataset.npz"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_probe_dir(args.probe_dir)
    layer_name = args.layer or best_layer_for_target(args.probe_dir / "metrics.csv", args.target)
    probe = bundle.get(probe_name=layer_name, seed=args.probe_seed)
    print(f"Probe: {layer_name!r} (target={args.target!r})")

    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.dataset, "r") as dataset_h5, h5py.File(args.embeddings, "r") as embeddings_h5:
        num_frames_total = dataset_h5["pixels"].shape[0]
        image_shape = tuple(int(d) for d in dataset_h5["pixels"].shape[1:3])
        frame_indices = rng.choice(num_frames_total, size=args.num_examples, replace=False)
        deltas = sample_deltas(args.num_examples, args.delta_min, args.delta_max, rng)

        env = make_pusht_env(image_shape=image_shape)
        try:
            delta_hs = build_delta_dataset(
                model, env, dataset_h5, embeddings_h5, probe, args.target_dim_index,
                frame_indices, deltas, device,
            )
        finally:
            env.close()

    coefficients = fit_delta_regression(deltas, delta_hs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        deltas=deltas,
        delta_hs=delta_hs,
        frame_indices=frame_indices,
        coefficients=coefficients,
        target=args.target,
        target_dim_index=args.target_dim_index,
        layer_name=layer_name,
        probe_seed=args.probe_seed,
    )
    print(f"Wrote {len(deltas)} (delta, delta_h) examples + fitted coefficients to {args.output}")

    in_sample = evaluate_cosine_sim(deltas, delta_hs, coefficients)
    held_out = held_out_cosine_sim(deltas, delta_hs, train_frac=0.8, seed=args.seed)
    print(
        f"in-sample cosine_sim (fit on all {len(deltas)}, optimistic): "
        f"mean={np.nanmean(in_sample):.4f} std={np.nanstd(in_sample):.4f}"
    )
    print(
        f"held-out cosine_sim (fit on 80%, evaluated on unseen 20%, n={len(held_out)}): "
        f"mean={np.nanmean(held_out):.4f} std={np.nanstd(held_out):.4f}"
    )


if __name__ == "__main__":
    main()
