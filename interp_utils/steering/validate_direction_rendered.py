#!/usr/bin/env python
"""Controlled, rendering-based check of steering_vector's predicted direction.

validate_direction.py compares against empirical embedding deltas between
DIFFERENT episodes mined by initial-state delta -- those episodes differ in
many uncontrolled ways (agent position, block position, rendering nuances),
not just the target dimension, so that comparison is confounded. This
script removes the confound: for a single real frame, build a SYNTHETIC
state identical to the real one except for the target dimension (e.g.
block_angle += delta), RE-RENDER it with the actual PushT simulator
(render_utils.render_pusht_state_vector), and encode the rendered frame.
The resulting embedding difference isolates the target dimension's effect
with everything else held fixed -- this is the proper ground truth to
compare steering_vector's predicted direction against.

Generalizes the teammate-provided interp_utils/_steering.py +
compare_steering_embeddings.py approach (which only covered block_position
x/y) to any single target dimension, including block_angle, by reusing our
own steering_math.steering_vector (independent-per-dimension, not the joint
pseudoinverse those scripts used -- see steering_math's module docstring
for why) instead of duplicating a second hook implementation.

Needs the model (torch) and a working PushT renderer (pygame) -- NOT
CPU/login-node only like validate_direction.py. Run as a real job, or
interactively only if pygame can render headlessly in your shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

from interp_utils.extract_pusht_encoder_cls_embeddings import preprocess_pixels
from interp_utils.steering.metrics import write_metrics_table
from interp_utils.steering.probe_io import ProbeSpec, best_layer_for_target, load_probe_dir
from interp_utils.steering.steering_math import predicted_raw_target, steering_vector
from render_utils import make_pusht_env, render_pusht_state_vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def encode_at_probe_layer(model, pixels: torch.Tensor, probe: ProbeSpec, device: torch.device) -> np.ndarray:
    """Encode one frame and pull out the CLS embedding in the exact space
    `probe` was fit on (pre-projection per-layer CLS, or post-projection
    for the "projected_emb" probe) -- same convention as
    extract_pusht_encoder_cls_embeddings.py.
    """
    pixels = preprocess_pixels(pixels, device)
    with torch.inference_mode():
        enc_output = model.encoder(pixels, interpolate_pos_encoding=True, output_hidden_states=True)
        if probe.layer_index is None:
            cls = model.projector(enc_output.last_hidden_state[:, 0])
        else:
            cls = enc_output.hidden_states[probe.layer_index + 1][:, 0]
    return cls[0].detach().cpu().numpy()


def compare_frame(
    model,
    env,
    dataset_h5: h5py.File,
    frame_index: int,
    probe: ProbeSpec,
    target_dim_index: int,
    delta_raw: float,
    device: torch.device,
) -> dict:
    """Compare steering_vector's predicted direction against the embedding
    delta between a real frame and the SAME scene re-rendered with only
    `probe.target_columns[target_dim_index]` shifted by `delta_raw`.

    Also renders the UNPERTURBED state (delta=0) and compares its embedding
    against the real dataset frame's embedding -- this is a sanity check on
    the rendering pipeline itself (camera/zoom/anti-aliasing might not
    exactly match how the dataset's frames were originally generated). If
    rerender_vs_real_cosine_sim is not close to 1 (or rerender_vs_real_norm
    is not small) even at delta=0, the rendering pipeline is its own
    confound and the perturbed-delta results above can't be trusted yet.
    """
    real_pixels = torch.from_numpy(np.asarray(dataset_h5["pixels"][frame_index])).unsqueeze(0)
    state = np.asarray(dataset_h5["state"][frame_index], dtype=np.float32)

    state_column = probe.target_columns[target_dim_index]
    synthetic_state = state.copy()
    synthetic_state[state_column] += delta_raw

    unperturbed_frame = render_pusht_state_vector(
        state.copy(), env=env, reset=True, close=False, step_after_set=True
    )
    synthetic_frame = render_pusht_state_vector(
        synthetic_state, env=env, reset=True, close=False, step_after_set=True
    )
    unperturbed_pixels = torch.from_numpy(np.asarray(unperturbed_frame)).unsqueeze(0)
    synthetic_pixels = torch.from_numpy(np.asarray(synthetic_frame)).unsqueeze(0)

    h_real = encode_at_probe_layer(model, real_pixels, probe, device)
    h_rerendered = encode_at_probe_layer(model, unperturbed_pixels, probe, device)
    h_synthetic = encode_at_probe_layer(model, synthetic_pixels, probe, device)
    empirical_delta_h = h_synthetic - h_real

    delta_vector = np.zeros(len(probe.target_columns))
    delta_vector[target_dim_index] = delta_raw
    predicted_delta_h = steering_vector(probe, delta_vector)

    y_real_pred = predicted_raw_target(probe, h_real)
    y_synthetic_pred = predicted_raw_target(probe, h_synthetic)
    # Does the probe, applied to the ACTUALLY rendered frame, read out a
    # change of exactly delta_raw? Tests probe quality on real, controlled
    # data -- distinct from cosine_sim (direction) and from
    # validate_direction.py's episode-pair-based probe_prediction_error
    # (which has the cross-episode confound this script avoids).
    probe_rendered_delta_error = float(
        (y_synthetic_pred - y_real_pred)[target_dim_index] - delta_raw
    )

    return {
        "frame_index": frame_index,
        "delta_raw": delta_raw,
        "empirical_norm": float(np.linalg.norm(empirical_delta_h)),
        "predicted_norm": float(np.linalg.norm(predicted_delta_h)),
        "cosine_sim": cosine_similarity(empirical_delta_h, predicted_delta_h),
        "probe_rendered_delta_error": probe_rendered_delta_error,
        "rerender_vs_real_norm": float(np.linalg.norm(h_rerendered - h_real)),
        "rerender_vs_real_cosine_sim": cosine_similarity(h_rerendered, h_real),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rendering-controlled check of steering_vector's predicted direction (needs model + PushT renderer)."
    )
    parser.add_argument("--dataset", type=Path, default=Path("stable-wm-data/datasets/pusht_expert_train.h5"))
    parser.add_argument("--policy", default="pusht/lewm", help="Passed to swm.wm.utils.load_pretrained, same as eval.py's cfg.policy.")
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        default="block_angle",
        choices=("agent_position", "block_position", "block_angle"),
    )
    parser.add_argument("--target-dim-index", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--layer", default=None, help="Probe name, e.g. layer_07 or projected_emb. Default: best layer for --target.")
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        required=True,
        help="One or more delta magnitudes to check, e.g. --deltas 0.1 0.2 0.3. The SAME sampled frames are reused across all of them for an apples-to-apples comparison of how cosine_sim/probe accuracy change with delta size.",
    )
    parser.add_argument("--num-frames", type=int, default=10, help="Keep this small -- a quick controlled sanity check, not a full sweep.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling which frames to check.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, default=None, help="If given, write all rows to <output-dir>/delta_sweep.csv/.json (metrics.write_metrics_table convention).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_probe_dir(args.probe_dir)
    layer_name = args.layer or best_layer_for_target(args.probe_dir / "metrics.csv", args.target)
    probe = bundle.get(probe_name=layer_name, seed=args.probe_seed)
    print(f"Probe: {layer_name!r} (target={args.target!r}, D={probe.feature_mean.shape[0]})")

    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    with h5py.File(args.dataset, "r") as dataset_h5:
        num_frames_total = dataset_h5["pixels"].shape[0]
        image_shape = tuple(int(d) for d in dataset_h5["pixels"].shape[1:3])
        rng = np.random.default_rng(args.seed)
        frame_indices = np.sort(rng.choice(num_frames_total, size=args.num_frames, replace=False))

        env = make_pusht_env(image_shape=image_shape)
        rows = []
        try:
            for delta in args.deltas:
                print(f"--- delta={delta} ---")
                for frame_index in frame_indices:
                    row = compare_frame(
                        model, env, dataset_h5, int(frame_index), probe,
                        args.target_dim_index, delta, device,
                    )
                    row["target"] = args.target
                    row["layer"] = layer_name
                    rows.append(row)
                    print(
                        f"  frame={row['frame_index']} delta={row['delta_raw']:.4f} "
                        f"||empirical||={row['empirical_norm']:.3f} "
                        f"||predicted||={row['predicted_norm']:.3f} "
                        f"cosine_sim={row['cosine_sim']:.4f} "
                        f"probe_rendered_delta_error={row['probe_rendered_delta_error']:.4f} "
                        f"rerender_vs_real: norm={row['rerender_vs_real_norm']:.3f} "
                        f"cosine_sim={row['rerender_vs_real_cosine_sim']:.4f}"
                    )
        finally:
            env.close()

    if args.output_dir is not None:
        write_metrics_table(args.output_dir, rows, filename_stem="delta_sweep")
        print(f"\nWrote {len(rows)} row(s) to {args.output_dir}/delta_sweep.csv/.json")

    rerender_norms = np.array([row["rerender_vs_real_norm"] for row in rows])
    rerender_cosines = np.array([row["rerender_vs_real_cosine_sim"] for row in rows])
    print()
    print(
        f"rerender_vs_real (delta=0 sanity check, pooled across all deltas): norm mean={np.mean(rerender_norms):.3f}  "
        f"cosine_sim mean={np.nanmean(rerender_cosines):.4f}  -- if norm is comparable to "
        "||empirical|| above or cosine_sim is well below 1, the rendering pipeline itself is a "
        "confound and the results below shouldn't be trusted yet."
    )

    print()
    print("Per-delta breakdown (does cosine_sim/probe accuracy hold up as delta magnitude changes?):")
    for delta in args.deltas:
        delta_rows = [row for row in rows if row["delta_raw"] == delta]
        cosine_sims = np.array([row["cosine_sim"] for row in delta_rows])
        rendered_errors = np.array([row["probe_rendered_delta_error"] for row in delta_rows])
        print(
            f"  delta={delta:.4f}: cosine_sim mean={np.nanmean(cosine_sims):.4f} std={np.nanstd(cosine_sims):.4f}  "
            f"probe_rendered_delta_error mean={np.mean(rendered_errors):.4f} std={np.std(rendered_errors):.4f}  "
            f"(n={len(delta_rows)})"
        )

    cosine_sims = np.array([row["cosine_sim"] for row in rows])
    mean_cos = float(np.nanmean(cosine_sims))

    print()
    print(
        f"overall mean cosine_sim (pooled across all deltas)={mean_cos:.4f}  std={np.nanstd(cosine_sims):.4f}  "
        f"min={np.nanmin(cosine_sims):.4f}  max={np.nanmax(cosine_sims):.4f}  (n={len(rows)})"
    )

    if mean_cos >= args.threshold:
        print(
            f"mean cosine_sim >= {args.threshold}: steering_vector's direction is reasonably "
            "aligned with the properly-controlled empirical delta."
        )
    else:
        print(
            f"mean cosine_sim < {args.threshold}: even with the cross-episode confound removed, "
            "the null space dominates the perturbation norm -- consider a direct delta regression "
            "(fit Δfeature -> Δh on (real, rendered) pairs directly) instead of inverting the probe."
        )


if __name__ == "__main__":
    main()
