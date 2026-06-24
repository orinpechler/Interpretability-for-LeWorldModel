#!/usr/bin/env python
"""Open-loop steering experiment: dataset only, no live env.

Replays a source episode's recorded actions through model.rollout() with a
probe-derived perturbation injected at the chosen encoder layer, and
compares the resulting (perturbed) predicted embedding trajectory against
the real embedding trajectory of a matched reference episode (one whose
initial state differs from the source's by approximately the requested
delta).

IMPORTANT: model.rollout()'s "predicted_emb" output is in PROJECTED-EMB
space, not raw per-layer CLS space -- LeWM.encode() applies self.projector
to the CLS token before storing it as info["emb"], and predict()/pred_proj
operate downstream of that. The steering vector is derived from a *layer_L*
probe (which was fit on pre-projection CLS embeddings), but it gets added
to the residual stream INSIDE the encoder, upstream of the projector -- so
by the time rollout() returns, everything is already in projected space.
The reference trajectory must therefore come from embeddings_h5["projected_emb"],
never embeddings_h5["encoder_cls_layers"] (that array is pre-projection and
is only used to FIT the probe / derive the steering vector).

ALSO IMPORTANT: one "timestep" of model.rollout()'s pixels/action_sequence is
NOT one raw env/dataset step. The model was trained with action_block-many
raw actions concatenated into a single chunked action vector per predictor
timestep (confirmed via stable_worldmodel.policy.WorldModelPolicy.get_action,
which reshapes a (.., horizon, action_dim*action_block) plan back into
(.., horizon*action_block, raw_action_dim) before stepping the real env one
raw action at a time -- i.e. action_block raw env steps per predictor step,
and the model's action_encoder.input_dim == action_block * raw_action_dim).
Pixels are likewise sampled once per predictor timestep, not every raw step.
infer_action_block() derives action_block from the model + dataset directly
rather than hardcoding it, since config/eval/pusht.yaml's action_block is a
*planning* config the open-loop path doesn't otherwise read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

from interp_utils.extract_pusht_encoder_cls_embeddings import preprocess_pixels
from interp_utils.steering.episode_pairs import EpisodePair, find_episode_pairs, load_initial_states
from interp_utils.steering.metrics import TrajectoryMetrics, summarize_trajectory, write_metrics_table
from interp_utils.steering.probe_io import ProbeSpec, best_layer_for_target, load_probe_dir
from interp_utils.steering.steered_model import attach_steering, detach_steering
from interp_utils.steering.steering_math import steering_vector


def episode_frame_indices(h5_file: h5py.File, episode_id: int) -> np.ndarray:
    """All raw frame row indices for one episode, sorted by step_idx."""
    episode_idx = np.asarray(h5_file["episode_idx"][:])
    step_idx = np.asarray(h5_file["step_idx"][:])
    mask = episode_idx == episode_id
    idx = np.nonzero(mask)[0]
    return idx[np.argsort(step_idx[idx])]


def infer_action_block(model, raw_action_dim: int) -> int:
    """Number of consecutive raw actions the model's action_encoder expects
    concatenated into one predictor-timestep action vector. Derived from the
    model + dataset directly (model.action_encoder.patch_embed.in_channels
    == action_block * raw_action_dim) rather than hardcoding a config value.
    """
    chunked_action_dim = model.action_encoder.patch_embed.in_channels
    if chunked_action_dim % raw_action_dim != 0:
        raise ValueError(
            f"Model's action_encoder expects {chunked_action_dim}-dim actions, "
            f"not divisible by the dataset's raw action_dim={raw_action_dim}."
        )
    return chunked_action_dim // raw_action_dim


def run_one_pair(
    model,
    dataset_h5: h5py.File,
    embeddings_h5: h5py.File,
    pair: EpisodePair,
    probe: ProbeSpec,
    target_dim_index: int,
    delta_raw: float,
    history_frames: int,
    rollout_steps: int,
    action_block: int,
    history_size: int | None,
    device: torch.device,
) -> TrajectoryMetrics:
    """Steer the model along `probe`'s direction by `delta_raw` while
    replaying `pair.src_episode`'s real actions, and compare the resulting
    predicted-embedding trajectory against `pair.ref_episode`'s real
    embeddings at the same relative (predictor-timestep) offsets.

    `history_frames`/`rollout_steps` are in predictor-timestep units; each
    predictor timestep spans `action_block` raw env/dataset steps (see
    infer_action_block / the module docstring).
    """
    src_idx_all = episode_frame_indices(dataset_h5, pair.src_episode)
    max_predictor_steps = len(src_idx_all) // action_block
    total_steps = min(history_frames + rollout_steps, max_predictor_steps)
    if total_steps <= history_frames:
        raise ValueError(
            f"src episode {pair.src_episode} has only {len(src_idx_all)} raw frames "
            f"({max_predictor_steps} predictor-steps at action_block={action_block}), "
            f"not enough for history_frames={history_frames}."
        )
    n_steps = total_steps - history_frames

    history_raw_idx = src_idx_all[[i * action_block for i in range(history_frames)]]
    pixels_raw = torch.from_numpy(np.asarray(dataset_h5["pixels"][history_raw_idx]))
    pixels = preprocess_pixels(pixels_raw, device)  # (H, C, h, w)
    pixels_info = pixels.unsqueeze(0).unsqueeze(0)  # (B=1, S=1, H, C, h, w)

    raw_action_idx = src_idx_all[: total_steps * action_block]
    raw_actions = np.asarray(dataset_h5["action"][raw_action_idx])
    raw_action_dim = raw_actions.shape[1]
    chunked_actions = raw_actions.reshape(total_steps, action_block * raw_action_dim)
    action_sequence = (
        torch.from_numpy(chunked_actions).float().to(device).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, T, action_block * raw_action_dim)

    vector = steering_vector(probe, target_dim_index, delta_raw)
    attach_steering(model, probe.layer_index, vector)
    try:
        info = {"pixels": pixels_info}
        with torch.inference_mode():
            out = model.rollout(info, action_sequence, history_size=history_size)
    finally:
        detach_steering(model)

    predicted = out["predicted_emb"][0, 0].detach().cpu().numpy()  # (H + n_steps + 1, D)

    needed_len = predicted.shape[0]
    ref_idx_all = episode_frame_indices(embeddings_h5, pair.ref_episode)
    max_ref_predictor_steps = len(ref_idx_all) // action_block
    if max_ref_predictor_steps < needed_len:
        raise ValueError(
            f"ref episode {pair.ref_episode} has only {len(ref_idx_all)} raw frames "
            f"({max_ref_predictor_steps} predictor-steps at action_block={action_block}), "
            f"need {needed_len} to compare against the predicted trajectory."
        )
    ref_strided_idx = ref_idx_all[[i * action_block for i in range(needed_len)]]
    reference = np.asarray(embeddings_h5["projected_emb"][ref_strided_idx])

    return summarize_trajectory(predicted, reference)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-loop activation-steering experiment on the PushT world model."
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
    parser.add_argument("--target-dim-index", type=int, default=0, help="Index into the target's columns, e.g. 0 for block_angle's single dim.")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--layer", default=None, help="Probe name, e.g. layer_07. Default: best layer for --target per probing.py's metrics.csv.")
    parser.add_argument("--delta", type=float, required=True, help="Target-unit delta to steer by.")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--history-frames", type=int, default=1, help="Number of real pixel frames to encode at rollout start, in predictor-timestep units.")
    parser.add_argument("--rollout-steps", type=int, default=5, help="Predictor-timestep units; the shortest PushT episodes only support ~9 predictor-steps at action_block=5.")
    parser.add_argument("--history-size", type=int, default=None, help="Predictor lookback window; default resolves from the model's own predictor.num_frames.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/steering/open_loop"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Resolve layer/vector/pairs and exit before loading the model or any frame data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_probe_dir(args.probe_dir)
    layer_name = args.layer or best_layer_for_target(args.probe_dir / "metrics.csv", args.target)
    probe = bundle.get(probe_name=layer_name, seed=args.probe_seed)

    with h5py.File(args.dataset, "r") as dataset_h5:
        episode_ids, initial_states = load_initial_states(dataset_h5)

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

    print(f"Resolved layer={layer_name!r} for target={args.target!r}")
    print(f"Found {len(pairs)} episode pair(s) (requested up to {args.num_pairs})")
    for pair in pairs[:5]:
        print(
            f"  src={pair.src_episode} ref={pair.ref_episode} "
            f"achieved_delta={pair.achieved_delta} error={pair.delta_error:.3f}"
        )

    if args.dry_run:
        print("--dry-run: exiting before loading the model or any frame data.")
        return

    if not pairs:
        raise SystemExit(
            "No episode pairs found -- widen --tolerance or pick a different --delta."
        )

    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    rows = []
    with h5py.File(args.dataset, "r") as dataset_h5, h5py.File(args.embeddings, "r") as embeddings_h5:
        action_block = infer_action_block(model, raw_action_dim=dataset_h5["action"].shape[1])
        print(f"Inferred action_block={action_block} from model.action_encoder vs dataset action_dim")

        for pair in pairs:
            try:
                result = run_one_pair(
                    model,
                    dataset_h5,
                    embeddings_h5,
                    pair,
                    probe,
                    target_dim_index=args.target_dim_index,
                    delta_raw=args.delta,
                    history_frames=args.history_frames,
                    rollout_steps=args.rollout_steps,
                    action_block=action_block,
                    history_size=args.history_size,
                    device=device,
                )
            except ValueError as exc:
                print(f"skipping pair src={pair.src_episode} ref={pair.ref_episode}: {exc}")
                continue

            row = {
                "src_episode": pair.src_episode,
                "ref_episode": pair.ref_episode,
                "delta_error": pair.delta_error,
                "layer": layer_name,
                "target": args.target,
                "requested_delta": args.delta,
                **asdict(result),
            }
            rows.append(row)
            print(
                f"src={pair.src_episode} ref={pair.ref_episode}: "
                f"mse_mean={result.mse_mean:.4f} cosine_sim_mean={result.cosine_sim_mean:.4f}"
            )

    write_metrics_table(args.output_dir, rows)
    print(f"Wrote {len(rows)} run(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
