#!/usr/bin/env python
"""Compare probe-steered embeddings with embeddings from re-rendered PushT states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters for PushT data
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from interp_utils.extract_pusht_encoder_cls_embeddings import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DEFAULT_WEIGHTS,
    preprocess_pixels,
)
from interp_utils._steering import (
    DEFAULT_LAYER,
    DEFAULT_PROBE_PATH,
    DEFAULT_SEED,
    load_block_position_probe,
    steer_encoder_cls,
)
from render_utils import make_pusht_env, render_pusht_state_vector


STATE_AGENT = slice(0, 2)
STATE_BLOCK = slice(2, 4)
STATE_ANGLE = 4


def clean_config(section: dict) -> dict:
    return {key: value for key, value in section.items() if not key.startswith("_")}


def remap_hf_vit_keys_for_current_transformers(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Map older HF ViT checkpoint keys to newer Transformers ViT key names."""
    model_keys = model.state_dict().keys()
    if not any(key.startswith("encoder.layers.") for key in model_keys):
        return state_dict
    if not any(key.startswith("encoder.encoder.layer.") for key in state_dict):
        return state_dict

    replacements = (
        ("encoder.encoder.layer.", "encoder.layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    )

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def load_lewm_model(config_path: Path, weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load LeWM while tolerating stable_pretraining ViT implementation drift."""
    from interp_utils.extract_pusht_encoder_cls_embeddings import load_model

    try:
        return load_model(config_path, weights_path, device)
    except RuntimeError as exc:
        message = str(exc)
        if "encoder.encoder.layer" not in message or "encoder.layers" not in message:
            raise

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from transformers import ViTConfig, ViTModel

    cfg = json.loads(config_path.read_text())
    encoder_cfg = clean_config(cfg["encoder"])
    if encoder_cfg["size"] != "tiny":
        raise ValueError(f"Fallback loader only knows ViT-tiny, got {encoder_cfg['size']!r}")

    vit_config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=encoder_cfg["image_size"],
        patch_size=encoder_cfg["patch_size"],
    )
    encoder = ViTModel(vit_config, add_pooling_layer=False)

    def make_mlp(name: str) -> MLP:
        mlp_cfg = clean_config(cfg[name])
        return MLP(
            input_dim=mlp_cfg["input_dim"],
            output_dim=mlp_cfg["output_dim"],
            hidden_dim=mlp_cfg["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**clean_config(cfg["predictor"])),
        action_encoder=Embedder(**clean_config(cfg["action_encoder"])),
        projector=make_mlp("projector"),
        pred_proj=make_mlp("pred_proj"),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = remap_hf_vit_keys_for_current_transformers(state_dict, model)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For PushT frames, compare projected CLS embeddings from probe steering "
            "against projected CLS embeddings from synthetic frames rendered with "
            "the same block-position displacement."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--probe-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--probe-layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--delta",
        type=float,
        default=10.0,
        help="Block-position displacement applied equally to x and y.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=100, help="Use 0 to process all selected frames.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--eval-valid-only",
        action="store_true",
        help="Select rows valid as eval.py starts: step_idx <= ep_len - goal_offset_steps - 1.",
    )
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--output-images",
        action="store_true",
        help="Write original/synthetic side-by-side PNGs to synthetic_frames/.",
    )
    parser.add_argument("--image-output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Skip frames that fail to render instead of stopping at the first error.",
    )
    return parser.parse_args()


def select_indices(
    *,
    num_frames: int,
    start_index: int,
    stride: int,
    max_frames: int,
) -> np.ndarray:
    if start_index < 0 or start_index >= num_frames:
        raise ValueError(f"start-index must be in [0, {num_frames}), got {start_index}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if max_frames < 0:
        raise ValueError(f"max-frames must be nonnegative, got {max_frames}")

    indices = np.arange(start_index, num_frames, stride, dtype=np.int64)
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def select_eval_valid_indices(
    handle: h5py.File,
    *,
    goal_offset_steps: int,
    start_index: int,
    stride: int,
    max_frames: int,
) -> np.ndarray:
    if goal_offset_steps < 0:
        raise ValueError(f"goal-offset-steps must be nonnegative, got {goal_offset_steps}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if max_frames < 0:
        raise ValueError(f"max-frames must be nonnegative, got {max_frames}")

    episode_idx = np.asarray(handle["episode_idx"])
    step_idx = np.asarray(handle["step_idx"])
    ep_len = np.asarray(handle["ep_len"])
    max_start_per_row = ep_len[episode_idx] - goal_offset_steps - 1
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0].astype(np.int64)
    valid_indices = valid_indices[valid_indices >= start_index][::stride]
    if max_frames > 0:
        valid_indices = valid_indices[:max_frames]
    return valid_indices


def delta_path_component(delta: float) -> str:
    text = f"{delta:g}"
    return text.replace("-", "neg").replace(".", "p")


def preprocess_single_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(frame)).unsqueeze(0)
    return preprocess_pixels(pixels, device)


def encode_projected(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    output = model.encoder(pixels, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])


def write_comparison_image(
    output_dir: Path,
    *,
    frame_index: int,
    original_frame: np.ndarray,
    synthetic_frame: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_frame = np.asarray(original_frame)[..., :3].astype(np.uint8, copy=False)
    synthetic_frame = np.asarray(synthetic_frame)[..., :3].astype(np.uint8, copy=False)
    comparison = np.concatenate([original_frame, synthetic_frame], axis=1)
    iio.imwrite(output_dir / f"frame_{frame_index:06d}_original_vs_synthetic.png", comparison)

    abs_diff = np.abs(
        original_frame.astype(np.int16) - synthetic_frame.astype(np.int16)
    ).astype(np.uint8)
    amplified_diff = np.clip(abs_diff.astype(np.uint16) * 8, 0, 255).astype(np.uint8)
    diff_panel = np.concatenate(
        [original_frame, synthetic_frame, abs_diff, amplified_diff],
        axis=1,
    )
    iio.imwrite(output_dir / f"frame_{frame_index:06d}_render_diff.png", diff_panel)


def compare_embeddings(
    steered: torch.Tensor,
    synthetic: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, float | int]:
    steered = steered.detach().float().cpu()
    synthetic = synthetic.detach().float().cpu()
    diff = steered - synthetic
    abs_diff = diff.abs()

    return {
        "cosine_similarity": float(F.cosine_similarity(steered, synthetic, dim=0).item()),
        "l2_distance": float(torch.linalg.vector_norm(diff).item()),
        "mse": float(torch.mean(diff.square()).item()),
        "mae": float(torch.mean(abs_diff).item()),
        "max_abs_error": float(torch.max(abs_diff).item()),
        "same_dims_epsilon": int((abs_diff <= epsilon).sum().item()),
        "same_dims_fraction": float((abs_diff <= epsilon).float().mean().item()),
        "sign_agreement_fraction": float((torch.sign(steered) == torch.sign(synthetic)).float().mean().item()),
    }


def summarize(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    metric_names = [name for name in rows[0] if name != "frame_index"]
    summary: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def write_outputs(
    output_path: Path,
    *,
    rows: list[dict[str, float | int]],
    summary: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "dataset": str(args.dataset),
            "weights": str(args.weights),
            "config": str(args.config),
            "probe_path": str(args.probe_path),
            "probe_seed": args.probe_seed,
            "probe_layer": args.probe_layer,
            "delta": args.delta,
            "epsilon": args.epsilon,
            "eval_valid_only": args.eval_valid_only,
            "goal_offset_steps": args.goal_offset_steps,
        },
        "summary": summary,
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    delta_component = delta_path_component(args.delta)
    if args.output is None:
        args.output = Path(f"metrics_delta_{delta_component}.json")
    if args.image_output_dir is None:
        args.image_output_dir = Path(f"synthetic_frames_delta_{delta_component}.py")
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    model = load_lewm_model(args.config, args.weights, device)
    probe = load_block_position_probe(
        args.probe_path,
        seed=args.probe_seed,
        layer=args.probe_layer,
        device=device,
    )

    rows: list[dict[str, float | int]] = []
    skipped: list[tuple[int, str]] = []

    with h5py.File(args.dataset, "r") as handle:
        pixels_ds = handle["pixels"]
        state_ds = handle["state"]
        if args.eval_valid_only:
            indices = select_eval_valid_indices(
                handle,
                goal_offset_steps=args.goal_offset_steps,
                start_index=args.start_index,
                stride=args.stride,
                max_frames=args.max_frames,
            )
        else:
            indices = select_indices(
                num_frames=int(pixels_ds.shape[0]),
                start_index=args.start_index,
                stride=args.stride,
                max_frames=args.max_frames,
            )
        image_shape = tuple(int(dim) for dim in pixels_ds.shape[1:3])

        env = make_pusht_env(image_shape=image_shape)
        try:
            for n, index in enumerate(indices, start=1):
                original_frame = np.asarray(pixels_ds[index])
                pixels = preprocess_single_frame(original_frame, device)
                state = np.asarray(state_ds[index], dtype=np.float32)
                synthetic_state = state.copy()
                synthetic_state[STATE_BLOCK] += np.array(
                    [args.delta, args.delta],
                    dtype=np.float32,
                )

                try:
                    synthetic_frame = render_pusht_state_vector(
                        synthetic_state,
                        env=env,
                        reset=True,
                        close=False,
                        step_after_set=True,
                    )
                except Exception as exc:
                    if not args.keep_going:
                        raise
                    skipped.append((int(index), repr(exc)))
                    continue

                synthetic_pixels = preprocess_single_frame(synthetic_frame, device)
                if args.output_images:
                    write_comparison_image(
                        args.image_output_dir,
                        frame_index=int(index),
                        original_frame=original_frame,
                        synthetic_frame=synthetic_frame,
                    )

                with torch.inference_mode():
                    steered_emb = steer_encoder_cls(
                        model,
                        pixels,
                        args.delta,
                        args.delta,
                        probe=probe,
                        layer=args.probe_layer,
                    )[0]
                    synthetic_emb = encode_projected(model, synthetic_pixels)[0]

                row = {
                    "frame_index": int(index),
                    **compare_embeddings(steered_emb, synthetic_emb, epsilon=args.epsilon),
                }
                rows.append(row)

                if n == 1 or n % 25 == 0:
                    print(
                        f"processed {n}/{len(indices)} frames; "
                        f"latest cosine={row['cosine_similarity']:.6f}",
                        flush=True,
                    )
        finally:
            env.close()

    if not rows:
        raise RuntimeError(f"No frames were processed successfully. Skipped: {skipped[:5]}")

    summary = summarize(rows)

    print("\n=== Summary ===")
    print(f"frames_processed: {len(rows)}")
    print(f"frames_skipped: {len(skipped)}")
    for name, stats in summary.items():
        print(
            f"{name}: mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
            f"min={stats['min']:.6g}, max={stats['max']:.6g}"
        )

    if skipped:
        print("\n=== Skipped Frames ===")
        for index, error in skipped[:10]:
            print(f"{index}: {error}")

    if args.output is not None:
        write_outputs(args.output, rows=rows, summary=summary, args=args)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
