#!/usr/bin/env python
"""Sweep open-loop activation-steering experiments across (layer, delta,
episode-pair) combinations.

Drives interp_utils.steering.open_loop.run_one_pair as an importable library
call in one process, so the model/dataset/embeddings handles are loaded
once and reused across the whole sweep, rather than re-launching a fresh
process (and reloading the model) per (layer, delta) combination.

Closed-loop sweeping is intentionally out of scope here: each closed-loop
run is a full CEM search + real env rollout, expensive enough that sweeping
many (layer, delta) combinations that way isn't practical. Use
closed_loop.py directly for the handful of configs you actually want videos
for, informed by which layer/delta this sweep's open-loop metrics suggest
are most effective.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import stable_worldmodel as swm
import torch

from interp_utils.steering.episode_pairs import find_episode_pairs, load_initial_states
from interp_utils.steering.metrics import write_metrics_table
from interp_utils.steering.open_loop import infer_action_block, run_one_pair
from interp_utils.steering.probe_io import load_probe_dir

_LAYER_RE = re.compile(r"^layer_(\d+)$")


def layer_sort_key(layer_name: str) -> int:
    """Numeric layer index for sorting/plotting, e.g. "layer_07" -> 7."""
    match = _LAYER_RE.match(layer_name)
    if not match:
        raise ValueError(f"Not a 'layer_<N>' probe name: {layer_name!r}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep open-loop activation-steering experiments across layers/deltas."
    )
    parser.add_argument("--dataset", type=Path, default=Path("stable-wm-data/datasets/pusht_expert_train.h5"))
    parser.add_argument("--embeddings", type=Path, default=Path("stable-wm-data/embeddings/pusht_encoder_cls_fp32.h5"))
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        default="block_angle",
        choices=("agent_position", "block_position", "block_angle"),
    )
    parser.add_argument("--target-dim-index", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Probe names, e.g. layer_05 layer_07. Default: every available ViT-layer probe (excludes projected_emb).",
    )
    parser.add_argument("--deltas", nargs="+", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--num-pairs-per-config", type=int, default=3)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/steering/aggregate"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_probe_dir(args.probe_dir)
    layer_names = args.layers or [n for n in bundle.available_probe_names() if _LAYER_RE.match(n)]
    layer_names = sorted(layer_names, key=layer_sort_key)
    if not layer_names:
        raise SystemExit(f"No layer probes found in {args.probe_dir} (or none matched --layers).")

    with h5py.File(args.dataset, "r") as dataset_h5:
        episode_ids, initial_states = load_initial_states(dataset_h5)

    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    rows: list[dict] = []
    with h5py.File(args.dataset, "r") as dataset_h5, h5py.File(args.embeddings, "r") as embeddings_h5:
        action_block = infer_action_block(model, raw_action_dim=dataset_h5["action"].shape[1])
        print(f"Inferred action_block={action_block}; sweeping {len(layer_names)} layer(s) x {len(args.deltas)} delta(s)")

        for layer_name in layer_names:
            probe = bundle.get(probe_name=layer_name, seed=args.probe_seed)

            for delta in args.deltas:
                delta_target = np.zeros(len(probe.target_columns))
                delta_target[args.target_dim_index] = delta
                pairs = find_episode_pairs(
                    episode_ids,
                    initial_states,
                    probe.target_columns,
                    delta_target,
                    args.tolerance,
                    max_pairs=args.num_pairs_per_config,
                )
                print(f"  layer={layer_name} delta={delta}: {len(pairs)} pair(s) found")

                for pair in pairs:
                    try:
                        result = run_one_pair(
                            model,
                            dataset_h5,
                            embeddings_h5,
                            pair,
                            probe,
                            delta_vector=delta_target,
                            history_frames=args.history_frames,
                            rollout_steps=args.rollout_steps,
                            action_block=action_block,
                            history_size=args.history_size,
                            device=device,
                        )
                    except ValueError as exc:
                        print(f"    skipping src={pair.src_episode} ref={pair.ref_episode}: {exc}")
                        continue

                    rows.append(
                        {
                            "layer": layer_name,
                            "delta": delta,
                            "src_episode": pair.src_episode,
                            "ref_episode": pair.ref_episode,
                            "delta_error": pair.delta_error,
                            "target": args.target,
                            **asdict(result),
                        }
                    )

    write_metrics_table(args.output_dir, rows, filename_stem="sweep_results")
    print(
        f"Wrote {len(rows)} run(s) across {len(layer_names)} layer(s) x "
        f"{len(args.deltas)} delta(s) to {args.output_dir}"
    )

    if rows:
        df = pd.DataFrame(rows)
        plot_metric_vs_layer(df, "mse_mean", args.output_dir / "mse_mean_vs_layer.png")
        plot_metric_vs_delta(df, "mse_mean", args.output_dir / "mse_mean_vs_delta.png")


def load_sweep_results(output_dir: Path) -> pd.DataFrame:
    """Load a previously-written sweep_results.csv back into a DataFrame."""
    return pd.read_csv(Path(output_dir) / "sweep_results.csv")


def plot_metric_vs_layer(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """`metric` vs. layer index, one line per delta value, averaged over
    episode pairs (error bars = std across pairs).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["layer_idx"] = df["layer"].map(layer_sort_key)
    grouped = df.groupby(["delta", "layer_idx"])[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots()
    for delta, sub in grouped.groupby("delta"):
        sub = sub.sort_values("layer_idx")
        ax.errorbar(sub["layer_idx"], sub["mean"], yerr=sub["std"], marker="o", label=f"delta={delta}")
    ax.set_xlabel("ViT layer index")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs steering layer")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_vs_delta(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """`metric` vs. delta, one line per layer, averaged over episode pairs
    (error bars = std across pairs).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["layer_idx"] = df["layer"].map(layer_sort_key)
    grouped = df.groupby(["layer_idx", "delta"])[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots()
    for layer_idx, sub in grouped.groupby("layer_idx"):
        sub = sub.sort_values("delta")
        ax.errorbar(sub["delta"], sub["mean"], yerr=sub["std"], marker="o", label=f"layer={layer_idx}")
    ax.set_xlabel("delta")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs steering delta")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
