#!/usr/bin/env python
"""Train linear probes from PushT embeddings to agent position.

The split is done by episode, so all frames from a trajectory are assigned to
either train or test. This avoids evaluating on temporally adjacent frames from
episodes seen during probe training.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
except ImportError:
    hdf5plugin = None


DEFAULT_EMBEDDINGS = Path("stable-wm-data/embeddings/pusht_encoder_cls_fp32.h5")
DEFAULT_DATASET = Path("stable-wm-data/datasets/pusht_expert_train.h5")
DEFAULT_OUTPUT_DIR = Path("probes/agent_position")
STD_EPS = 1e-12


@dataclass
class ProbeMetrics:
    probe: str
    train_frames: int
    test_frames: int
    mse: float
    rmse: float
    mse_x: float
    mse_y: float
    rmse_x: float
    rmse_y: float
    r: float
    r_x: float
    r_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-layer linear probes to predict agent x/y position."
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for probe fitting/evaluation.",
    )
    parser.add_argument(
        "--no-save-probes",
        action="store_true",
        help="Only write metrics and split metadata, not probe weights.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")


def make_episode_split(
    episode_idx: np.ndarray,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = np.unique(episode_idx)
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)

    num_train = int(round(train_frac * len(shuffled)))
    num_train = min(max(num_train, 1), len(shuffled) - 1)
    train_episodes = np.sort(shuffled[:num_train])
    test_episodes = np.sort(shuffled[num_train:])
    train_mask = np.isin(episode_idx, train_episodes)
    return train_episodes, test_episodes, train_mask


def iter_slices(num_rows: int, chunk_size: int):
    for start in range(0, num_rows, chunk_size):
        yield start, min(start + chunk_size, num_rows)


def read_features(
    embeddings_h5: h5py.File,
    probe_name: str,
    layer_index: int | None,
    start: int,
    end: int,
) -> np.ndarray:
    if probe_name == "projected_emb":
        return np.asarray(embeddings_h5["projected_emb"][start:end], dtype=np.float64)
    if layer_index is None:
        raise ValueError("layer_index is required for encoder layer probes.")
    return np.asarray(
        embeddings_h5["encoder_cls_layers"][start:end, layer_index, :],
        dtype=np.float64,
    )


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64, device=device)


def standardize_array(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (array - mean) / std


def moment_stats(
    sum_values: np.ndarray,
    sum_squares: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    count: int,
) -> dict[str, list[float]]:
    mean = sum_values / count
    var = (sum_squares / count) - (mean * mean)
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std < STD_EPS, 1.0, std)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "range": (max_values - min_values).tolist(),
        "min": min_values.tolist(),
        "max": max_values.tolist(),
    }


def feature_stats(
    embeddings_h5: h5py.File,
    train_mask: np.ndarray,
    probe_name: str,
    layer_index: int | None,
    chunk_size: int,
) -> dict[str, list[float]]:
    if probe_name == "projected_emb":
        feature_dim = int(embeddings_h5["projected_emb"].shape[1])
    else:
        feature_dim = int(embeddings_h5["encoder_cls_layers"].shape[2])

    count = 0
    sum_x = np.zeros(feature_dim, dtype=np.float64)
    sum_x_sq = np.zeros(feature_dim, dtype=np.float64)
    min_x = np.full(feature_dim, np.inf, dtype=np.float64)
    max_x = np.full(feature_dim, -np.inf, dtype=np.float64)

    for start, end in iter_slices(train_mask.shape[0], chunk_size):
        mask = train_mask[start:end]
        if not mask.any():
            continue
        features = read_features(embeddings_h5, probe_name, layer_index, start, end)[mask]
        count += features.shape[0]
        sum_x += np.sum(features, axis=0)
        sum_x_sq += np.sum(features * features, axis=0)
        min_x = np.minimum(min_x, np.min(features, axis=0))
        max_x = np.maximum(max_x, np.max(features, axis=0))

    return moment_stats(sum_x, sum_x_sq, min_x, max_x, count)


def fit_linear_probe(
    embeddings_h5: h5py.File,
    states: h5py.Dataset,
    train_mask: np.ndarray,
    probe_name: str,
    layer_index: int | None,
    chunk_size: int,
    device: torch.device,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    num_rows = train_mask.shape[0]
    if probe_name == "projected_emb":
        feature_dim = int(embeddings_h5["projected_emb"].shape[1])
    else:
        feature_dim = int(embeddings_h5["encoder_cls_layers"].shape[2])

    xtx = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64, device=device)
    xty = torch.zeros((feature_dim + 1, 2), dtype=torch.float64, device=device)

    for start, end in iter_slices(num_rows, chunk_size):
        mask = train_mask[start:end]
        if not mask.any():
            continue
        features_np = read_features(embeddings_h5, probe_name, layer_index, start, end)[mask]
        targets_np = np.asarray(states[start:end, :2], dtype=np.float64)[mask]
        features = to_device(standardize_array(features_np, feature_mean, feature_std), device)
        targets = to_device(standardize_array(targets_np, target_mean, target_std), device)
        ones = torch.ones((features.shape[0], 1), dtype=torch.float64, device=device)
        design = torch.cat((features, ones), dim=1)
        xtx.add_(design.T @ design)
        xty.add_(design.T @ targets)

    return torch.linalg.solve(xtx, xty).cpu().numpy()


def pearson_r(sum_true, sum_pred, sum_true_sq, sum_pred_sq, sum_true_pred, count):
    numerator = sum_true_pred - (sum_true * sum_pred / count)
    true_var = sum_true_sq - (sum_true * sum_true / count)
    pred_var = sum_pred_sq - (sum_pred * sum_pred / count)
    denominator = np.sqrt(true_var * pred_var)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def evaluate_probe(
    embeddings_h5: h5py.File,
    states: h5py.Dataset,
    test_mask: np.ndarray,
    weights: np.ndarray,
    probe_name: str,
    layer_index: int | None,
    chunk_size: int,
    train_frames: int,
    device: torch.device,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> ProbeMetrics:
    num_rows = test_mask.shape[0]
    feature_dim = weights.shape[0] - 1
    weights_t = to_device(weights, device)
    sse = torch.zeros(2, dtype=torch.float64, device=device)
    sum_true = torch.zeros(2, dtype=torch.float64, device=device)
    sum_pred = torch.zeros(2, dtype=torch.float64, device=device)
    sum_true_sq = torch.zeros(2, dtype=torch.float64, device=device)
    sum_pred_sq = torch.zeros(2, dtype=torch.float64, device=device)
    sum_true_pred = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0

    for start, end in iter_slices(num_rows, chunk_size):
        mask = test_mask[start:end]
        if not mask.any():
            continue
        features_np = read_features(embeddings_h5, probe_name, layer_index, start, end)[mask]
        targets_np = np.asarray(states[start:end, :2], dtype=np.float64)[mask]
        features = to_device(standardize_array(features_np, feature_mean, feature_std), device)
        targets = to_device(standardize_array(targets_np, target_mean, target_std), device)
        preds = features @ weights_t[:feature_dim] + weights_t[feature_dim]

        residual = preds - targets
        sse.add_(torch.sum(residual * residual, dim=0))
        sum_true.add_(torch.sum(targets, dim=0))
        sum_pred.add_(torch.sum(preds, dim=0))
        sum_true_sq.add_(torch.sum(targets * targets, dim=0))
        sum_pred_sq.add_(torch.sum(preds * preds, dim=0))
        sum_true_pred.add_(torch.sum(targets * preds, dim=0))
        count += targets.shape[0]

    sse = sse.cpu().numpy()
    sum_true = sum_true.cpu().numpy()
    sum_pred = sum_pred.cpu().numpy()
    sum_true_sq = sum_true_sq.cpu().numpy()
    sum_pred_sq = sum_pred_sq.cpu().numpy()
    sum_true_pred = sum_true_pred.cpu().numpy()

    r_xy = pearson_r(sum_true, sum_pred, sum_true_sq, sum_pred_sq, sum_true_pred, count)

    flat_count = count * 2
    flat_r = pearson_r(
        np.array([sum_true.sum()]),
        np.array([sum_pred.sum()]),
        np.array([sum_true_sq.sum()]),
        np.array([sum_pred_sq.sum()]),
        np.array([sum_true_pred.sum()]),
        flat_count,
    )[0]

    mse_xy = sse / count
    rmse_xy = np.sqrt(mse_xy)
    return ProbeMetrics(
        probe=probe_name,
        train_frames=train_frames,
        test_frames=count,
        mse=float(mse_xy.mean()),
        rmse=float(np.sqrt(mse_xy.mean())),
        mse_x=float(mse_xy[0]),
        mse_y=float(mse_xy[1]),
        rmse_x=float(rmse_xy[0]),
        rmse_y=float(rmse_xy[1]),
        r=float(flat_r),
        r_x=float(r_xy[0]),
        r_y=float(r_xy[1]),
    )


def target_stats(
    states: h5py.Dataset,
    mask: np.ndarray,
    chunk_size: int,
) -> dict[str, list[float]]:
    count = 0
    sum_y = np.zeros(2, dtype=np.float64)
    sum_y_sq = np.zeros(2, dtype=np.float64)
    min_y = np.full(2, np.inf, dtype=np.float64)
    max_y = np.full(2, -np.inf, dtype=np.float64)

    for start, end in iter_slices(mask.shape[0], chunk_size):
        chunk_mask = mask[start:end]
        if not chunk_mask.any():
            continue
        targets = np.asarray(states[start:end, :2], dtype=np.float64)[chunk_mask]
        count += targets.shape[0]
        sum_y += np.sum(targets, axis=0)
        sum_y_sq += np.sum(targets * targets, axis=0)
        min_y = np.minimum(min_y, np.min(targets, axis=0))
        max_y = np.maximum(max_y, np.max(targets, axis=0))

    mean_y = sum_y / count
    stats = moment_stats(sum_y, sum_y_sq, min_y, max_y, count)
    stats["mean"] = mean_y.tolist()
    return stats


def write_metrics(output_dir: Path, metrics: list[ProbeMetrics]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(metric) for metric in metrics]
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")

    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    print(f"Using torch device: {device}", flush=True)

    with h5py.File(args.embeddings, "r") as embeddings_h5, h5py.File(args.dataset, "r") as dataset_h5:
        num_frames = int(embeddings_h5["encoder_cls_layers"].shape[0])
        num_layers = int(embeddings_h5["encoder_cls_layers"].shape[1])
        if num_layers != 12:
            print(f"Found {num_layers} encoder layers; training one probe per layer.")
        if int(dataset_h5["state"].shape[0]) != num_frames:
            raise ValueError("Embeddings and source dataset have different frame counts.")

        if "completed" in embeddings_h5 and not bool(np.asarray(embeddings_h5["completed"][:]).all()):
            raise ValueError("Embedding file has incomplete rows according to the completed mask.")

        episode_idx = np.asarray(embeddings_h5["episode_idx"][:])
        train_episodes, test_episodes, train_mask = make_episode_split(
            episode_idx,
            args.train_frac,
            args.seed,
        )
        test_mask = ~train_mask
        train_frames = int(train_mask.sum())
        test_frames = int(test_mask.sum())

        split_info = {
            "split": "episode",
            "seed": args.seed,
            "train_frac_requested": args.train_frac,
            "normalization": "features and targets standardized with train split mean/std",
            "num_episodes": int(len(train_episodes) + len(test_episodes)),
            "num_train_episodes": int(len(train_episodes)),
            "num_test_episodes": int(len(test_episodes)),
            "num_frames": num_frames,
            "num_train_frames": train_frames,
            "num_test_frames": test_frames,
            "train_frame_frac": train_frames / num_frames,
            "test_frame_frac": test_frames / num_frames,
        }
        train_target_stats = target_stats(dataset_h5["state"], train_mask, args.chunk_size)
        test_target_stats = target_stats(dataset_h5["state"], test_mask, args.chunk_size)
        target_mean = np.asarray(train_target_stats["mean"], dtype=np.float64)
        target_std = np.asarray(train_target_stats["std"], dtype=np.float64)
        normalized_test_range = (
            (
                np.asarray(test_target_stats["max"], dtype=np.float64)
                - np.asarray(test_target_stats["min"], dtype=np.float64)
            )
            / target_std
        )
        normalized_test_std = np.asarray(test_target_stats["std"], dtype=np.float64) / target_std
        split_info["train_target_mean"] = train_target_stats["mean"]
        split_info["train_target_std"] = train_target_stats["std"]
        split_info["test_target_std"] = test_target_stats["std"]
        split_info["test_target_range"] = test_target_stats["range"]
        split_info["test_target_min"] = test_target_stats["min"]
        split_info["test_target_max"] = test_target_stats["max"]
        split_info["normalized_test_target_std"] = normalized_test_std.tolist()
        split_info["normalized_test_target_range"] = normalized_test_range.tolist()
        (args.output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n")
        np.savez_compressed(
            args.output_dir / "episode_split.npz",
            train_episodes=train_episodes,
            test_episodes=test_episodes,
        )
        print(f"Raw test target std [x, y]: {test_target_stats['std']}", flush=True)
        print(f"Raw test target range [x, y]: {test_target_stats['range']}", flush=True)
        print(f"Normalized test target std [x, y]: {normalized_test_std.tolist()}", flush=True)
        print(f"Normalized test target range [x, y]: {normalized_test_range.tolist()}", flush=True)

        probe_specs = [(f"layer_{layer:02d}", layer) for layer in range(num_layers)]
        probe_specs.append(("projected_emb", None))

        metrics: list[ProbeMetrics] = []
        saved_weights = {
            "target_mean": target_mean,
            "target_std": target_std,
        }
        states = dataset_h5["state"]
        for probe_name, layer_index in probe_specs:
            print(f"Training {probe_name}...", flush=True)
            stats = feature_stats(
                embeddings_h5=embeddings_h5,
                train_mask=train_mask,
                probe_name=probe_name,
                layer_index=layer_index,
                chunk_size=args.chunk_size,
            )
            feature_mean = np.asarray(stats["mean"], dtype=np.float64)
            feature_std = np.asarray(stats["std"], dtype=np.float64)
            weights = fit_linear_probe(
                embeddings_h5=embeddings_h5,
                states=states,
                train_mask=train_mask,
                probe_name=probe_name,
                layer_index=layer_index,
                chunk_size=args.chunk_size,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_mean=target_mean,
                target_std=target_std,
            )
            metric = evaluate_probe(
                embeddings_h5=embeddings_h5,
                states=states,
                test_mask=test_mask,
                weights=weights,
                probe_name=probe_name,
                layer_index=layer_index,
                chunk_size=args.chunk_size,
                train_frames=train_frames,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_mean=target_mean,
                target_std=target_std,
            )
            metrics.append(metric)
            saved_weights[probe_name] = weights
            saved_weights[f"{probe_name}_feature_mean"] = feature_mean
            saved_weights[f"{probe_name}_feature_std"] = feature_std
            print(
                f"{probe_name}: normalized test MSE={metric.mse:.6g}, RMSE={metric.rmse:.6g}, "
                f"R={metric.r:.6g} (RMSE_x={metric.rmse_x:.6g}, "
                f"RMSE_y={metric.rmse_y:.6g}, R_x={metric.r_x:.6g}, R_y={metric.r_y:.6g})",
                flush=True,
            )

        write_metrics(args.output_dir, metrics)
        if not args.no_save_probes:
            np.savez_compressed(args.output_dir / "linear_probe_weights.npz", **saved_weights)

    print(f"Wrote probe outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
