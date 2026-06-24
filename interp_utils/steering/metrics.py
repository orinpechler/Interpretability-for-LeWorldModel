"""Compare a predicted embedding trajectory against a reference trajectory.

Both `pred` and `ref` are (T, D) arrays (T timesteps, D embedding dims),
already in the same embedding space (see open_loop.py for why that matters:
predicted_emb lives in projected-embedding space, not raw per-layer CLS
space). Mirrors probing.py's dataclass -> CSV/JSON convention so Stage 4 can
reuse the same writer.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def mse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Mean squared error over D, per timestep. Shape (T,)."""
    return np.mean((pred - ref) ** 2, axis=-1)


def mae(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Mean absolute error over D, per timestep. Shape (T,)."""
    return np.mean(np.abs(pred - ref), axis=-1)


def per_dim_abs_error_stats(pred: np.ndarray, ref: np.ndarray) -> dict[str, np.ndarray]:
    """Max/min/mean absolute error over D, per timestep. Each value (T,)."""
    abs_err = np.abs(pred - ref)
    return {
        "max": np.max(abs_err, axis=-1),
        "min": np.min(abs_err, axis=-1),
        "mean": np.mean(abs_err, axis=-1),
    }


def cosine_similarity(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Shape (T,). NaN where either row is zero."""
    pred_norm = np.linalg.norm(pred, axis=-1)
    ref_norm = np.linalg.norm(ref, axis=-1)
    denom = pred_norm * ref_norm
    dot = np.sum(pred * ref, axis=-1)
    return np.divide(dot, denom, out=np.full_like(dot, np.nan, dtype=np.float64), where=denom > 0)


def magnitude_diff(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """||pred_t|| - ||ref_t||, per timestep. Shape (T,)."""
    return np.linalg.norm(pred, axis=-1) - np.linalg.norm(ref, axis=-1)


@dataclass
class TrajectoryMetrics:
    # per-timestep (length T), JSON-serializable as lists
    mse_t: list[float]
    mae_t: list[float]
    max_abs_err_t: list[float]
    min_abs_err_t: list[float]
    mean_abs_err_t: list[float]
    cosine_sim_t: list[float]
    magnitude_diff_t: list[float]
    # trajectory-level aggregates
    mse_mean: float
    mse_last: float
    mae_mean: float
    mae_last: float
    cosine_sim_mean: float
    cosine_sim_last: float
    magnitude_diff_mean: float
    magnitude_diff_last: float


def summarize_trajectory(pred: np.ndarray, ref: np.ndarray) -> TrajectoryMetrics:
    """Compute the full metrics battery for one (pred, ref) trajectory pair."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if pred.ndim != 2:
        raise ValueError(f"Expected (T, D) arrays, got shape {pred.shape}")

    mse_t = mse(pred, ref)
    mae_t = mae(pred, ref)
    dim_stats = per_dim_abs_error_stats(pred, ref)
    cosine_t = cosine_similarity(pred, ref)
    magnitude_t = magnitude_diff(pred, ref)

    return TrajectoryMetrics(
        mse_t=mse_t.tolist(),
        mae_t=mae_t.tolist(),
        max_abs_err_t=dim_stats["max"].tolist(),
        min_abs_err_t=dim_stats["min"].tolist(),
        mean_abs_err_t=dim_stats["mean"].tolist(),
        cosine_sim_t=cosine_t.tolist(),
        magnitude_diff_t=magnitude_t.tolist(),
        mse_mean=float(np.mean(mse_t)),
        mse_last=float(mse_t[-1]),
        mae_mean=float(np.mean(mae_t)),
        mae_last=float(mae_t[-1]),
        cosine_sim_mean=float(np.nanmean(cosine_t)),
        cosine_sim_last=float(cosine_t[-1]),
        magnitude_diff_mean=float(np.mean(magnitude_t)),
        magnitude_diff_last=float(magnitude_t[-1]),
    )


def write_metrics_table(output_dir: Path, rows: list[dict], filename_stem: str = "metrics_by_run") -> None:
    """Write `rows` (plain dicts, e.g. asdict(TrajectoryMetrics) plus any run
    metadata merged in) to output_dir/{filename_stem}.json and .csv, mirroring
    probing.py's write_dict_rows/write_metrics convention.
    """
    if not rows:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{filename_stem}.json").write_text(json.dumps(rows, indent=2) + "\n")

    with (output_dir / f"{filename_stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "TrajectoryMetrics",
    "mse",
    "mae",
    "per_dim_abs_error_stats",
    "cosine_similarity",
    "magnitude_diff",
    "summarize_trajectory",
    "write_metrics_table",
]
