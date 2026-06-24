#!/usr/bin/env python
"""Quick sklearn-style regression probe on the CONCATENATED multi-layer CLS
representation (all 12 layers stacked into one 2304-dim feature vector):
does combining layers beat the single best layer?

Consistent with the rest of this package's lightweight, sklearn-based "is
there something here" scripts (binary_separability.py,
orientation_classification.py, goal_relative_orientation.py) -- a fast,
throwaway check we train ourselves, not an extension of the heavier
teammate-authored probing.py pipeline (torch closed-form OLS, episode
splits, writes to probes/<target>/).

I/O: same chunk-aligned block-sampling trick as the other scripts here
(encoder_cls_layers chunks are (1024, 12, 192)) -- reads contiguous,
chunk-aligned blocks instead of scattering across the whole ~21GB array.

Existing best single-layer R (from probes/<target>/metrics.csv, for
comparison): block_angle layer_09 R=0.933, block_position layer_09
R=0.996, agent_position layer_10 R=0.984 (layer_09 is NOT agent_position's
best layer -- R=0.801 there).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET_COLUMNS = {
    "agent_position": [0, 1],
    "block_position": [2, 3],
    "block_angle": [4],
}


def sample_blocks(
    dataset_path: Path,
    embeddings_path: Path,
    target_columns: list[int],
    starts: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features_list, targets_list = [], []
    with h5py.File(dataset_path, "r") as fs, h5py.File(embeddings_path, "r") as fe:
        for start in starts:
            end = start + block_size
            layers = np.asarray(fe["encoder_cls_layers"][start:end, :, :])
            features_list.append(layers.reshape(layers.shape[0], -1))
            targets_list.append(np.asarray(fs["state"][start:end, target_columns]))
    return np.concatenate(features_list), np.concatenate(targets_list)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        return float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])
    return float(np.mean([np.corrcoef(y_true[:, k], y_pred[:, k])[0, 1] for k in range(y_true.shape[1])]))


def fit_and_evaluate(model, x_train, y_train, x_test, y_test) -> tuple[float, float, float]:
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    r2 = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return r2, rmse, pearson_r(y_test, preds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regress PushT state targets from the concatenated all-layer CLS representation."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("/scratch-shared/orinxAI/stable-wm-data/datasets/pusht_expert_train.h5")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("/scratch-shared/orinxAI/embeddings/pusht_encoder_cls_fp32.h5")
    )
    parser.add_argument("--target", choices=tuple(TARGET_COLUMNS.keys()), default="block_angle")
    parser.add_argument("--num-blocks", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength.")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=None,
        help="If given, also sweep PCA(n_components) + LinearRegression (fit on train only) "
        "at each n_components, alongside the raw-feature Ridge baseline. "
        "E.g. --pca-components 10 50 100 300 600 1200",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_columns = TARGET_COLUMNS[args.target]

    with h5py.File(args.dataset, "r") as f:
        total_rows = f["state"].shape[0]

    rng = np.random.default_rng(args.seed)
    num_slots = total_rows // args.block_size
    starts = np.sort(rng.choice(num_slots, size=args.num_blocks, replace=False)) * args.block_size
    print(f"Sampling {args.num_blocks} contiguous blocks of {args.block_size} rows from {total_rows} total rows")

    features, targets = sample_blocks(args.dataset, args.embeddings, target_columns, starts, args.block_size)
    print(f"Loaded {len(features)} examples, concatenated feature dim={features.shape[1]}")

    train_idx, test_idx = train_test_split(np.arange(len(features)), test_size=args.test_frac, random_state=args.seed)

    feature_mean, feature_std = features[train_idx].mean(axis=0), features[train_idx].std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    target_mean, target_std = targets[train_idx].mean(axis=0), targets[train_idx].std(axis=0)
    target_std[target_std < 1e-8] = 1.0

    x_train = (features[train_idx] - feature_mean) / feature_std
    x_test = (features[test_idx] - feature_mean) / feature_std
    y_train = (targets[train_idx] - target_mean) / target_std
    y_test = (targets[test_idx] - target_mean) / target_std

    r2, rmse, r_mean = fit_and_evaluate(Ridge(alpha=args.alpha), x_train, y_train, x_test, y_test)
    print(
        f"\nconcat_layers, raw 2304-dim + Ridge ({args.target}): n_train={len(train_idx)} n_test={len(test_idx)} "
        f"R2={r2:.4f} RMSE(normalized)={rmse:.4f} mean_pearson_r={r_mean:.4f}"
    )

    if args.pca_components is not None:
        print(f"\nPCA(n_components) + LinearRegression sweep ({args.target}):")
        for n_components in args.pca_components:
            pca = PCA(n_components=n_components, random_state=args.seed)
            x_train_pca = pca.fit_transform(x_train)
            x_test_pca = pca.transform(x_test)
            explained = float(pca.explained_variance_ratio_.sum())

            pca_r2, pca_rmse, pca_r_mean = fit_and_evaluate(
                LinearRegression(), x_train_pca, y_train, x_test_pca, y_test
            )
            print(
                f"  n_components={n_components:5d}  explained_var={explained:.4f}  "
                f"R2={pca_r2:.4f}  RMSE(normalized)={pca_rmse:.4f}  mean_pearson_r={pca_r_mean:.4f}"
            )


if __name__ == "__main__":
    main()
