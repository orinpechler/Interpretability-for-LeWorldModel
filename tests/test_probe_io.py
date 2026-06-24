"""Smoke test for interp_utils/steering/probe_io.py.

No torch, no model, no GPU. Run directly: python tests/test_probe_io.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from interp_utils.steering.probe_io import best_layer_for_target, load_probe_dir


def write_fake_probe_dir(probe_dir: Path, *, seeds=(0, 1), num_layers=3, d=8, k=1) -> None:
    split = {
        "target": "block_angle",
        "target_columns": [4],
        "target_names": ["block_angle"],
        "base_seed": seeds[0],
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "splits": [],
    }
    (probe_dir / "split.json").write_text(json.dumps(split))

    rng = np.random.default_rng(0)
    arrays = {}
    for seed in seeds:
        arrays[f"seed_{seed}_target_mean"] = rng.normal(size=k)
        arrays[f"seed_{seed}_target_std"] = np.abs(rng.normal(size=k)) + 0.5
        for layer in range(num_layers):
            name = f"layer_{layer:02d}"
            arrays[f"seed_{seed}_{name}"] = rng.normal(size=(d + 1, k))
            arrays[f"seed_{seed}_{name}_feature_mean"] = rng.normal(size=d)
            arrays[f"seed_{seed}_{name}_feature_std"] = np.abs(rng.normal(size=d)) + 0.5
        arrays[f"seed_{seed}_projected_emb"] = rng.normal(size=(d + 1, k))
        arrays[f"seed_{seed}_projected_emb_feature_mean"] = rng.normal(size=d)
        arrays[f"seed_{seed}_projected_emb_feature_std"] = np.abs(rng.normal(size=d)) + 0.5
    np.savez(probe_dir / "linear_probe_weights.npz", **arrays)

    rows = []
    rmse_by_layer = {f"layer_{i:02d}": float(10 - i) for i in range(num_layers)}  # layer_02 is "best" (lowest)
    for name, rmse in rmse_by_layer.items():
        rows.append({"target": "block_angle", "probe": name, "rmse_mean": rmse})
    rows.append({"target": "block_angle", "probe": "projected_emb", "rmse_mean": 0.01})
    with (probe_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "probe", "rmse_mean"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_probe_dir_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp)
        write_fake_probe_dir(probe_dir)
        bundle = load_probe_dir(probe_dir)

        assert bundle.target == "block_angle"
        assert bundle.target_columns == [4]
        assert set(bundle.available_seeds()) == {0, 1}
        assert "layer_00" in bundle.available_probe_names()
        assert "projected_emb" in bundle.available_probe_names()

        spec = bundle.get(probe_name="layer_01", seed=0)
        assert spec.layer_index == 1
        assert spec.weight.shape == (9, 1)

        spec_proj = bundle.get(probe_name="projected_emb", seed=0)
        assert spec_proj.layer_index is None


def test_missing_files_raise_actionable_error():
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp)
        try:
            load_probe_dir(probe_dir)
        except FileNotFoundError as exc:
            assert "linear_probe_weights.npz" in str(exc)
            assert "split.json" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError for empty probe_dir")


def test_unknown_probe_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp)
        write_fake_probe_dir(probe_dir)
        bundle = load_probe_dir(probe_dir)
        try:
            bundle.get(probe_name="layer_99", seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unknown probe name")


def test_best_layer_for_target():
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp)
        write_fake_probe_dir(probe_dir, num_layers=3)
        best = best_layer_for_target(probe_dir / "metrics.csv", target="block_angle")
        # rmse_by_layer = {layer_00: 10, layer_01: 9, layer_02: 8} -> layer_02 is lowest
        assert best == "layer_02"
        # projected_emb (rmse 0.01) must be excluded from "best layer" selection
        assert best != "projected_emb"


if __name__ == "__main__":
    test_load_probe_dir_round_trip()
    test_missing_files_raise_actionable_error()
    test_unknown_probe_name_raises()
    test_best_layer_for_target()
    print("test_probe_io.py: all checks passed")
