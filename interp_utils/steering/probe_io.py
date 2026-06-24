"""Load linear-probe weights produced by interp_utils/probing.py.

Expects a probe directory containing the three files probing.py writes
when run without --no-save-probes: split.json, linear_probe_weights.npz,
and metrics.csv (the seed-aggregated metrics table).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPLIT_FILENAME = "split.json"
WEIGHTS_FILENAME = "linear_probe_weights.npz"
METRICS_FILENAME = "metrics.csv"

_LAYER_PROBE_RE = re.compile(r"^layer_(\d+)$")


@dataclass(frozen=True)
class ProbeSpec:
    """A single trained linear probe, fully resolved for one seed."""

    target: str
    target_columns: list[int]
    target_names: list[str]
    probe_name: str  # e.g. "layer_07" or "projected_emb"
    layer_index: int | None  # 0-11 for ViT-layer probes, None for projected_emb
    seed: int
    weight: np.ndarray  # (D+1, K) -- last row is the bias
    feature_mean: np.ndarray  # (D,)
    feature_std: np.ndarray  # (D,)
    target_mean: np.ndarray  # (K,)
    target_std: np.ndarray  # (K,)


class ProbeBundle:
    """Parsed contents of a probe directory: split metadata + raw npz arrays."""

    def __init__(self, split: dict, weights_npz: dict[str, np.ndarray]):
        self._split = split
        self._npz = weights_npz

    @property
    def target(self) -> str:
        return self._split["target"]

    @property
    def target_columns(self) -> list[int]:
        return list(self._split["target_columns"])

    @property
    def target_names(self) -> list[str]:
        return list(self._split["target_names"])

    def available_seeds(self) -> list[int]:
        return list(self._split["seeds"])

    def available_probe_names(self) -> list[str]:
        seed = self.available_seeds()[0]
        prefix = f"seed_{seed}_"
        suffixes = ("_feature_mean", "_feature_std")
        names = []
        for key in self._npz:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if rest in ("target_mean", "target_std"):
                continue
            if rest.endswith(suffixes):
                continue
            names.append(rest)
        return sorted(names)

    def get(self, *, probe_name: str, seed: int) -> ProbeSpec:
        if seed not in self.available_seeds():
            raise ValueError(
                f"Seed {seed} not found in probe bundle. Available seeds: "
                f"{self.available_seeds()}"
            )

        prefix = f"seed_{seed}_"
        weight_key = f"{prefix}{probe_name}"
        feature_mean_key = f"{weight_key}_feature_mean"
        feature_std_key = f"{weight_key}_feature_std"
        target_mean_key = f"{prefix}target_mean"
        target_std_key = f"{prefix}target_std"

        missing = [
            k
            for k in (weight_key, feature_mean_key, feature_std_key, target_mean_key, target_std_key)
            if k not in self._npz
        ]
        if missing:
            raise ValueError(
                f"Probe '{probe_name}' (seed {seed}) is missing expected array(s) "
                f"{missing} in {WEIGHTS_FILENAME}. Available probe names: "
                f"{self.available_probe_names()}"
            )

        match = _LAYER_PROBE_RE.match(probe_name)
        layer_index = int(match.group(1)) if match else None

        return ProbeSpec(
            target=self.target,
            target_columns=self.target_columns,
            target_names=self.target_names,
            probe_name=probe_name,
            layer_index=layer_index,
            seed=seed,
            weight=self._npz[weight_key],
            feature_mean=self._npz[feature_mean_key],
            feature_std=self._npz[feature_std_key],
            target_mean=self._npz[target_mean_key],
            target_std=self._npz[target_std_key],
        )


def load_probe_dir(probe_dir: Path) -> ProbeBundle:
    """Load split.json + linear_probe_weights.npz from probe_dir.

    Raises FileNotFoundError naming the exact missing path(s) and how to
    produce them, rather than failing deep inside numpy/json with an
    unhelpful traceback.
    """
    probe_dir = Path(probe_dir)
    split_path = probe_dir / SPLIT_FILENAME
    weights_path = probe_dir / WEIGHTS_FILENAME

    missing = [p for p in (split_path, weights_path) if not p.exists()]
    if missing:
        missing_str = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing probe output file(s):\n  {missing_str}\n"
            "Run interp_utils/probing.py (without --no-save-probes) and "
            f"point --probe-dir at the directory containing its output, "
            f"or copy/move that output into {probe_dir}."
        )

    split = json.loads(split_path.read_text())
    with np.load(weights_path) as npz:
        weights_npz = {k: npz[k] for k in npz.files}

    bundle = ProbeBundle(split, weights_npz)

    expected_seeds = set(split["seeds"])
    npz_seeds = {
        int(m.group(1))
        for key in weights_npz
        if (m := re.match(r"^seed_(\d+)_", key))
    }
    if not expected_seeds.issubset(npz_seeds):
        raise ValueError(
            f"{WEIGHTS_FILENAME} is missing data for seed(s) "
            f"{sorted(expected_seeds - npz_seeds)} declared in {SPLIT_FILENAME}. "
            "The two files appear to be out of sync -- re-run probing.py."
        )

    return bundle


def best_layer_for_target(
    metrics_csv: Path,
    target: str,
    metric: str = "rmse_mean",
    minimize: bool = True,
) -> str:
    """Return the probe_name (e.g. "layer_07") with the best `metric` value
    for `target`, among rows whose probe matches "layer_<N>" (excludes
    "projected_emb", since callers asking for a default *layer* want a ViT
    block index to hook into, not the post-projector embedding).
    """
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_csv}\n"
            "Run interp_utils/probing.py first, or pass --layer explicitly "
            "instead of relying on the best-layer default."
        )

    best_probe: str | None = None
    best_value: float | None = None
    with metrics_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target"] != target:
                continue
            if not _LAYER_PROBE_RE.match(row["probe"]):
                continue
            if metric not in row:
                raise ValueError(
                    f"Metric '{metric}' not found in {metrics_csv}. "
                    f"Available columns: {list(row.keys())}"
                )
            value = float(row[metric])
            if best_value is None or (value < best_value if minimize else value > best_value):
                best_value = value
                best_probe = row["probe"]

    if best_probe is None:
        raise ValueError(
            f"No layer probe rows found for target '{target}' in {metrics_csv}."
        )
    return best_probe
