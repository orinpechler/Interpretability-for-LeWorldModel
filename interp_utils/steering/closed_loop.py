#!/usr/bin/env python
"""Closed-loop activation-steering experiment: real PushT physics via
CEMSolver/WorldModelPolicy/World, with optional baseline-vs-steered
comparison videos.

Reuses eval.py's already-undecorated helpers (img_transform, get_dataset,
get_episodes_length) directly -- eval.py is not edited, since @hydra.main
only wraps run(), so importing these plain functions has no side effects.

Hydra config (not argparse, unlike open_loop.py): this script needs
cfg.world / cfg.eval / cfg.solver / cfg.plan_config exactly like eval.py
does to construct swm.World / CEMSolver, so it reuses eval.py's own Hydra
config tree (config/eval/pusht.yaml) via config/eval/pusht_steering.yaml,
which extends it with a `steering` block. Invoke like:

    python -m interp_utils.steering.closed_loop \\
        steering.probe_dir=/path/to/probes/block_angle steering.delta=0.3

The goal-vs-init scoping subtlety (get_cost's self.encode(goal) must stay
unperturbed while self.rollout(...)'s internal encode must be perturbed) is
already solved by steered_model.SteeredLeWM -- no new mechanism here, this
module only wires that mechanism into the real env/solver stack.

Semantic note: world.evaluate's `goal` image comes from the SRC episode's
own future frame (start_step + goal_offset), not the matched REF episode.
This is unmodified eval.py/World behavior and is intentional here: the
question being tested is "does perturbing the agent's *perceived* current
state make its real, physically-simulated behavior resemble what it would
do if it perceived ref's actual state," not "what if its goal changed."
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

from eval import get_dataset, img_transform
from interp_utils.steering.episode_pairs import EpisodePair, find_episode_pairs, load_initial_states
from interp_utils.steering.probe_io import ProbeBundle, ProbeSpec, best_layer_for_target, load_probe_dir
from interp_utils.steering.steered_model import attach_steering, detach_steering
from interp_utils.steering.steering_math import steering_vector


def select_episode_pairs(cfg: DictConfig, probe: ProbeSpec) -> list[EpisodePair]:
    """Mine (src, ref) episode pairs for the closed-loop run, filtering out
    any whose src episode is too short for the configured goal_offset_steps.
    """
    with h5py.File(cfg.steering.dataset_path, "r") as dataset_h5:
        episode_ids, initial_states = load_initial_states(dataset_h5)
        ep_len = np.asarray(dataset_h5["ep_len"][:])

    delta_target = np.zeros(len(probe.target_columns))
    delta_target[cfg.steering.target_dim_index] = cfg.steering.delta

    # Over-fetch candidates since some will be filtered out by episode length.
    candidates = find_episode_pairs(
        episode_ids,
        initial_states,
        probe.target_columns,
        delta_target,
        cfg.steering.tolerance,
        max_pairs=cfg.steering.num_pairs * 5,
    )

    min_len = cfg.eval.goal_offset_steps + 1
    pairs = [p for p in candidates if ep_len[p.src_episode] > min_len]
    return pairs[: cfg.steering.num_pairs]


def build_steered_solver(cfg: DictConfig, probe: ProbeSpec, device: str):
    """Load the model, configure (but leave disabled) the steering hook, and
    construct the CEMSolver around it.

    Returns (model, solver). Steering starts disabled -- callers toggle it
    via attach_steering/detach_steering per mode (baseline/steered), reusing
    the same model/solver objects for both passes.
    """
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    vector = steering_vector(probe, cfg.steering.target_dim_index, cfg.steering.delta)
    attach_steering(model, probe.layer_index, vector)
    detach_steering(model)  # configured, but off until a "steered" pass requests it

    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return model, solver


def build_process_and_transform(cfg: DictConfig, dataset) -> tuple[dict, dict]:
    """Mirrors eval.py's run()'s process/transform construction exactly."""
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    process: dict = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    return process, transform


def resolve_probe(cfg: DictConfig) -> tuple[ProbeSpec, str]:
    bundle: ProbeBundle = load_probe_dir(Path(cfg.steering.probe_dir))
    layer_name = cfg.steering.layer or best_layer_for_target(
        Path(cfg.steering.probe_dir) / "metrics.csv", cfg.steering.target
    )
    probe = bundle.get(probe_name=layer_name, seed=cfg.steering.probe_seed)
    return probe, layer_name


@hydra.main(version_base=None, config_path="../../config/eval", config_name="pusht_steering")
def run(cfg: DictConfig) -> None:
    assert cfg.policy != "random", "Steering requires a trained policy (set policy=pusht/lewm)."
    assert cfg.steering.mode in ("baseline", "steered", "both"), cfg.steering.mode
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    probe, layer_name = resolve_probe(cfg)
    print(f"Resolved steering layer={layer_name!r} for target={cfg.steering.target!r}")

    pairs = select_episode_pairs(cfg, probe)
    print(f"Found {len(pairs)} usable episode pair(s) (requested up to {cfg.steering.num_pairs})")
    for pair in pairs[:5]:
        print(
            f"  src={pair.src_episode} ref={pair.ref_episode} "
            f"achieved_delta={pair.achieved_delta} error={pair.delta_error:.4f}"
        )

    if cfg.steering.dry_run:
        print("steering.dry_run=true: exiting before loading the model, world, or solver.")
        return

    if not pairs:
        raise SystemExit(
            "No usable episode pairs found -- widen steering.tolerance, pick a "
            "different steering.delta, or check steering.dataset_path."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    cfg.world.num_envs = len(pairs)  # must match episodes_idx length below
    world = swm.World(**cfg.world, image_shape=(224, 224))

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process, transform = build_process_and_transform(cfg, dataset)

    model, solver = build_steered_solver(cfg, probe, device)
    config = swm.PlanConfig(**cfg.plan_config)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )
    world.set_policy(policy)

    src_episodes = [pair.src_episode for pair in pairs]
    start_steps = [0] * len(pairs)  # always start at the mined initial state

    video_root = (
        Path(cfg.steering.video_dir)
        if cfg.steering.video_dir
        else Path(swm.data.utils.get_cache_dir(), cfg.policy).parent / "steering"
    )
    video_root.mkdir(parents=True, exist_ok=True)

    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    results = {}

    if cfg.steering.mode in ("baseline", "both"):
        detach_steering(model)
        print("Running baseline (unsteered) rollout...")
        baseline_dir = video_root / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        results["baseline"] = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=src_episodes,
            callables=callables,
            video=baseline_dir,
        )

    if cfg.steering.mode in ("steered", "both"):
        vector = steering_vector(probe, cfg.steering.target_dim_index, cfg.steering.delta)
        attach_steering(model, probe.layer_index, vector)
        print("Running steered rollout...")
        steered_dir = video_root / "steered"
        steered_dir.mkdir(parents=True, exist_ok=True)
        results["steered"] = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=src_episodes,
            callables=callables,
            video=steered_dir,
        )

    print(results)

    results_path = video_root / "steering_results.txt"
    with results_path.open("a") as handle:
        handle.write("\n==== CONFIG ====\n")
        handle.write(OmegaConf.to_yaml(cfg))
        handle.write("\n==== PAIRS ====\n")
        for pair in pairs:
            handle.write(
                f"src={pair.src_episode} ref={pair.ref_episode} "
                f"achieved_delta={pair.achieved_delta.tolist()} error={pair.delta_error}\n"
            )
        handle.write("\n==== RESULTS ====\n")
        handle.write(f"{results}\n")
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    run()
