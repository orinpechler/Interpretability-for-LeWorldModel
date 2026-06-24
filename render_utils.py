"""Render a PushT RGB frame from an explicit simulator state.

PushT states in this project use the column order:
``[agent_x, agent_y, block_x, block_y, block_angle]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


PUSHT_STATE_DIM = 5


def make_pusht_state(
    *,
    block_position: Sequence[float] | np.ndarray,
    block_angle: float,
    agent_position: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build a PushT state vector from semantic state components."""
    agent_position = np.asarray(agent_position, dtype=np.float32)
    block_position = np.asarray(block_position, dtype=np.float32)

    if agent_position.shape != (2,):
        raise ValueError(f"agent_position must have shape (2,), got {agent_position.shape}")
    if block_position.shape != (2,):
        raise ValueError(f"block_position must have shape (2,), got {block_position.shape}")

    return np.array(
        [
            agent_position[0],
            agent_position[1],
            block_position[0],
            block_position[1],
            block_angle,
        ],
        dtype=np.float32,
    )


def render_pusht_state(
    *,
    block_position: Sequence[float] | np.ndarray,
    block_angle: float,
    agent_position: Sequence[float] | np.ndarray,
    env: Any | None = None,
    image_shape: tuple[int, int] = (224, 224),
    env_kwargs: dict[str, Any] | None = None,
    render_kwargs: dict[str, Any] | None = None,
    reset: bool = True,
    close: bool = True,
) -> np.ndarray:
    """Render one PushT frame for a requested state.

    Args:
        block_position: ``(x, y)`` block position.
        block_angle: Block angle in radians, using PushT's native convention.
        agent_position: ``(x, y)`` agent position.
        env: Optional existing PushT environment. If omitted, one is created.
        image_shape: Image size passed to a newly created environment.
        env_kwargs: Extra keyword arguments for environment construction.
        render_kwargs: Extra keyword arguments for ``env.render``.
        reset: Reset the environment before setting the state.
        close: Close an internally created environment before returning.

    Returns:
        RGB frame as a ``uint8`` NumPy array with shape ``(H, W, 3)``.
    """
    state = make_pusht_state(
        block_position=block_position,
        block_angle=block_angle,
        agent_position=agent_position,
    )

    owns_env = env is None
    if env is None:
        env = make_pusht_env(image_shape=image_shape, **(env_kwargs or {}))

    try:
        if reset:
            _call_reset(env)
        _set_pusht_state(env, state)
        frame = _render_frame(env, **(render_kwargs or {}))
    finally:
        if owns_env and close:
            env.close()

    return _as_rgb_uint8(frame)


def make_pusht_env(image_shape: tuple[int, int] = (224, 224), **kwargs: Any) -> Any:
    """Create a PushT environment through stable-worldmodel/gymnasium."""
    try:
        from stable_worldmodel.envs.pusht.env import PushT

        return PushT(image_shape=image_shape, **kwargs)
    except ModuleNotFoundError as exc:
        if exc.name != "stable_worldmodel":
            raise
    except TypeError:
        from stable_worldmodel.envs.pusht.env import PushT

        return PushT(**kwargs)

    try:
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rendering PushT states requires stable-worldmodel. Install the repo "
            "dependencies, then call render_pusht_state again."
        ) from exc

    return gym.make("swm/PushT-v1", image_shape=image_shape, **kwargs)


def _call_reset(env: Any) -> None:
    try:
        env.reset()
    except TypeError:
        env.reset(seed=None)


def _set_pusht_state(env: Any, state: np.ndarray) -> None:
    target = getattr(env, "unwrapped", env)
    for method_name in ("_set_state", "set_state"):
        if not hasattr(target, method_name):
            continue

        method = getattr(target, method_name)
        try:
            method(state=state)
        except TypeError:
            method(state)
        return

    raise AttributeError("PushT environment does not expose _set_state or set_state.")


def _render_frame(env: Any, **render_kwargs: Any) -> Any:
    target = getattr(env, "unwrapped", env)

    if hasattr(env, "render"):
        try:
            return env.render(**render_kwargs)
        except TypeError:
            if "mode" not in render_kwargs:
                return env.render(mode="rgb_array", **render_kwargs)
            raise

    if hasattr(target, "_render_frame"):
        return target._render_frame(**render_kwargs)

    raise AttributeError("PushT environment does not expose render or _render_frame.")


def _as_rgb_uint8(frame: Any) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA frame, got shape {frame.shape}")

    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if frame.max(initial=0.0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0)

    return frame.astype(np.uint8, copy=False)
