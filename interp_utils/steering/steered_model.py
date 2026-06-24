"""Activation steering for stable_worldmodel.wm.lewm.LeWM.

LeWM.encode() reads only the CLS token of its ViT encoder's final hidden
state (model.encoder.encoder.layer[-1]'s output, after ViTModel's own final
layernorm). Patch tokens are never read anywhere downstream. So "steer at
layer L" means: add a vector to the CLS-token position of
model.encoder.encoder.layer[L]'s output, and let it propagate forward
through the remaining blocks naturally (a plain torch forward hook).

The subtlety this module resolves: LeWM.get_cost() calls self.encode(goal)
(must stay UNPERTURBED -- it's the planner's target, not the observation
being steered) and then self.rollout(...) (which internally calls
self.encode() on the current/init frame -- this IS what we want to
perturb). Since get_cost() is inherited unmodified from LeWM, and we only
override rollout() here, the goal-encode is never touched by the hook and
the init-encode always is -- with zero edits to the installed package.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
from einops import rearrange
from stable_worldmodel.wm.lewm import LeWM
from torch import nn


def make_additive_hook(vector: torch.Tensor, cls_token_index: int = 0) -> Callable:
    """Build a forward-hook fn for nn.Module.register_forward_hook that adds
    `vector` (D,) to the CLS-token position of the hidden-state output.

    Handles a plain-tensor output (the current transformers<5 ViTLayer
    contract) as well as tuple/object outputs defensively, in case that
    contract ever changes.
    """

    def hook(module: nn.Module, inputs: tuple, output):
        if isinstance(output, torch.Tensor):
            modified = output.clone()
            modified[:, cls_token_index, :] = modified[:, cls_token_index, :] + vector
            return modified
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified = hidden_states.clone()
            modified[:, cls_token_index, :] = modified[:, cls_token_index, :] + vector
            return (modified, *output[1:])
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = output.last_hidden_state.clone()
            output.last_hidden_state[:, cls_token_index, :] += vector
            return output
        raise TypeError(
            f"make_additive_hook: unsupported forward-hook output type {type(output)}"
        )

    return hook


class HookHandle:
    """RAII wrapper: register a forward hook on __enter__, remove it on
    __exit__, so scoping a perturbation to a single call is a `with` block
    rather than bespoke enable/disable flags scattered through rollout().
    """

    def __init__(self, module: nn.Module, hook_fn: Callable):
        self._module = module
        self._hook_fn = hook_fn
        self._handle = None

    def __enter__(self) -> "HookHandle":
        self._handle = self._module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc_info) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class SteeredLeWM(LeWM):
    """Drop-in LeWM subclass. Carries no extra construction state of its
    own -- `attach_steering` sets plain instance attributes after swapping
    `model.__class__`, so the class object itself needs no new __init__.

    Overrides ONLY rollout() -- copied verbatim from LeWM.rollout
    (stable_worldmodel/wm/lewm/lewm.py), including the `if 'emb' not in
    info:` caching guard, with the single change that the `self.encode(_init)`
    call is wrapped in `self._steering_hook_scope()`. get_cost() is
    inherited unmodified.
    """

    def rollout(self, info: dict, action_sequence: torch.Tensor, history_size: int | None = None) -> dict:
        if history_size is None:
            history_size = getattr(self.predictor, "num_frames", 3)

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # encode initial state, or reuse cached embedding from a prior rollout.
        if "emb" not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            with self._steering_hook_scope():
                _init = self.encode(_init)
            info["emb"] = _init["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_flat = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )  # (BS, T, A_emb)

        # rollout predictor autoregressively for n_steps + 1 (final) steps
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + n_steps + 1, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def _steering_hook_scope(self):
        if not getattr(self, "_steer_enabled", False):
            return contextlib.nullcontext()
        layer = self.encoder.encoder.layer[self._steer_layer]
        hook_fn = make_additive_hook(self._steer_vector)
        return HookHandle(layer, hook_fn)


def attach_steering(model: LeWM, layer_index: int, vector) -> LeWM:
    """Swap model.__class__ to SteeredLeWM (mutates in place -- the same
    object identity is preserved, so existing references, e.g. inside an
    already-constructed CEMSolver, see the change), and configure the
    perturbation. Idempotent: calling again with a new layer/vector just
    updates the attributes.
    """
    num_layers = len(model.encoder.encoder.layer)
    if not 0 <= layer_index < num_layers:
        raise ValueError(f"layer_index {layer_index} out of range [0, {num_layers}).")

    if model.__class__ is not SteeredLeWM:
        model.__class__ = SteeredLeWM

    param = next(model.parameters())
    model._steer_layer = layer_index
    model._steer_vector = torch.as_tensor(vector, dtype=param.dtype, device=param.device)
    model._steer_enabled = True
    return model


def detach_steering(model: LeWM) -> LeWM:
    """Disable steering. Does NOT revert __class__ back to LeWM: with
    _steer_enabled=False, SteeredLeWM.rollout behaves byte-identically to
    LeWM.rollout (the hook scope degrades to a no-op nullcontext), so
    reverting the class buys nothing and adds a failure mode (forgetting to
    revert before reusing `model` elsewhere).
    """
    model._steer_enabled = False
    return model
