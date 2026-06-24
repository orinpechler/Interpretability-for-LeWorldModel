"""Smoke test for interp_utils/steering/steered_model.py.

Uses a tiny synthetic encoder shaped like the real HF ViTModel
(`.encoder.layer` ModuleList, forward(pixels, interpolate_pos_encoding=True)
-> object with `.last_hidden_state`) instead of the real ViT, so this runs
in milliseconds with no checkpoint needed. Tests the actual mechanism the
plan flagged as the key subtlety: the hook must perturb only the CLS token
of the init/current-frame encode inside rollout(), and must NOT affect a
plain encode() call made outside rollout() (which is what get_cost() does
for the goal image) even while steering is attached/enabled.

Run directly: python tests/test_steered_model_hook.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from interp_utils.steering.steered_model import attach_steering, detach_steering


D = 4  # embed dim
NUM_LAYERS = 3
NUM_TOKENS = 2  # CLS + 1 patch token


class ToyBlock(nn.Module):
    """Stands in for a real ViTLayer: returns a plain tensor (matching the
    pinned transformers<5 ViTLayer.forward contract), purely additive so the
    expected effect of a hook is exactly predictable.
    """

    def __init__(self, dim: int, layer_id: int):
        super().__init__()
        # a real parameter so next(encoder.parameters()) (used by LeWM.encode) works
        self.scale = nn.Parameter(torch.full((dim,), float(layer_id + 1)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.scale


class ToyViTEncoderInner(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList([ToyBlock(dim, i) for i in range(num_layers)])


class ToyViTModel(nn.Module):
    """Mimics model.encoder: model.encoder.encoder.layer[i] is the hookable
    ModuleList, forward(pixels, interpolate_pos_encoding=True) returns an
    object with .last_hidden_state, shape (batch, num_tokens, dim).
    """

    def __init__(self, dim: int, num_layers: int, num_tokens: int):
        super().__init__()
        self.encoder = ToyViTEncoderInner(dim, num_layers)
        self.dim = dim
        self.num_tokens = num_tokens

    def forward(self, pixels: torch.Tensor, interpolate_pos_encoding: bool = True):
        batch = pixels.shape[0]
        hidden_states = torch.zeros(batch, self.num_tokens, self.dim, dtype=pixels.dtype)
        for block in self.encoder.layer:
            hidden_states = block(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states)


class IdentityPredictor(nn.Module):
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return x


def make_toy_lewm():
    from stable_worldmodel.wm.lewm import LeWM

    return LeWM(
        encoder=ToyViTModel(D, NUM_LAYERS, NUM_TOKENS),
        predictor=IdentityPredictor(),
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )


def make_rollout_inputs(batch: int = 1, samples: int = 1, history: int = 1, n_steps: int = 2):
    pixels = torch.zeros(batch, samples, history, 1, 1, 1)
    action_sequence = torch.zeros(batch, samples, history + n_steps, D)
    return {"pixels": pixels}, action_sequence


def baseline_cls_embedding() -> torch.Tensor:
    """The CLS-token embedding an unsteered encode() produces for the toy
    encoder: sum of each ToyBlock's `scale` parameter (purely additive).
    """
    return sum(float(i + 1) for i in range(NUM_LAYERS)) * torch.ones(D)


def test_unsteered_rollout_matches_expected():
    model = make_toy_lewm()
    info, action_sequence = make_rollout_inputs()
    out = model.rollout(dict(info), action_sequence, history_size=1)

    cls_emb_t0 = out["predicted_emb"][0, 0, 0]  # (B=0, S=0, t=0, :)
    assert torch.allclose(cls_emb_t0, baseline_cls_embedding())


def test_steering_perturbs_only_cls_token_of_init_encode():
    model = make_toy_lewm()
    vector = torch.tensor([10.0, -10.0, 5.0, 0.0])
    layer_index = 1  # inject after the 2nd block; must propagate through block 2 onward

    attach_steering(model, layer_index=layer_index, vector=vector)

    info, action_sequence = make_rollout_inputs()
    out = model.rollout(dict(info), action_sequence, history_size=1)
    steered_cls = out["predicted_emb"][0, 0, 0]

    # Toy blocks are purely additive, so the injected vector survives
    # unchanged through every later block -- exact expected value.
    expected = baseline_cls_embedding() + vector
    assert torch.allclose(steered_cls, expected), (steered_cls, expected)


def test_steering_does_not_affect_plain_encode_call():
    """This is the goal-vs-init scoping property: get_cost() calls
    self.encode(goal) directly (not via rollout()), and that call must stay
    unperturbed even while steering is attached and enabled.
    """
    model = make_toy_lewm()
    vector = torch.tensor([10.0, -10.0, 5.0, 0.0])
    attach_steering(model, layer_index=1, vector=vector)

    goal_info = {"pixels": torch.zeros(1, 1, 1, 1, 1)}  # (b, t, c, h, w), no S dim here
    goal_out = model.encode(dict(goal_info))
    goal_cls = goal_out["emb"][0, 0]

    assert torch.allclose(goal_cls, baseline_cls_embedding()), (
        "encode() called outside rollout() must NOT be perturbed by the hook"
    )


def test_detach_steering_restores_unsteered_behavior():
    model = make_toy_lewm()
    vector = torch.tensor([10.0, -10.0, 5.0, 0.0])
    attach_steering(model, layer_index=1, vector=vector)
    detach_steering(model)

    info, action_sequence = make_rollout_inputs()
    out = model.rollout(dict(info), action_sequence, history_size=1)
    cls_emb = out["predicted_emb"][0, 0, 0]
    assert torch.allclose(cls_emb, baseline_cls_embedding())


def test_hook_is_removed_after_each_rollout_call():
    """The hook must not leak: after one steered rollout() call, the layer
    should have zero forward hooks registered (HookHandle removes on exit).
    """
    model = make_toy_lewm()
    vector = torch.tensor([1.0, 1.0, 1.0, 1.0])
    attach_steering(model, layer_index=0, vector=vector)

    info, action_sequence = make_rollout_inputs()
    model.rollout(dict(info), action_sequence, history_size=1)

    layer = model.encoder.encoder.layer[0]
    assert len(layer._forward_hooks) == 0, "hook was not removed after rollout() returned"


def test_invalid_layer_index_raises():
    model = make_toy_lewm()
    try:
        attach_steering(model, layer_index=NUM_LAYERS, vector=torch.zeros(D))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range layer_index")


if __name__ == "__main__":
    test_unsteered_rollout_matches_expected()
    test_steering_perturbs_only_cls_token_of_init_encode()
    test_steering_does_not_affect_plain_encode_call()
    test_detach_steering_restores_unsteered_behavior()
    test_hook_is_removed_after_each_rollout_call()
    test_invalid_layer_index_raises()
    print("test_steered_model_hook.py: all checks passed")
