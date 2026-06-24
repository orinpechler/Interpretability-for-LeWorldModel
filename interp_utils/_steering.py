"""Steer PushT block-position information in encoder CLS activations.

The default configuration targets the best block-position probe used in the
analysis: seed 4, encoder layer 9, and the linear probe saved by probing.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULT_PROBE_PATH = Path("probes/block_position/linear_probe_weights.npz")
DEFAULT_SEED = 4
DEFAULT_LAYER = 9


@dataclass(frozen=True)
class LinearProbeSteering:
    """Linear probe parameters needed for coupled linear-probe steering."""

    weights: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    def predict(self, cls: torch.Tensor) -> torch.Tensor:
        """Predict raw block position from unstandardized CLS activations."""
        cls_std = (cls - self.feature_mean) / self.feature_std
        pred_std = cls_std @ self.weights[:-1] + self.weights[-1]
        return pred_std * self.target_std + self.target_mean

    def standardized_target_delta(self, delta_xy: torch.Tensor) -> torch.Tensor:
        delta_xy = torch.as_tensor(delta_xy, dtype=self.dtype, device=self.device)
        if delta_xy.shape[-1] != 2:
            raise ValueError(f"delta_xy must have final dimension 2, got {tuple(delta_xy.shape)}")
        return delta_xy / self.target_std

    def activation_delta(self, delta_xy: torch.Tensor) -> torch.Tensor:
        """Return the minimum-norm CLS delta that changes probe output by delta_xy."""
        target_delta_std = self.standardized_target_delta(delta_xy)
        probe_weights = self.weights[:-1]

        # In standardized activation space, solve the coupled linear system:
        # activation_delta_std @ W = target_delta_std.
        # The pseudoinverse gives the minimum-L2-norm activation_delta_std.
        activation_delta_std = target_delta_std @ torch.linalg.pinv(probe_weights)
        return activation_delta_std * self.feature_std

    def steer_cls(self, cls: torch.Tensor, delta_xy: torch.Tensor) -> torch.Tensor:
        return cls + self.activation_delta(delta_xy).to(dtype=cls.dtype, device=cls.device)


def load_block_position_probe(
    probe_path: str | Path = DEFAULT_PROBE_PATH,
    *,
    seed: int = DEFAULT_SEED,
    layer: int = DEFAULT_LAYER,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> LinearProbeSteering:
    """Load the seed/layer-specific block-position probe for steering."""
    probe_name = f"layer_{layer:02d}"
    prefix = f"seed_{seed}_{probe_name}"
    probe_path = Path(probe_path)
    target_std_key = f"seed_{seed}_target_std"
    required = [
        prefix,
        f"{prefix}_feature_mean",
        f"{prefix}_feature_std",
        f"seed_{seed}_target_mean",
        target_std_key,
    ]
    device = torch.device(device) if device is not None else torch.device("cpu")
    with np.load(probe_path) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{probe_path} is missing required probe arrays: {missing}")

        return LinearProbeSteering(
            weights=torch.as_tensor(payload[prefix], dtype=dtype, device=device),
            feature_mean=torch.as_tensor(payload[f"{prefix}_feature_mean"], dtype=dtype, device=device),
            feature_std=torch.as_tensor(payload[f"{prefix}_feature_std"], dtype=dtype, device=device),
            target_mean=torch.as_tensor(payload[f"seed_{seed}_target_mean"], dtype=dtype, device=device),
            target_std=torch.as_tensor(payload[target_std_key], dtype=dtype, device=device),
        )


def _encoder_blocks(encoder: nn.Module) -> nn.ModuleList:
    if hasattr(encoder, "vit") and hasattr(encoder.vit, "encoder"):
        return encoder.vit.encoder.layer
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    if hasattr(encoder, "layers"):
        return encoder.layers
    raise AttributeError("Could not find transformer blocks on encoder.")


def _move_probe_to_activation(
    probe: LinearProbeSteering,
    activation: torch.Tensor,
) -> LinearProbeSteering:
    return LinearProbeSteering(
        weights=probe.weights.to(device=activation.device, dtype=activation.dtype),
        feature_mean=probe.feature_mean.to(device=activation.device, dtype=activation.dtype),
        feature_std=probe.feature_std.to(device=activation.device, dtype=activation.dtype),
        target_mean=probe.target_mean.to(device=activation.device, dtype=activation.dtype),
        target_std=probe.target_std.to(device=activation.device, dtype=activation.dtype),
    )


def steer_encoder_cls(
    model: nn.Module,
    pixels: torch.Tensor,
    delta_x: float | torch.Tensor,
    delta_y: float | torch.Tensor,
    *,
    probe: LinearProbeSteering | None = None,
    probe_path: str | Path = DEFAULT_PROBE_PATH,
    seed: int = DEFAULT_SEED,
    layer: int = DEFAULT_LAYER,
    interpolate_pos_encoding: bool = True,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode pixels after steering the selected encoder-layer CLS activation.

    Args:
        model: JEPA model with ``encoder`` and ``projector`` attributes.
        pixels: Preprocessed pixels in encoder format, usually ``(B, C, H, W)``.
        delta_x: Desired raw block-x change in pixels; standardized by target_std.
        delta_y: Desired raw block-y change in pixels; standardized by target_std.
        probe: Optional preloaded probe from ``load_block_position_probe``.
        probe_path: Probe archive used when ``probe`` is not supplied.
        seed: Probe seed. Defaults to 4.
        layer: Zero-based encoder block whose output CLS token is steered.
        interpolate_pos_encoding: Passed through to the HuggingFace ViT encoder.
        return_details: Also return original/steered CLS and probe deltas.

    Returns:
        The final projected representation, matching ``model.encode``'s per-frame
        embedding before it is reshaped back into a time sequence.
    """
    if not hasattr(model, "encoder") or not hasattr(model, "projector"):
        raise AttributeError("model must expose encoder and projector attributes.")

    blocks = _encoder_blocks(model.encoder)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"Layer {layer} is out of range for {len(blocks)} encoder blocks.")

    if probe is None:
        probe_dtype = pixels.dtype if pixels.is_floating_point() else torch.float32
        probe = load_block_position_probe(
            probe_path,
            seed=seed,
            layer=layer,
            device=pixels.device,
            dtype=probe_dtype,
        )

    details: dict[str, torch.Tensor] = {}
    delta_xy = torch.stack(
        [
            torch.as_tensor(delta_x, dtype=torch.float32, device=pixels.device),
            torch.as_tensor(delta_y, dtype=torch.float32, device=pixels.device),
        ],
        dim=-1,
    )

    def steer_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        local_probe = _move_probe_to_activation(probe, hidden)
        cls_before = hidden[:, 0]
        cls_after = local_probe.steer_cls(cls_before, delta_xy)
        steered_hidden = hidden.clone()
        steered_hidden[:, 0] = cls_after
        if return_details:
            details["cls_before"] = cls_before.detach()
            details["cls_after"] = cls_after.detach()
            details["probe_pred_before"] = local_probe.predict(cls_before).detach()
            details["probe_pred_after"] = local_probe.predict(cls_after).detach()
            details["standardized_delta_xy"] = local_probe.standardized_target_delta(delta_xy).detach()
        if isinstance(output, tuple):
            return (steered_hidden, *output[1:])
        return steered_hidden

    handle = blocks[layer].register_forward_hook(steer_hook)
    try:
        output = model.encoder(
            pixels,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
    finally:
        handle.remove()

    projected = model.projector(output.last_hidden_state[:, 0])
    if not return_details:
        return projected
    details["projected_emb"] = projected.detach()
    details["requested_delta_xy"] = delta_xy.detach()
    details["probe_delta"] = details["probe_pred_after"] - details["probe_pred_before"]
    return projected, details


def steer_info_pixels(
    model: nn.Module,
    info: dict,
    delta_x: float | torch.Tensor,
    delta_y: float | torch.Tensor,
    **kwargs,
) -> dict:
    """JEPA-style convenience wrapper that writes steered embeddings into info."""
    pixels = info["pixels"].float()
    batch = pixels.size(0)
    flat_pixels = pixels.reshape(batch * pixels.size(1), *pixels.shape[2:])
    result = steer_encoder_cls(model, flat_pixels, delta_x, delta_y, **kwargs)
    if isinstance(result, tuple):
        emb, details = result
        info["steering_details"] = details
    else:
        emb = result
    info["emb"] = emb.reshape(batch, pixels.size(1), -1)
    if "action" in info:
        info["act_emb"] = model.action_encoder(info["action"])
    return info
