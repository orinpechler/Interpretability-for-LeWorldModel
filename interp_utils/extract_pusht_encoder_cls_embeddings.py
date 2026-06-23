#!/usr/bin/env python
"""Extract PushT encoder CLS embeddings from every ViT layer.

The output rows preserve the source HDF5 frame order:
    output["encoder_cls_layers"][i] corresponds to source["pixels"][i].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters for PushT data
import numpy as np
import stable_pretraining as spt
import torch
from torch.utils.data import DataLoader, Dataset

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


DEFAULT_DATASET = Path("stable-wm-data/datasets/pusht_expert_train.h5")
DEFAULT_CONFIG = Path("stable-wm-data/hf_pusht/config.json")
DEFAULT_WEIGHTS = Path("stable-wm-data/hf_pusht/weights.pt")
DEFAULT_OUTPUT = Path("stable-wm-data/cache/pusht_encoder_cls_fp32.h5")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-layer ViT CLS embeddings and projected embeddings for PushT."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="Compression for embedding datasets. Default is fastest/no compression.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume writing into an existing output file using the completed mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def clean_config(section: dict) -> dict:
    return {k: v for k, v in section.items() if not k.startswith("_")}


def load_model(config_path: Path, weights_path: Path, device: torch.device) -> JEPA:
    cfg = json.loads(config_path.read_text())

    encoder_cfg = clean_config(cfg["encoder"])
    encoder = spt.backbone.utils.vit_hf(
        encoder_cfg["size"],
        patch_size=encoder_cfg["patch_size"],
        image_size=encoder_cfg["image_size"],
        pretrained=False,
        use_mask_token=encoder_cfg.get("use_mask_token", False),
    )

    def make_mlp(name: str) -> MLP:
        mlp_cfg = clean_config(cfg[name])
        return MLP(
            input_dim=mlp_cfg["input_dim"],
            output_dim=mlp_cfg["output_dim"],
            hidden_dim=mlp_cfg["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**clean_config(cfg["predictor"])),
        action_encoder=Embedder(**clean_config(cfg["action_encoder"])),
        projector=make_mlp("projector"),
        pred_proj=make_mlp("pred_proj"),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


class H5PixelsDataset(Dataset):
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as handle:
            self.length = int(handle["pixels"].shape[0])
        self._handle = None
        self._pixels = None

    def __len__(self) -> int:
        return self.length

    def _open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
            self._pixels = self._handle["pixels"]

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        self._open()
        pixels = np.asarray(self._pixels[index])
        return index, torch.from_numpy(pixels)


def h5_compression_args(name: str) -> dict:
    if name == "none":
        return {}
    return {"compression": name}


def preprocess_pixels(pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    pixels = pixels.to(device, non_blocking=True).float()
    pixels = pixels.permute(0, 3, 1, 2).div_(255.0)
    mean = IMAGENET_MEAN.to(device=device, dtype=pixels.dtype)
    std = IMAGENET_STD.to(device=device, dtype=pixels.dtype)
    return (pixels - mean) / std


def copy_index_dataset(src: h5py.File, dst: h5py.File, name: str) -> None:
    if name in dst:
        return
    dst.create_dataset(name, data=src[name][...], chunks=src[name].chunks)


def prepare_output(
    output_path: Path,
    source_path: Path,
    config_path: Path,
    weights_path: Path,
    num_frames: int,
    num_layers: int,
    embed_dim: int,
    chunk_size: int,
    compression: str,
    overwrite: bool,
    resume: bool,
) -> h5py.File:
    if output_path.exists() and overwrite:
        output_path.unlink()
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"{output_path} already exists. Use --resume to continue or --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(output_path, "a")
    comp = h5_compression_args(compression)
    layer_shape = (num_frames, num_layers, embed_dim)
    proj_shape = (num_frames, embed_dim)

    if "encoder_cls_layers" not in handle:
        handle.create_dataset(
            "encoder_cls_layers",
            shape=layer_shape,
            dtype="float32",
            chunks=(min(chunk_size, num_frames), num_layers, embed_dim),
            **comp,
        )
    if "projected_emb" not in handle:
        handle.create_dataset(
            "projected_emb",
            shape=proj_shape,
            dtype="float32",
            chunks=(min(chunk_size, num_frames), embed_dim),
            **comp,
        )
    if "completed" not in handle:
        handle.create_dataset(
            "completed",
            shape=(num_frames,),
            dtype="bool",
            chunks=(min(chunk_size, num_frames),),
        )
    if "source_frame_index" not in handle:
        handle.create_dataset(
            "source_frame_index",
            data=np.arange(num_frames, dtype=np.int64),
            chunks=(min(chunk_size, num_frames),),
        )

    with h5py.File(source_path, "r") as source:
        for name in ("episode_idx", "step_idx", "ep_offset", "ep_len"):
            copy_index_dataset(source, handle, name)

    handle.attrs["dataset_name"] = "pusht_expert_train"
    handle.attrs["source_h5"] = str(source_path)
    handle.attrs["config"] = str(config_path)
    handle.attrs["weights"] = str(weights_path)
    handle.attrs["num_frames"] = num_frames
    handle.attrs["num_layers"] = num_layers
    handle.attrs["embed_dim"] = embed_dim
    handle.attrs["dtype"] = "float32"
    handle.attrs["row_mapping"] = "row i corresponds to source_h5['pixels'][i]"
    handle.attrs["encoder_cls_layers"] = "CLS token from each transformer layer output"
    handle.attrs["projected_emb"] = "projector(final layer CLS token)"
    handle.flush()
    return handle


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    model = load_model(args.config, args.weights, device)

    dataset = H5PixelsDataset(args.dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    with h5py.File(args.dataset, "r") as source:
        num_frames = int(source["pixels"].shape[0])
    if num_frames != len(dataset):
        raise RuntimeError(f"Dataset length mismatch: {num_frames} != {len(dataset)}")

    with torch.inference_mode():
        sample_pixels = torch.zeros(1, 3, 224, 224, device=device)
        sample_output = model.encoder(
            sample_pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
        )
        num_layers = len(sample_output.hidden_states) - 1
        embed_dim = int(sample_output.last_hidden_state.shape[-1])

    output = prepare_output(
        args.output,
        args.dataset,
        args.config,
        args.weights,
        num_frames,
        num_layers,
        embed_dim,
        args.chunk_size,
        args.compression,
        args.overwrite,
        args.resume,
    )

    completed = output["completed"]
    encoder_cls_layers = output["encoder_cls_layers"]
    projected_emb = output["projected_emb"]
    processed = int(completed[:].sum()) if args.resume else 0

    try:
        for batch_num, (indices, pixels) in enumerate(loader, start=1):
            indices_np = indices.numpy()
            if args.resume and completed[indices_np].all():
                continue

            pixels_device = preprocess_pixels(pixels, device)

            with torch.inference_mode():
                enc_output = model.encoder(
                    pixels_device,
                    interpolate_pos_encoding=True,
                    output_hidden_states=True,
                )
                cls_layers = torch.stack(
                    [state[:, 0] for state in enc_output.hidden_states[1:]],
                    dim=1,
                )
                projected = model.projector(enc_output.last_hidden_state[:, 0])

            encoder_cls_layers[indices_np] = cls_layers.float().cpu().numpy()
            projected_emb[indices_np] = projected.float().cpu().numpy()
            completed[indices_np] = True
            processed += len(indices_np)

            if batch_num == 1 or batch_num % 25 == 0:
                output.flush()
                print(f"processed {processed}/{num_frames} frames", flush=True)
        output.flush()
    finally:
        output.close()


if __name__ == "__main__":
    main()
