#!/bin/bash
#SBATCH --job-name=download-pusht-ckpt
#SBATCH --output=logs/download-pusht-ckpt-%j.log
#SBATCH --partition=staging
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

python -c "import numpy, pyarrow, datasets, huggingface_hub; print(numpy.__version__, pyarrow.__version__, datasets.__version__, huggingface_hub.__version__); from datasets import config; print('deps ok')"

DOWNLOAD_DIR="$STABLEWM_HOME/hf_pusht"
OUT_DIR="$STABLEWM_HOME/checkpoints/pusht"
OUT_CKPT="$OUT_DIR/lewm_object.ckpt"

mkdir -p "$DOWNLOAD_DIR" "$OUT_DIR"

hf download quentinll/lewm-pusht \
  --local-dir "$DOWNLOAD_DIR" || true

if [ ! -f "$DOWNLOAD_DIR/weights.pt" ]; then
    echo "Missing weights.pt"
    exit 1
fi

if [ ! -f "$DOWNLOAD_DIR/config.json" ]; then
    echo "Missing config.json"
    exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"

python - <<'PY'
import json
import torch
from pathlib import Path

import stable_pretraining as spt

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


STABLEWM_HOME = Path("/scratch-shared/orinxAI/stable-wm-data")

src = STABLEWM_HOME / "hf_pusht"
out = STABLEWM_HOME / "checkpoints" / "pusht" / "lewm_object.ckpt"


def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


cfg = json.loads((src / "config.json").read_text())

print("\n=== CONFIG ===")
print(json.dumps(cfg, indent=2))


encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False,
    use_mask_token=False,
)


def make_mlp(k):
    c = clean(cfg[k])
    return MLP(
        input_dim=c["input_dim"],
        output_dim=c["output_dim"],
        hidden_dim=c["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )


model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**clean(cfg["predictor"])),
    action_encoder=Embedder(**clean(cfg["action_encoder"])),
    projector=make_mlp("projector"),
    pred_proj=make_mlp("pred_proj"),
)


# Load original HF checkpoint
sd = torch.load(
    src / "weights.pt",
    map_location="cpu",
    weights_only=False,
)


print("\n=== ORIGINAL KEYS ===")
print(list(sd.keys())[:10])


# transformers is pinned to <5 (see pyproject.toml) specifically so the
# ViTModel built here matches the checkpoint's original key names
# (encoder.layer.*.attention.attention.query, etc.) with no remapping.
model.load_state_dict(sd, strict=True)

print("\nCheckpoint successfully loaded into target model")


out.parent.mkdir(parents=True, exist_ok=True)


# IMPORTANT:
# load_pretrained() expects a state_dict
torch.save(
    model.state_dict(),
    out,
)


# Verify saved checkpoint
saved = torch.load(out, map_location="cpu")

print("\n=== SAVED KEYS ===")
print(list(saved.keys())[:10])

print("\nSaved checkpoint:")
print(out)

PY

ls -lh "$OUT_CKPT"
echo "STABLEWM_HOME=$STABLEWM_HOME"