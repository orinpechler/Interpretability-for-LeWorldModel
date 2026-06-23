#!/bin/bash
#SBATCH --job-name=download-pusht-ckpt
#SBATCH --output=logs/download-pusht-ckpt-%j.out
#SBATCH --error=logs/download-pusht-ckpt-%j.err
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/jobs/logs"

export STABLEWM_HOME="$REPO/stable-wm-data"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

python -c "import numpy, pyarrow, datasets, huggingface_hub; print(numpy.__version__, pyarrow.__version__, datasets.__version__, huggingface_hub.__version__); from datasets import config; print('deps ok')"

DOWNLOAD_DIR="$STABLEWM_HOME/hf_pusht"
OUT_DIR="$STABLEWM_HOME/pusht"
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
import stable_worldmodel as swm

from jepa import JEPA
from module import ARPredictor, Embedder, MLP

def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())

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

sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)

out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)

print(f"Saved checkpoint to {out}")
PY

ls -lh "$OUT_CKPT"
echo "STABLEWM_HOME=$STABLEWM_HOME"