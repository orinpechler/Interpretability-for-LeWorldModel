#!/bin/bash
#SBATCH --job-name=download-pusht
#SBATCH --output=logs/download-pusht-%j.log
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"
mkdir -p "$STABLEWM_HOME/datasets"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DOWNLOAD_DIR="$STABLEWM_HOME/hf_downloads/lewm-pusht"
mkdir -p "$DOWNLOAD_DIR"

hf download quentinll/lewm-pusht \
  --repo-type dataset \
  --local-dir "$DOWNLOAD_DIR" || true

if [ ! -f "$DOWNLOAD_DIR/pusht_expert_train.h5.zst" ]; then
    echo "Missing downloaded file: $DOWNLOAD_DIR/pusht_expert_train.h5.zst"
    exit 1
fi

if [ ! -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5" ]; then
python - <<EOF
import zstandard as zstd

src = "$DOWNLOAD_DIR/pusht_expert_train.h5.zst"
dst = "$STABLEWM_HOME/datasets/pusht_expert_train.h5"

with open(src, "rb") as fin, open(dst, "wb") as fout:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(fin, fout)

print(f"Decompressed to {dst}")
EOF
else
    echo "Already decompressed: $STABLEWM_HOME/datasets/pusht_expert_train.h5"
fi

ls -lh "$STABLEWM_HOME/datasets/pusht_expert_train.h5"
echo "STABLEWM_HOME=$STABLEWM_HOME"
echo "CONDA_ENV=leworldmodel"
