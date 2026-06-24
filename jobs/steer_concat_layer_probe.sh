#!/bin/bash
#SBATCH --job-name=steer-concat-layer-probe
#SBATCH --output=logs/steer-concat-layer-probe-%j.log
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"
export EMBEDDINGS_DIR="/scratch-shared/orinxAI/embeddings"
export PYTHONPATH="$REPO:$PYTHONPATH"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="${DATASET:-$STABLEWM_HOME/datasets/pusht_expert_train.h5}"
EMBEDDINGS="${EMBEDDINGS:-$EMBEDDINGS_DIR/pusht_encoder_cls_fp32.h5}"
TARGET="${TARGET:-block_angle}"
NUM_BLOCKS="${NUM_BLOCKS:-30}"
PCA_COMPONENTS="${PCA_COMPONENTS:-10 25 50 100 300 600 1200}"

if [ ! -f "$DATASET" ]; then
    echo "Missing PushT dataset: $DATASET"
    exit 1
fi

if [ ! -f "$EMBEDDINGS" ]; then
    echo "Missing PushT embeddings: $EMBEDDINGS"
    exit 1
fi

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Target: $TARGET  Blocks: $NUM_BLOCKS"

srun python -m interp_utils.steering.concat_layer_probe \
    --dataset "$DATASET" \
    --embeddings "$EMBEDDINGS" \
    --target "$TARGET" \
    --num-blocks "$NUM_BLOCKS" \
    --pca-components $PCA_COMPONENTS
