#!/bin/bash
#SBATCH --job-name=steer-orientation-classification
#SBATCH --output=logs/steer-orientation-classification-%j.log
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
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
LAYER_INDEX="${LAYER_INDEX:-9}"
QUADRANT_A="${QUADRANT_A:-0}"
QUADRANT_B="${QUADRANT_B:-2}"
NUM_BLOCKS="${NUM_BLOCKS:-12}"

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
echo "Layer: $LAYER_INDEX  Quadrants: $QUADRANT_A vs $QUADRANT_B  Blocks: $NUM_BLOCKS"

srun python -m interp_utils.steering.orientation_classification \
    --dataset "$DATASET" \
    --embeddings "$EMBEDDINGS" \
    --layer-index "$LAYER_INDEX" \
    --quadrant-a "$QUADRANT_A" \
    --quadrant-b "$QUADRANT_B" \
    --num-blocks "$NUM_BLOCKS"
