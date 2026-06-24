#!/bin/bash
#SBATCH --job-name=probing
#SBATCH --output=logs/probing-%j.out
#SBATCH --error=logs/probing-%j.err
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"
export EMBEDDINGS_DIR="/scratch-shared/orinxAI/embeddings"
export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-lewm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
LEGACY_DATASET="$STABLEWM_HOME/pusht_expert_train.h5"
EMBEDDINGS="$EMBEDDINGS_DIR/pusht_encoder_cls_fp32.h5"

# Probing target. Uncomment one of these defaults, or pass a target as the
# first sbatch argument, e.g. sbatch jobs/probing.sh block_position.
# TARGET="${1:-agent_position}"
# TARGET="${1:-block_position}"
TARGET="${1:-block_angle}"
OUTPUT_DIR="$REPO/probes/$TARGET"

if [ ! -f "$DATASET" ] && [ -f "$LEGACY_DATASET" ]; then
    mkdir -p "$STABLEWM_HOME/datasets"
    ln -s "$LEGACY_DATASET" "$DATASET"
fi

if [ ! -f "$DATASET" ]; then
    echo "Missing PushT dataset: $DATASET"
    echo "Run: sbatch $REPO/jobs/download_pusht_data.sh"
    exit 1
fi

if [ ! -f "$EMBEDDINGS" ]; then
    echo "Missing PushT embeddings: $EMBEDDINGS"
    echo "Run: sbatch $REPO/jobs/extract_pusht_embeddings.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Dataset: $DATASET"
echo "Embeddings: $EMBEDDINGS"
echo "Target: $TARGET"
echo "Output: $OUTPUT_DIR"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

srun python interp_utils/probing.py \
    --dataset "$DATASET" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$OUTPUT_DIR" \
    --target "$TARGET" \
    --train-frac 0.70 \
    --seed 0 \
    --num-seeds 5 \
    --device cuda
