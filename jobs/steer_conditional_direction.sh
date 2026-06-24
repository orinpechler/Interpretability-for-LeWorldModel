#!/bin/bash
#SBATCH --job-name=steer-conditional-direction
#SBATCH --output=logs/steer-conditional-direction-%j.log
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"
export PYTHONPATH="$REPO:$PYTHONPATH"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="${DATASET:-$REPO/outputs/steering/delta_dataset_block_angle.npz}"
STATE_DATASET="${STATE_DATASET:-$STABLEWM_HOME/datasets/pusht_expert_train.h5}"
STATE_COLUMN="${STATE_COLUMN:-4}"
NUM_BINS="${NUM_BINS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/steering/conditional_direction/block_angle}"

if [ ! -f "$DATASET" ]; then
    echo "Missing delta-regression dataset: $DATASET"
    echo "Run: sbatch $REPO/jobs/steer_delta_regression.sh"
    exit 1
fi

if [ ! -f "$STATE_DATASET" ]; then
    echo "Missing PushT dataset: $STATE_DATASET"
    echo "Run: sbatch $REPO/jobs/download_pusht_data.sh"
    exit 1
fi

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Dataset: $DATASET  State column: $STATE_COLUMN  Bins: $NUM_BINS"

srun python -m interp_utils.steering.conditional_direction \
    --dataset "$DATASET" \
    --state-dataset "$STATE_DATASET" \
    --state-column "$STATE_COLUMN" \
    --num-bins "$NUM_BINS" \
    --output-dir "$OUTPUT_DIR"
