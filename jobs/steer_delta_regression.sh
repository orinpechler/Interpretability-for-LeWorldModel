#!/bin/bash
#SBATCH --job-name=steer-delta-regression
#SBATCH --output=logs/steer-delta-regression-%j.log
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"
export EMBEDDINGS_DIR="/scratch-shared/orinxAI/embeddings"
export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-lewm}"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
EMBEDDINGS="$EMBEDDINGS_DIR/pusht_encoder_cls_fp32.h5"

# Override any of these via `sbatch --export=ALL,VAR=value jobs/steer_delta_regression.sh`
TARGET="${TARGET:-block_angle}"
TARGET_DIM_INDEX="${TARGET_DIM_INDEX:-0}"
PROBE_DIR="${PROBE_DIR:-$STABLEWM_HOME/probes/$TARGET}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5000}"
DELTA_MIN="${DELTA_MIN:-0.05}"
DELTA_MAX="${DELTA_MAX:-0.5}"
OUTPUT="${OUTPUT:-$REPO/outputs/steering/delta_dataset_${TARGET}.npz}"

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

if [ ! -f "$PROBE_DIR/linear_probe_weights.npz" ] || [ ! -f "$PROBE_DIR/split.json" ] || [ ! -f "$PROBE_DIR/metrics.csv" ]; then
    echo "Missing linear probe output in: $PROBE_DIR"
    echo "Run: python interp_utils/probing.py --target $TARGET --output-dir $PROBE_DIR"
    exit 1
fi

mkdir -p "$MPLCONFIGDIR"
cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Probe dir: $PROBE_DIR"
echo "Target: $TARGET (dim index $TARGET_DIM_INDEX)  Examples: $NUM_EXAMPLES  Delta range: [$DELTA_MIN, $DELTA_MAX]"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

srun python -m interp_utils.steering.delta_regression \
    --dataset "$DATASET" \
    --embeddings "$EMBEDDINGS" \
    --probe-dir "$PROBE_DIR" \
    --target "$TARGET" \
    --target-dim-index "$TARGET_DIM_INDEX" \
    --num-examples "$NUM_EXAMPLES" \
    --delta-min "$DELTA_MIN" \
    --delta-max "$DELTA_MAX" \
    --output "$OUTPUT" \
    --device cuda
