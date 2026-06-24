#!/bin/bash
#SBATCH --job-name=steer-open-loop
#SBATCH --output=logs/steer-open-loop-%j.log
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
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

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
EMBEDDINGS="$EMBEDDINGS_DIR/pusht_encoder_cls_fp32.h5"

# Override any of these via `sbatch --export=ALL,VAR=value jobs/steer_open_loop.sh`
TARGET="${TARGET:-block_angle}"
PROBE_DIR="${PROBE_DIR:-$STABLEWM_HOME/probes/$TARGET}"
DELTA="${DELTA:-0.3}"
TOLERANCE="${TOLERANCE:-0.02}"
NUM_PAIRS="${NUM_PAIRS:-5}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/steering/open_loop}"

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

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Probe dir: $PROBE_DIR"
echo "Target: $TARGET  Delta: $DELTA  Tolerance: $TOLERANCE"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

srun python -m interp_utils.steering.open_loop \
    --dataset "$DATASET" \
    --embeddings "$EMBEDDINGS" \
    --probe-dir "$PROBE_DIR" \
    --target "$TARGET" \
    --delta "$DELTA" \
    --tolerance "$TOLERANCE" \
    --num-pairs "$NUM_PAIRS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda
