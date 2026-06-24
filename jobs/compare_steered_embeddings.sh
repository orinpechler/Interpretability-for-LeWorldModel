#!/bin/bash
#SBATCH --job-name=compare-steered-emb
#SBATCH --output=/home/scur0129/Interpretability-for-LeWorldModel/logs/compare-steered-emb-%j.log
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/logs"

export STABLEWM_HOME="/scratch-shared/orinxAI/stable-wm-data"
export PYTHONPATH="$REPO:$PYTHONPATH"
export MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-lewm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
CONFIG="$STABLEWM_HOME/hf_pusht/config.json"
WEIGHTS="$STABLEWM_HOME/hf_pusht/weights.pt"
PROBE="$REPO/probes/block_position/linear_probe_weights.npz"
COMPARE_OUTPUT_DIR="$REPO/probes/block_position/steered_vs_rendered"
DELTA="${1:-0}"
MODE="${2:-one}"
DELTA_PATH="${DELTA//-/neg}"
DELTA_PATH="${DELTA_PATH//./p}"
if [ "$MODE" = "all_eval" ]; then
    MAX_FRAMES=0
    EVAL_FLAGS=(--eval-valid-only --goal-offset-steps 25)
    OUTPUT="$COMPARE_OUTPUT_DIR/metrics_delta_${DELTA_PATH}_all_eval.json"
    IMAGE_OUTPUT_DIR="$COMPARE_OUTPUT_DIR/synthetic_frames_delta_${DELTA_PATH}_all_eval.py"
else
    MAX_FRAMES=1
    EVAL_FLAGS=()
    OUTPUT="$COMPARE_OUTPUT_DIR/metrics_delta_${DELTA_PATH}.json"
    IMAGE_OUTPUT_DIR="$COMPARE_OUTPUT_DIR/synthetic_frames_delta_${DELTA_PATH}.py"
fi

if [ ! -f "$DATASET" ]; then
    echo "Missing PushT dataset: $DATASET"
    echo "Run: sbatch $REPO/jobs/download_pusht_data.sh"
    exit 1
fi

if [ ! -f "$CONFIG" ] || [ ! -f "$WEIGHTS" ]; then
    echo "Missing LeWM config/weights:"
    echo "  $CONFIG"
    echo "  $WEIGHTS"
    echo "Run: sbatch $REPO/jobs/download_pusht_model.sh"
    exit 1
fi

if [ ! -f "$PROBE" ]; then
    echo "Missing block-position probe: $PROBE"
    echo "Run: sbatch $REPO/jobs/probing.sh block_position"
    exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$IMAGE_OUTPUT_DIR" "$(dirname "$OUTPUT")"

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Dataset: $DATASET"
echo "Weights: $WEIGHTS"
echo "Probe: $PROBE"
echo "Probe seed: 4"
echo "Probe layer: 9"
echo "Delta: $DELTA"
echo "Mode: $MODE"
echo "Max frames: $MAX_FRAMES"
echo "Image output: $IMAGE_OUTPUT_DIR"
echo "Metrics output: $OUTPUT"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

srun python interp_utils/compare_steered_embeddings.py \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --probe-path "$PROBE" \
    --probe-seed 4 \
    --probe-layer 9 \
    --delta "$DELTA" \
    --max-frames "$MAX_FRAMES" \
    --output "$OUTPUT" \
    --output-images \
    --image-output-dir "$IMAGE_OUTPUT_DIR" \
    --device cuda \
    --epsilon 1e-2 \
    "${EVAL_FLAGS[@]}"
