#!/bin/bash
#SBATCH --job-name=steer-closed-loop
#SBATCH --output=logs/steer-closed-loop-%j.log
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
export PYTHONPATH="$REPO:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-lewm}"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

# Override any of these via `sbatch --export=ALL,VAR=value jobs/steer_closed_loop.sh`
TARGET="${TARGET:-block_angle}"
PROBE_DIR="${PROBE_DIR:-$STABLEWM_HOME/probes/$TARGET}"
DELTA="${DELTA:-0.3}"
TOLERANCE="${TOLERANCE:-0.02}"
NUM_PAIRS="${NUM_PAIRS:-5}"
MODE="${MODE:-both}"

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
CHECKPOINT="$STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt"

if [ ! -f "$DATASET" ]; then
    echo "Missing PushT dataset: $DATASET"
    echo "Run: sbatch $REPO/jobs/download_pusht_data.sh"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing LeWM checkpoint: $CHECKPOINT"
    echo "Run: sbatch $REPO/jobs/download_pusht_model.sh"
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
echo "Target: $TARGET  Delta: $DELTA  Tolerance: $TOLERANCE  Mode: $MODE"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

srun python -m interp_utils.steering.closed_loop \
    steering.probe_dir="$PROBE_DIR" \
    steering.target="$TARGET" \
    steering.delta="$DELTA" \
    steering.tolerance="$TOLERANCE" \
    steering.num_pairs="$NUM_PAIRS" \
    steering.mode="$MODE"
