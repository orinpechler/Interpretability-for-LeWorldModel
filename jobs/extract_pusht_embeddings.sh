#!/bin/bash
#SBATCH --job-name=extract-pusht-emb
#SBATCH --output=logs/extract-pusht-emb-%j.out
#SBATCH --error=logs/extract-pusht-emb-%j.err
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gpus=1
##SBATCH --account=<your-snellius-project-account>

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/jobs/logs"

export PATH="$HOME/.local/bin:$PATH"
export STABLEWM_HOME="$REPO/stable-wm-data"
export PYTHONPATH="$REPO:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-lewm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

source "$REPO/.venv/bin/activate"

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
LEGACY_DATASET="$STABLEWM_HOME/pusht_expert_train.h5"
CONFIG="$STABLEWM_HOME/hf_pusht/config.json"
WEIGHTS="$STABLEWM_HOME/hf_pusht/weights.pt"
OUTPUT_DIR="$STABLEWM_HOME/embeddings"
OUTPUT="$OUTPUT_DIR/pusht_encoder_cls_fp32.h5"

if [ ! -f "$DATASET" ] && [ -f "$LEGACY_DATASET" ]; then
    mkdir -p "$STABLEWM_HOME/datasets"
    ln -s "$LEGACY_DATASET" "$DATASET"
fi

if [ ! -f "$DATASET" ]; then
    echo "Missing PushT dataset: $DATASET"
    echo "Run: sbatch $REPO/jobs/install_pusht_data.job"
    exit 1
fi

if [ ! -f "$CONFIG" ] || [ ! -f "$WEIGHTS" ]; then
    echo "Missing LeWM config/weights:"
    echo "  $CONFIG"
    echo "  $WEIGHTS"
    echo "Run: sbatch $REPO/jobs/download_pusht_model.job"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

srun python interp_utils/extract_pusht_encoder_cls_embeddings.py \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --output "$OUTPUT" \
    --batch-size 256 \
    --num-workers "${SLURM_CPUS_PER_TASK:-6}" \
    --device cuda \
    --resume
