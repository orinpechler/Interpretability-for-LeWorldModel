#!/bin/bash
#SBATCH --job-name=eval-pusht-lewm
#SBATCH --output=logs/eval-pusht-lewm-%j.out
#SBATCH --error=logs/eval-pusht-lewm-%j.err
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --export=NONE

set -e

REPO="$HOME/Interpretability-for-LeWorldModel"
mkdir -p "$REPO/jobs/logs"

export STABLEWM_HOME="$REPO/stable-wm-data"
export PYTHONPATH="$REPO:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-lewm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate leworldmodel

DATASET="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
LEGACY_DATASET="$STABLEWM_HOME/pusht_expert_train.h5"

if [ ! -f "$DATASET" ] && [ -f "$LEGACY_DATASET" ]; then
    mkdir -p "$STABLEWM_HOME/datasets"
    ln -s "$LEGACY_DATASET" "$DATASET"
fi

if [ ! -f "$DATASET" ]; then
    echo "Missing PushT dataset: $DATASET"
    echo "Run: sbatch $REPO/jobs/install_pusht_data.job"
    exit 1
fi

CHECKPOINT_DIR="$STABLEWM_HOME/checkpoints/pusht/lewm"
SAVED_WEIGHTS="$STABLEWM_HOME/hf_pusht/weights.pt"
SAVED_CONFIG="$STABLEWM_HOME/hf_pusht/config.json"
OBJECT_CHECKPOINT="$STABLEWM_HOME/pusht/lewm_object.ckpt"

if [ ! -f "$STABLEWM_HOME/pusht/lewm_object.ckpt" ]; then
    echo "Missing LeWM checkpoint: $STABLEWM_HOME/pusht/lewm_object.ckpt"
    echo "Run: sbatch $REPO/jobs/download_pusht_model.job"
    exit 1
fi

if [ ! -f "$SAVED_WEIGHTS" ] || [ ! -f "$SAVED_CONFIG" ]; then
    echo "Missing local LeWM weights/config expected by eval.py:"
    echo "  $SAVED_WEIGHTS"
    echo "  $SAVED_CONFIG"
    echo "The object checkpoint exists at: $OBJECT_CHECKPOINT"
    echo "Run: sbatch $REPO/jobs/download_pusht_model.job"
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR"
ln -sf "$SAVED_WEIGHTS" "$CHECKPOINT_DIR/weights.pt"
ln -sf "$SAVED_CONFIG" "$CHECKPOINT_DIR/config.json"

mkdir -p "$MPLCONFIGDIR"

python - <<'PY'
try:
    from stable_worldmodel.envs.pusht.env import PushT  # noqa: F401
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    raise SystemExit(
        f"Missing Python package for PushT eval: {missing}\n"
        "Regenerate the conda env with: bash generate_conda_env.sh && sbatch jobs/install_env.sh"
    )
PY

cd "$REPO"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "STABLEWM_HOME=${STABLEWM_HOME}"

srun python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
srun python eval.py --config-name=pusht.yaml policy=pusht/lewm
