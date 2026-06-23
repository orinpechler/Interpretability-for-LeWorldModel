#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ins_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=slurm_output_ins_env_%A.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/Interpretability-for-LeWorldModel"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f conda_environment.yaml