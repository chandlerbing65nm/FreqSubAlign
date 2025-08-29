#!/bin/bash

#SBATCH --job-name=chandlertasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465001897
#SBATCH --output=logs/output_%j.txt

# Activate your virtual environment if needed
conda init
conda activate ttadapt

# Set the working directory
cd /users/doloriel/work/Repo/SWaveletA

# Run the Python script
python -m main_tta_ss2