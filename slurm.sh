#!/bin/bash

#SBATCH --job-name=videotta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465001897
#SBATCH --output=/users/doloriel/work/Repo/ViTTA/logs/videotta_swin_ucf101.out

# Activate your virtual environment if needed
conda init
conda activate videotta

# Set the working directory
cd /users/doloriel/work/Repo/ViTTA

# Run the Python script
python -m videotta 