#!/bin/bash
#SBATCH --account=PAS3162
#SBATCH --job-name=train_dino2
#SBATCH --time=48:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
source $(conda info --base)/etc/profile.d/conda.sh

conda activate cv_env

python trained/BioClip2-ft-did/train.py