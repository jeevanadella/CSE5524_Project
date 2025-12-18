#!/bin/bash
#SBATCH --account=PAS3162
#SBATCH --job-name=bioclip2-ft-did_combined
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

python trained/BioClip2-ft-did_combined/train.py \
    --batch_size 128

python trained/BioClip2-ft-did_combined/evaluation.py