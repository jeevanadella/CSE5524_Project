#!/bin/bash

#SBATCH --account=PAS3162
#SBATCH --job-name=bioclip2-ft-did_lora
#SBATCH --time=48:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%j.out

# 1. Load the system Python module
module reset
module load python/3.12

# 2. Activate the environment securely
source activate cse5524_env

# 3. FIX: Downgrade NumPy to version 1.x to prevent crash
pip install "numpy<2.0"

# 4. FIX: Ensure PyTorch is a valid version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install PEFT for LoRA
pip install peft

# 6. Run Training
echo "Starting training on $(hostname)..."
echo "GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

python trained/BioClip2-ft-did_lora/train.py

python trained/BioClip2-ft-did_lora/evaluation.py