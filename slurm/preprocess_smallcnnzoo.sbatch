#!/bin/bash
#SBATCH --job-name=preprocess_smallcnnzoo
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/preprocess_smallcnnzoo_%j.out
#SBATCH --mail-user=moos@cwi.nl
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd /gpfs/home1/mmiddelkoop/SANE
mkdir -p slurm/logs

source .venv/bin/activate

echo "=== Starting preprocessing ==="

python3 data/preprocess_dataset_smallcnnzoo_mnist.py

echo "=== Preprocessing complete ==="