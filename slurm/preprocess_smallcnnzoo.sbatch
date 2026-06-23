#!/bin/bash
#SBATCH --job-name=preprocess_smallcnnzoo
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/preprocess_smallcnnzoo_%j.out
#SBATCH --mail-user=moos@cwi.nl
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd /gpfs/home1/mmiddelkoop/SANE
mkdir -p slurm/logs

source .venv/bin/activate

echo "=== Starting preprocessing ==="

python3 data/preprocess_dataset_cnn_cifar10.py

echo "=== Preprocessing complete ==="