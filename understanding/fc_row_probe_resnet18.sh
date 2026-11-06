#!/bin/bash
#SBATCH --job-name=fc_row_probe_resnet18
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/understanding/logs/fc_row_probe_resnet18_%j.out

# usage: sbatch fc_row_probe_resnet18.sh
# Reads fc from 12,000 checkpoints (~0.35 s each on GPFS, 16 threads); later runs use the saved features.

set -euo pipefail

cd "$HOME/SANE/understanding"
source "$HOME/SANE/.venv/bin/activate"

python fc_row_probe_resnet18.py
