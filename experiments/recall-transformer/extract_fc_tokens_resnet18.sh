#!/bin/bash
#SBATCH --job-name=extract_fc_tokens_resnet18
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/extract_fc_tokens_resnet18_%j.out

# usage: sbatch extract_fc_tokens_resnet18.sh
# needs tokens_ep5-60.bin from: sbatch build_tokens_resnet18.sh

set -euo pipefail

cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python extract_fc_tokens_resnet18.py
