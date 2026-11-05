#!/bin/bash
#SBATCH --job-name=recall_transformer_resnet18
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=300G  # the 272 GB token cache is read into RAM
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/train_recall_transformer_resnet18_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch train_recall_transformer_resnet18.sh [seed: 0]
# needs tokens_ep5-60.bin from: sbatch build_tokens_resnet18.sh

set -euo pipefail

cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python train_recall_transformer_resnet18.py --seed="${1:-0}"
