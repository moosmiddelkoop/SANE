#!/bin/bash
#SBATCH --job-name=recall_prediction_last_layer
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/recall_prediction_last_layer_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/resnet18-cifar100"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python recall_prediction_last_layer.py
