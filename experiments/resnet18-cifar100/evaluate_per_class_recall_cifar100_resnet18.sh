#!/bin/bash
#SBATCH --job-name=eval_recall_cifar100_resnet18
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/eval_recall_cifar100_resnet18_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/resnet18-cifar100"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python evaluate_per_class_recall_cifar100_resnet18.py
