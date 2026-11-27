#!/bin/bash
#SBATCH --job-name=eval_recall_cifar100_resnet18
#SBATCH --partition=<gpu_partition>
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/logs/eval_recall_cifar100_resnet18_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

cd "/path/to/tmlr2026_model_editing/resnet18_cifar100_recall_labels"
mkdir -p logs
source "/path/to/venv/bin/activate"

python evaluate_per_class_recall_cifar100_resnet18.py
