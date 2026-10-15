#!/bin/bash
#SBATCH --job-name=plot_accuracy_cifar100_resnet18
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/plot_accuracy_cifar100_resnet18_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/resnet18-cifar100"
source "$HOME/SANE/.venv/bin/activate"

python plot_accuracy_over_epochs.py
