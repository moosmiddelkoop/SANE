#!/bin/bash
#SBATCH --job-name=recall_mlp_cifar10
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/recall_mlp_cifar10_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-cifar10"
mkdir -p logs

# resume-safe extraction of the raw zoo (no-op once fully extracted)
unzip -q -n "/projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10.zip" \
    -d /gpfs/scratch1/shared/mmiddelkoop/unthi_zoo -x "__MACOSX/*"

source "$HOME/SANE/.venv/bin/activate"

python property_prediction_cifar10_smallcnnzoo_mlp.py
