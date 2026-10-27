#!/bin/bash
#SBATCH --job-name=property_prediction_cifar100_resnet18_mlp
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/property_prediction_cifar100_resnet18_mlp_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch property_prediction_cifar100_resnet18_mlp.sh <tokensize: 288 or 576> [encoder: sane (default) or lq]

set -euo pipefail

TOKENSIZE=$1
ENCODER=${2:-sane}

cd "$HOME/SANE/experiments/resnet18-cifar100"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

# Ray (used by DatasetTokens) puts its Unix sockets in $TMPDIR, which is on GPFS on
# Snellius; sockets there fail at random. Use node-local memory instead.
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "tokensize: $TOKENSIZE, encoder: $ENCODER"
python property_prediction_cifar100_resnet18_mlp.py --tokensize="$TOKENSIZE" --encoder="$ENCODER"
