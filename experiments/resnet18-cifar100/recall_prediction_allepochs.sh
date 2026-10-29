#!/bin/bash
#SBATCH --job-name=recall_prediction_allepochs
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/recall_prediction_allepochs_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch recall_prediction_allepochs.sh [encoder: sane (default) or lq]
# lq, or sane with a cached embeddings file, needs no GPU:
# sbatch --partition=rome --gpus=0 recall_prediction_allepochs.sh lq
# Encoding 1000 models x 13 checkpoints in 3 x 28 chunks is most of the runtime; with a
# cached embeddings file the heads alone take minutes.

set -euo pipefail

cd "$HOME/SANE/experiments/resnet18-cifar100"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

# Ray (used by DatasetTokens) puts its Unix sockets in $TMPDIR, which is on GPFS on
# Snellius; sockets there fail at random. Use node-local memory instead.
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python recall_prediction_allepochs.py --encoder="${1:-sane}"
