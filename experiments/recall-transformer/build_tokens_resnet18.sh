#!/bin/bash
#SBATCH --job-name=recall_transformer_tokens_resnet18
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/build_tokens_resnet18_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch build_tokens_resnet18.sh
# Each CPU is one Ray worker, and each worker needs ~11 GB: 32 CPUs with the default 57 GB ran
# out of memory on one train chunk (25 models x 13 raw checkpoints of 45 MB). 16 CPUs with 180 GB
# is what resnet18-cifar100/recall_prediction_allepochs.sh used for the same loading (~1.5 h).

set -euo pipefail

cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

# Ray (used by DatasetTokens) puts its Unix sockets in $TMPDIR, which is on GPFS on
# Snellius; sockets there fail at random. Use node-local memory instead.
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python build_tokens_resnet18.py
