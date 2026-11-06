#!/bin/bash
#SBATCH --job-name=recall_transformer_export
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/export_metanets_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch export_metanets.sh <zoo: mnist, fmnist, cifar10 or svhn>
# needs the extracted zoo (build_tokens.sh) and results/<zoo>-{scratch,sane}-seed0.pt

set -euo pipefail

cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

# Ray (used by DatasetTokens) sockets fail at random on GPFS; use node-local memory
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python export_metanets.py --zoo="$1"
