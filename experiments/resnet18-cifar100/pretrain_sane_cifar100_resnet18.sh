#!/bin/bash
#SBATCH --job-name=pretrain_sane_cifar100_resnet18
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/pretrain_sane_cifar100_resnet18_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch pretrain_sane_cifar100_resnet18.sh <tokensize: 288 or 576> [epochs, default 50]
# The 1-GPU share of an H100 node has 180 GiB RAM; dataset.pt (~146 GB) is loaded into it once.
# Expected runtime: ~11 h for 288 x 50 epochs and for 576 x 100 epochs (same number of steps).

set -euo pipefail

TOKENSIZE=$1
EPOCHS=${2:-50}

cd "$HOME/SANE/experiments/resnet18-cifar100"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

# Ray puts its session dir (with Unix sockets) in $TMPDIR, which is on GPFS on Snellius;
# sockets there fail at random. Use node-local memory instead.
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "tokensize: $TOKENSIZE, epochs: $EPOCHS"
python pretrain_sane_cifar100_resnet18.py --tokensize="$TOKENSIZE" --epochs="$EPOCHS"
