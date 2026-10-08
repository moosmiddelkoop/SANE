#!/bin/bash
#SBATCH --job-name=probe_windowsize_memory
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/resnet18-cifar100/logs/probe_windowsize_memory_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -uo pipefail  # no -e: an OOM pair must not stop the other pairs

cd "$HOME/SANE/experiments/resnet18-cifar100"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

nvidia-smi --query-gpu=name,memory.total --format=csv
for pair in "256 32" "512 32" "1024 32" "1536 32" "2048 32" "2048 16" "2048 8"; do
    set -- $pair
    python probe_windowsize_memory.py --windowsize=$1 --batchsize=$2
done
