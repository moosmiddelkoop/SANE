#!/bin/bash
#SBATCH --job-name=compare_spike_ckpts_ep8
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/compare_spike_ckpts_ep8_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-mnist"
mkdir -p logs

source "$HOME/SANE/.venv/bin/activate"

python compare_spike_checkpoints_ep8.py
