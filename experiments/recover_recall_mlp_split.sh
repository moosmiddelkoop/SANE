#!/bin/bash
#SBATCH --job-name=recover_recall_split
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs/recover_recall_split_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments"
mkdir -p logs

source "$HOME/SANE/.venv/bin/activate"

python recover_recall_mlp_split.py
