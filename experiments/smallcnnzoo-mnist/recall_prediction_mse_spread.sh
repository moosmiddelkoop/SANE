#!/bin/bash
#SBATCH --job-name=recall_mse_spread
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/recall_mse_spread_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-mnist"
mkdir -p logs

source "$HOME/SANE/.venv/bin/activate"

python recall_prediction_mse_spread.py
python plot_mse_spread.py
