#!/bin/bash
#SBATCH --job-name=recall_mlp_fmnist
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/recall_mlp_fmnist_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-fmnist"
mkdir -p logs

# resume-safe extraction of the raw zoo (no-op once fully extracted)
# extraction verified complete 2026-07-30; re-enable for a fresh scratch copy
# unzip -q -n "/projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist.zip" \
#     -d /gpfs/scratch1/shared/mmiddelkoop/unthi_zoo -x "__MACOSX/*"

source "$HOME/SANE/.venv/bin/activate"

python property_prediction_fmnist_smallcnnzoo_mlp.py
