#!/bin/bash
#SBATCH --job-name=recall_mlp_svhn
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/recall_mlp_svhn_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-svhn"
mkdir -p logs

# resume-safe extraction of the raw zoo (no-op once fully extracted)
# extraction verified complete 2026-07-30; re-enable for a fresh scratch copy
# unzip -q -n "/projects/prjs2156/shared/wsl/unthi_zoo/unthi_svhn.zip" \
#     -d /gpfs/scratch1/shared/mmiddelkoop/unthi_zoo -x "__MACOSX/*"

source "$HOME/SANE/.venv/bin/activate"

python property_prediction_svhn_smallcnnzoo_mlp.py
