#!/bin/bash
#SBATCH --job-name=consolidate_unthi_mnist
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
# note: #SBATCH lines don't expand variables like $HOME, and the output dir
# must exist at submission time
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/consolidate_unthi_mnist_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/data"
source "$HOME/SANE/.venv/bin/activate"

python consolidate_preprocessed.py /projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist_preprocessed
