#!/bin/bash
#SBATCH --job-name=preprocess_resnet18_cifar100
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/preprocess_resnet18_cifar100_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# dataset.pt is ~150 GB (supersample=10) and the whole dataset stays in RAM until it is
# saved, so this needs more memory than a rome node (224 GiB) safely gives

set -euo pipefail

cd "$HOME/SANE/data"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python preprocess_dataset_resnet18_cifar100_tk288.py
