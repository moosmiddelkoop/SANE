#!/bin/bash
#SBATCH --job-name=preprocess_dataset_resnet18_cifar100_tk288_allepochs
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --exclusive  # billed as a full node anyway (memory), so take the whole node
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/preprocess_dataset_resnet18_cifar100_tk288_allepochs_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# dataset.pt is ~150 GB (fp16, 39 windows per model). Peak RAM is at the end of train: the stacked
# chunks (~105 GB) and their concatenation (~105 GB).

set -euo pipefail

cd "$HOME/SANE/data"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

# Ray puts its session dir (with Unix sockets) in $TMPDIR, which is on GPFS on Snellius;
# sockets there fail at random ("plasma_store socket not found", "Unable to register worker
# with raylet"). Use node-local memory instead.
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python preprocess_dataset_resnet18_cifar100_tk288_allepochs.py
