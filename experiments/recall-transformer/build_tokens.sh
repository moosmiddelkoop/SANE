#!/bin/bash
#SBATCH --job-name=recall_transformer_tokens
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/build_tokens_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch build_tokens.sh <zoo: mnist, fmnist, cifar10 or svhn>
# mnist, cifar10 and svhn: unzips the zoo to scratch (/projects is near its inode quota)
# and creates its split.json there. fmnist: uses the copy on /projects, which needs a
# split.json made by hand (the directory belongs to silic).

set -euo pipefail

ZOO=$1
cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

if [ "$ZOO" != fmnist ]; then
    EXTRACT_DIR=/scratch-shared/mmiddelkoop/unthi_zoo
    mkdir -p "$EXTRACT_DIR"
    # -n: never overwrite, so a rerun skips the files that are already there
    unzip -q -n "/projects/prjs2156/shared/wsl/unthi_zoo/unthi_$ZOO.zip" -d "$EXTRACT_DIR" -x "__MACOSX/*"
    if [ ! -f "$EXTRACT_DIR/unthi_$ZOO/split.json" ]; then
        python "$HOME/SANE/data/create_split.py" "$EXTRACT_DIR/unthi_$ZOO"
    fi
fi

# Ray (used by DatasetTokens) puts its Unix sockets in $TMPDIR, which is on GPFS on
# Snellius; sockets there fail at random. Use node-local memory instead.
export RAY_TMPDIR=/dev/shm/ray_$SLURM_JOB_ID
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python build_tokens.py --zoo="$ZOO"
