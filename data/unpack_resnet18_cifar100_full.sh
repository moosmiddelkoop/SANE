#!/bin/bash
#SBATCH --job-name=unpack_resnet18_cifar100_full
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/unpack_resnet18_cifar100_full_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# Unpacks every 5th epoch (checkpoint_000000, _000005, ..., _000060) and the metadata of
# each model. The other epochs stay in the tar: all 61 epochs need ~2.5 TiB, these 13 ~0.6 TiB.
set -euo pipefail

TARGET_DIR="/projects/prjs2156/shared/wsl/cifar100_resnet18"
TAR_FILE="${TARGET_DIR}/tune_zoo_cifar100_resnet18_kaiming_uniform.tar"

if [ "$(stat -c %s "$TAR_FILE")" != "2787963269120" ]; then
    echo "incomplete download: $(stat -c %s "$TAR_FILE") bytes"
    exit 1
fi

# excludes checkpoint dirs whose epoch does not end in 0 or 5
tar -xf "$TAR_FILE" -C "$TARGET_DIR" --exclude='checkpoint_0000?[1-46-9]'

ls "${TARGET_DIR}/tune_zoo_cifar100_resnet18_kaiming_uniform" | wc -l
echo "unpack complete"
