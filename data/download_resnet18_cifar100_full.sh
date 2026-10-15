#!/bin/bash
#SBATCH --job-name=download_resnet18_cifar100_full
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/download_resnet18_cifar100_full_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# Full ResNet-18 CIFAR-100 zoo (all epochs, ~2.8 TB tar). Kept as a tar: extracting needs
# about as much space again and may exceed the project inode quota.
URL="https://ai.unisg.ch/files/tar_files/tune_zoo_cifar100_resnet18_kaiming_uniform.tar"
OUTPUT_FILE="/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform.tar"

# -C - resumes a partial file, so a resubmit continues where the last job stopped
curl -L -C - --retry 20 --retry-delay 60 --retry-all-errors -o "$OUTPUT_FILE" "$URL"

ls -l "$OUTPUT_FILE"
echo "expected size: 2787963269120 bytes"
