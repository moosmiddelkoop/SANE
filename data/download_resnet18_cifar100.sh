#!/bin/bash
#SBATCH --job-name=download_resnet18_cifar100
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_resnet18_cifar100_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# make sure log diretory exists
cd "$HOME/SANE/data"
mkdir -p logs

# Define the URL and target directory
URL="https://zenodo.org/records/6977382/files/cifar100_resnet18_epoch60.zip"

TARGET_DIR="/projects/prjs/shared/wsl"

# Create the target directory if it doesn't exist
mkdir -p ${TARGET_DIR}

# Define the output file path
OUTPUT_FILE="${TARGET_DIR}/cifar100_resnet18_epoch60.zip"

# Download the zip file
curl -L ${URL} -o ${OUTPUT_FILE}

# Unzip the downloaded file
unzip ${OUTPUT_FILE} -d ${TARGET_DIR}

# Optionally, remove the zip file after extraction
rm ${OUTPUT_FILE}

echo "Download and extraction complete."
