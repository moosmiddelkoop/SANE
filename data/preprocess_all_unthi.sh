#!/bin/bash
#SBATCH --job-name=preprocess_all_unthi
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/preprocess_all_unthi_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# Unzips all raw small model zoo data into scratch-shared per image dataset, preprocess them, 
# delete the unzipped data, and store the preprocessed data in the shared project directory.
# With the preprocess_dataset_smallcnnzoo.py script as of 15-07-2026, this takes about 5 hours

set -euo pipefail

cd "$HOME/SANE/data"
source "$HOME/SANE/.venv/bin/activate"

ZOO_DIR=/projects/prjs2156/shared/wsl/unthi_zoo
# extract to scratch: /projects is near its inode quota and the stuck
# extractions there are owned by silic (group has no write on the dirs)
EXTRACT_DIR=/scratch-shared/mmiddelkoop/unthi_zoo
mkdir -p "$EXTRACT_DIR"

# CIFAR-10 already preprocessed (dataset.pt in unthi_cifar10_preprocessed/)
echo "Preprocess MNIST"
unzip -q -o "$ZOO_DIR/unthi_mnist.zip" -d "$EXTRACT_DIR"
python preprocess_dataset_smallcnnzoo.py --in_dir="$EXTRACT_DIR/unthi_mnist/" --out_dir="$ZOO_DIR/unthi_mnist_preprocessed/"
rm -rf "$EXTRACT_DIR/unthi_mnist"

echo "Preprocess CIFAR-10"
unzip -q -o "$ZOO_DIR/unthi_cifar10.zip" -d "$EXTRACT_DIR"
python preprocess_dataset_smallcnnzoo.py --in_dir="$EXTRACT_DIR/unthi_cifar10/" --out_dir="$ZOO_DIR/unthi_cifar10_preprocessed/"
rm -rf "$EXTRACT_DIR/unthi_cifar10"

echo "Preprocess Fashion-MNIST"
unzip -q -o "$ZOO_DIR/unthi_fmnist.zip" -d "$EXTRACT_DIR"
python preprocess_dataset_smallcnnzoo.py --in_dir="$EXTRACT_DIR/unthi_fmnist/" --out_dir="$ZOO_DIR/unthi_fmnist_preprocessed/"
rm -rf "$EXTRACT_DIR/unthi_fmnist"

echo "Preprocess SVHN"
unzip -q -o "$ZOO_DIR/unthi_svhn.zip" -d "$EXTRACT_DIR"
python preprocess_dataset_smallcnnzoo.py --in_dir="$EXTRACT_DIR/unthi_svhn/" --out_dir="$ZOO_DIR/unthi_svhn_preprocessed/"
rm -rf "$EXTRACT_DIR/unthi_svhn"
