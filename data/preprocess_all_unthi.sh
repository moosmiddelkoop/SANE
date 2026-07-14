#!/bin/bash
#SBATCH --job-name=preprocess_all_unthi
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/data/logs/preprocess_all_unthi_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/data"
source "$HOME/SANE/.venv/bin/activate"

echo "Preprocess CIFAR-10"
# unzip -q -o /projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10.zip -d /projects/prjs2156/shared/wsl/unthi_zoo/
python preprocess_dataset_smallcnnzoo.py --in_dir=/projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10/ --out_dir=/projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10_preprocessed/
rm -rf /projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10

echo "Preprocess Fashion-MNIST"
unzip -q -o /projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist.zip -d /projects/prjs2156/shared/wsl/unthi_zoo/
python preprocess_dataset_smallcnnzoo.py --in_dir=/projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist/ --out_dir=/projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist_preprocessed/
rm -rf /projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist

echo "Preprocess SVHN"
unzip -q -o /projects/prjs2156/shared/wsl/unthi_zoo/unthi_svhn.zip -d /projects/prjs2156/shared/wsl/unthi_zoo/
python preprocess_dataset_smallcnnzoo.py --in_dir=/projects/prjs2156/shared/wsl/unthi_zoo/unthi_svhn/ --out_dir=/projects/prjs2156/shared/wsl/unthi_zoo/unthi_svhn_preprocessed/
rm -rf /projects/prjs2156/shared/wsl/unthi_zoo/unthi_svhn
