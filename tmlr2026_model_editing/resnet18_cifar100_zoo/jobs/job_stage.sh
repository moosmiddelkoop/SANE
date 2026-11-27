#SBATCH --partition=<partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Stage CIFAR-100 ResNet-18 zoo epochs 10/20/30/40 into flat arrays for the graph net
source /path/to/scratch/gmn_resnet/.venv/bin/activate
tar xzf gmn_code.tgz
export OMP_NUM_THREADS=2 PYTHONPATH=$PWD/gmn_code
ZOO=/path/to/zoos/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform
REF=/path/to/scratch/sane_zoo/cifar100_resnet18/zoo_meta.npz
CIFAR=/path/to/scratch/sane_zoo/cifar100_resnet18/cifar100.npz
OUTROOT=/path/to/scratch/gmn_resnet
echo "node=$(hostname) start=$(date)"
stage() { E=$1; python gmn_code/stage_zoo_epoch.py $ZOO $E $OUTROOT/zoo_ep$E $REF $CIFAR --workers 8 > stage_ep$E.log 2>&1 \
  && python gmn_code/make_edit_sets.py $OUTROOT/zoo_ep$E/zoo_meta.npz $OUTROOT/zoo_ep$E/edit_sets.npz --n 8 --n_val 5 >> stage_ep$E.log 2>&1 \
  || echo "FAILED ep$E"; }
stage 10 & p1=$!; stage 20 & p2=$!; wait $p1; wait $p2
stage 30 & p3=$!; stage 40 & p4=$!; wait $p3; wait $p4
for E in 10 20 30 40; do cp $OUTROOT/zoo_ep$E/stage_info.json stage_info_ep$E.json 2>/dev/null || true; echo "--- ep$E"; tail -n 6 stage_ep$E.log; done
ls -la $OUTROOT/zoo_ep*/ || true
echo "end=$(date)"
