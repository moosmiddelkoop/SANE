#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00

# Graph net on CIFAR-100 ResNet-18 zoo epoch __E__: seeds __SA__ (editor, +reliability) and __SB__ (judge), 300 epochs each
source /path/to/scratch/gmn_resnet/.venv/bin/activate
tar xzf gmn_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/gmn_code
E=__E__; SA=__SA__; SB=__SB__
ROOT=/path/to/scratch/gmn_resnet/zoo_ep$E; RUNS=/path/to/scratch/gmn_resnet/runs_ep$E
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo 'no GPU'; exit 3; }
mkdir -p $RUNS/s$SA $RUNS/s$SB
python gmn_code/train_gmn.py $ROOT $RUNS/s$SA --seed $SA --epochs 300 --reliability > train_ep${E}_s$SA.log 2>&1 || echo "FAILED seed $SA"
python gmn_code/train_gmn.py $ROOT $RUNS/s$SB --seed $SB --epochs 300 > train_ep${E}_s$SB.log 2>&1 || echo "FAILED seed $SB"
for S in $SA $SB; do cp $RUNS/s$S/gmn_d64_s${S}_e300_results.json $RUNS/s$S/gmn_d64_s${S}_e300_history.json . 2>/dev/null || true
  for f in $RUNS/s$S/gmn_d64_s${S}_e300_*pred_test.npy; do cp $f ep${E}_$(basename $f); done; done
cp $RUNS/s$SA/reliability_test_models.json reliability_ep$E.json 2>/dev/null || true; cp $RUNS/s$SA/test_model_ids.npy test_model_ids_ep$E.npy
grep -h -E "DONE|best_val|final|FAILED" train_ep${E}_s*.log | tail -n 12
echo "end=$(date)"
