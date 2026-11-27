#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00

# Oracle-stop unlearning: ignore psi stop, freeze each edit when TRUE recall_c meets the criterion (checked every 5 steps, <=200 steps); all 4 epochs
source /path/to/scratch/gmn_resnet/.venv/bin/activate
tar xzf gmn_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/gmn_code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo 'no GPU'; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
for ES in "10 0 1" "20 2 3" "30 4 5" "40 6 7"; do
  set -- $ES; E=$1; SA=$2; SB=$3
  export ROOT=/path/to/scratch/gmn_resnet/zoo_ep$E RUNS=/path/to/scratch/gmn_resnet/runs_ep$E EDITOR_SEED=$SA
  [ -f $RUNS/s$SA/gmn_d64_s${SA}_e300_best_val_r2x.pt ] || { echo "SKIP ep$E: no editor checkpoint"; continue; }
  ARCH=/path/to/scratch/gmn_resnet/results_ep$E; mkdir -p $ARCH compact_ep$E
  rm -rf results; mkdir -p results; cp $ROOT/edit_sets.npz .
  T ep$E full oracle stop
  python gmn_code/run_edit_gmn.py meta:gmn unlearn --n_per_class 6 --beta 0.1 --eta 1e-3 --max_steps 200 --log_every 5 --batch 40 --judges gmn@s$SB --delta_full_n 1 --oracle_stop --time_limit 3000 --tag meta_gmn_oracle_unlearn_test > results/log_meta_gmn_oracle_unlearn_test.txt 2>&1 || echo "FAILED ep$E fixed full"
  grep -h -E "DONE|Error" results/log_meta_gmn_oracle_unlearn_test.txt | tail -n 2
  T ep$E fc oracle stop
  python gmn_code/run_edit_gmn.py meta_fc:gmn unlearn --beta 0.1 --eta 1e-3 --n_per_class 6 --max_steps 200 --log_every 5 --batch 40 --judges gmn@s$SB --oracle_stop --time_limit 2400 --tag meta_fc_gmn_oracle_unlearn_test > results/log_meta_fc_gmn_oracle_unlearn_test.txt 2>&1 || echo "FAILED ep$E fixed fc"
  grep -h -E "DONE|Error" results/log_meta_fc_gmn_oracle_unlearn_test.txt | tail -n 2
  T ep$E relearn oracle edits
  for f in results/meta_gmn_oracle_unlearn_test.npz results/meta_fc_gmn_oracle_unlearn_test.npz; do [ -f $f ] && { python gmn_code/relearn_resnet.py $f --n_per_class 1 --ks 1,5,20 --lr_mult 1e-3 > results/log_relearn_$(basename $f .npz).txt 2>&1 || echo "FAILED relearn $f"; tail -n 1 results/log_relearn_$(basename $f .npz).txt; }; done
  cp results/*.npz $ARCH/; python gmn_code/compact_results.py compact_ep$E results/*.npz; cp results/log_*.txt compact_ep$E/
done
ls -lh compact_ep*/ | head -n 30
echo "end=$(date)"
