#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

# Unlearning edits on CIFAR-100 ResNet-18 zoo epoch __E__ with the graph net (editor seed __SA__, judge seed __SB__) + controls + baselines + relearn
source /path/to/scratch/gmn_resnet/.venv/bin/activate
tar xzf gmn_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/gmn_code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E=__E__; SA=__SA__; SB=__SB__
export ROOT=/path/to/scratch/gmn_resnet/zoo_ep$E RUNS=/path/to/scratch/gmn_resnet/runs_ep$E EDITOR_SEED=$SA
ARCH=/path/to/scratch/gmn_resnet/results_ep$E; mkdir -p results compact $ARCH
cp $ROOT/edit_sets.npz .
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo 'no GPU'; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
run() { tag=$1; shift; T start $tag; python gmn_code/run_edit_gmn.py "$@" --tag $tag > results/log_$tag.txt 2>&1 || echo "FAILED $tag"; grep -h -E "DONE|Error|error" results/log_$tag.txt | tail -n 3; }
# metanet edits (editor seed), judge = second seed; eta 1e-3 normalised, <=1000 steps, trajectories every 50 steps
run meta_gmn_unlearn_test            meta:gmn    unlearn --n_per_class 6 --beta 0.1   --eta 1e-3 --max_steps 1000 --log_every 50 --batch 40 --judges gmn@s$SB --delta_full_n 1 --time_limit 7200
run meta_fc_gmn_unlearn_test         meta_fc:gmn unlearn --n_per_class 6 --beta 0.1   --eta 1e-3 --max_steps 1000 --log_every 50 --batch 40 --judges gmn@s$SB --time_limit 4800
run meta_gmn_bnorm_unlearn_test      meta:gmn    unlearn --n_per_class 3 --beta 0.005 --eta 1e-3 --max_steps 1000 --log_every 50 --batch 40 --judges gmn@s$SB --delta_full_n 1 --time_limit 3600
# data baselines (small budget): 2 per class, lr 1e-3 * rms(theta)
run ga_real_unlearn_test             ga_real     unlearn --n_per_class 2 --lr_mult 1e-3 --ga_steps 200 --batch 40 --judges gmn@s$SB --time_limit 1800
run ft_retain_unlearn_test           ft_retain   unlearn --n_per_class 2 --lr_mult 1e-3 --ft_steps 100 --batch 40 --judges gmn@s$SB --time_limit 1800
T relearn
for f in results/meta_gmn_unlearn_test.npz results/meta_fc_gmn_unlearn_test.npz results/ga_real_unlearn_test.npz results/ft_retain_unlearn_test.npz; do
  [ -f $f ] && { python gmn_code/relearn_resnet.py $f --n_per_class 1 --ks 1,5,20 --lr_mult 1e-3 > results/log_relearn_$(basename $f .npz).txt 2>&1 || echo "FAILED relearn $f"; tail -n 1 results/log_relearn_$(basename $f .npz).txt; }
done
T analyze
python gmn_code/analyze_edits.py results ep${E} > analyze_ep${E}.txt 2>&1 || echo "FAILED analyze"; tail -n 25 analyze_ep${E}.txt
T archive
cp results/*.npz $ARCH/ ; python gmn_code/compact_results.py compact_ep$E results/*.npz; cp results/log_*.txt compact_ep$E/
du -sh results compact_ep$E $ARCH; ls -lh compact_ep$E | head -n 20
echo "end=$(date)"
