#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --kill-on-invalid-dep=yes
# full-gradient SANE-W unlearning (token head, all 77 windows re-encoded per step), psi-stop, GMN protocol (eta 1e-3*||theta0||, beta 0.1,
# <=1000 steps, 6/class, batch 8), derived controls incl. random_full_matched, judge graph net s3; relearn of the stored full deltas.
source $HOME/sane_zoo/.venv/bin/activate   # scratch venv lost files to the /path/to/scratch clean-up (2026-11-05)
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; SRC=$G/zoo_ep20; R=$G/sane_ep20; ARCH=$G/results_ep20_sane
mkdir -p logs results compact
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
export ROOT=$R RUNS=$G/runs_ep20 EDITOR_SEED=0 SANE_RUNS=$R/runs; cp $R/edit_sets.npz .; mkdir -p $ARCH
run() { tag=$1; shift; T start $tag; python code/run_edit_sane_gmn.py "$@" --tag $tag > logs/log_$tag.txt 2>&1 || echo "FAILED $tag"; grep -h -E "DONE|Error|error" logs/log_$tag.txt | tail -n 3; }
run meta_sane_full_unlearn_test meta:sane_token_full unlearn --n_per_class 6 --beta 0.1 --eta 1e-3 --max_steps 1000 --log_every 50 --batch 8 --judges gmn@s3 --delta_full_n 1 --time_limit 16200
T relearn
f=results/meta_sane_full_unlearn_test.npz; [ -f $f ] && { python code/relearn_resnet_fixidx.py $f --n_per_class 1 --ks 1,5,20 --lr_mult 1e-3 > logs/log_relearn_$(basename $f .npz).txt 2>&1 || echo "FAILED relearn"; tail -n 1 logs/log_relearn_$(basename $f .npz).txt; }
T archive; cp results/*.npz $ARCH/; python code/compact_results.py compact results/*.npz; cp logs/*.txt compact/; ls -lh compact | head -n 20
echo "end=$(date)"
