#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
# Re-run the relearning attack (k=1,5,20, Adam lr 1e-3*rms theta, 1 edit/class) for the epoch-20 GMN-study full-parameter edits with the
# stored full deltas keyed by EDIT INDEX (relearn_resnet.py keyed them by model id). Methods: psi full (beta 0.1), psi full ORACLE, GA*, FT*.
source $HOME/sane_zoo/.venv/bin/activate   # scratch venv lost files to the /path/to/scratch clean-up (2026-11-05)
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; SRC=$G/zoo_ep20; R=$G/sane_ep20; ARCH=$G/results_ep20_sane
mkdir -p logs results compact
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
export ROOT=$SRC
for k in meta_gmn meta_gmn_oracle ga_real ft_retain; do
  f=$G/results_ep20/${k}_unlearn_test.npz; T relearn $k
  python code/relearn_resnet_fixidx.py $f --n_per_class 1 --ks 1,5,20 --lr_mult 1e-3 --methods main > logs/relearn_fixidx_$k.txt 2>&1 || echo "FAILED $k"; tail -n 1 logs/relearn_fixidx_$k.txt
done
mkdir -p $G/results_ep20/relearn_fixidx; cp results/relearn_*.npz $G/results_ep20/relearn_fixidx/; ls -lh results/
echo "end=$(date)"
