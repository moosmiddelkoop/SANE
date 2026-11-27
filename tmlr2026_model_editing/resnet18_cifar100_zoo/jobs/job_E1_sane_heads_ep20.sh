#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --kill-on-invalid-dep=yes
# Heads on the epoch-20 SANE AE (seed 0): ridge linear probe on zbar, MLP head on zbar, token head [zbar, z_fc(c)], ridge baselines on s(W)
# and fc-row statistics, zoo-mean constant; selection on val R2x, scoring = GMN Tier A metrics + 2000-resample bootstrap on the 150 test models.
# Then a 2-edit smoke test of both SANE-W editor specs and the edit-index relearn (the edit jobs depend on this job succeeding).
source $HOME/sane_zoo/.venv/bin/activate   # scratch venv lost files to the /path/to/scratch clean-up (2026-11-05)
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; SRC=$G/zoo_ep20; R=$G/sane_ep20; ARCH=$G/results_ep20_sane
mkdir -p logs results compact
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
T features; python code/zoo_features.py $R > logs/features.txt 2>&1; tail -n 2 logs/features.txt; cp zoo_features.npz $R/
T heads
python code/fit_heads_ep20.py $R/runs/sane_s0_emb.npz zoo_features.npz heads_ep20_s0 --seed 0 --epochs 300 --save_heads heads_s0.pt --test_ids test_model_ids_ep20.npy --n_boot 2000 > logs/heads.txt 2>&1
tail -n 14 logs/heads.txt; cp heads_s0.pt heads_ep20_s0_* $R/runs/
T smoke editors
export ROOT=$R RUNS=$G/runs_ep20 EDITOR_SEED=0 SANE_RUNS=$R/runs; cp $R/edit_sets.npz .
python code/run_edit_sane_gmn.py meta_fc:sane_token_last unlearn --classes 0 --n_per_class 2 --max_steps 3 --log_every 1 --batch 2 --judges gmn@s3 --tag smoke_fc > logs/smoke_fc.txt 2>&1; tail -n 2 logs/smoke_fc.txt
python code/run_edit_sane_gmn.py meta:sane_token_full unlearn --classes 0 --n_per_class 2 --max_steps 2 --log_every 1 --batch 2 --judges gmn@s3 --delta_full_n 1 --tag smoke_full > logs/smoke_full.txt 2>&1; tail -n 2 logs/smoke_full.txt
python code/relearn_resnet_fixidx.py results/smoke_full.npz --n_per_class 1 --ks 1 --lr_mult 1e-3 --methods main > logs/smoke_relearn.txt 2>&1; tail -n 1 logs/smoke_relearn.txt
ls -lh heads_ep20_s0_* results/ || true
echo "end=$(date)"
