#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
# Per-layer norm-matched random-vector control for the STORED epoch-20 SANE-W full-gradient psi-stop edits (exact stored parent deltas
# delta_full16, first edit of each class per 8-edit batch), same script as the GMN rows (build_layer_rv_ep20.py). Three parent files
# (E3a classes 0-69, completion A classes 69-84, B classes 85-99); seed base differs per file because edit indices restart at 0.
source $HOME/sane_zoo/.venv/bin/activate
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; export ROOT=$G/zoo_ep20; S=$G/results_ep20_sane
mkdir -p logs out
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
python code/build_layer_rv_ep20.py $S/meta_sane_full_unlearn_test.npz      out/layer_rv_meta_sane_full_unlearn_test.npz      --seed_base 30261112 > logs/lrv_e3a.txt 2>&1; tail -n 3 logs/lrv_e3a.txt
python code/build_layer_rv_ep20.py $S/meta_sane_full_unlearn_test_clsA.npz out/layer_rv_meta_sane_full_unlearn_test_clsA.npz --seed_base 31261112 > logs/lrv_A.txt 2>&1; tail -n 3 logs/lrv_A.txt
python code/build_layer_rv_ep20.py $S/meta_sane_full_unlearn_test_clsB.npz out/layer_rv_meta_sane_full_unlearn_test_clsB.npz --seed_base 32261112 > logs/lrv_B.txt 2>&1; tail -n 3 logs/lrv_B.txt
ls -lh out; echo "end=$(date)"
