#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
# Per-layer norm-matched random-vector control (edit_resnet.random_layer_matched; 21 groups = 20 conv(+BN) + fc) for the STORED epoch-20
# graph-net full-gradient edits (psi-stop and ORACLE stop), using the exact stored parent deltas (delta_full16, 109 edits per run).
# The editor checkpoint runs_ep20/s2 no longer exists (scratch clean-up), so no re-walk. psi-stop run: + relearn k=1,5,20 (100 edits, 1/class).
source $HOME/sane_zoo/.venv/bin/activate
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; export ROOT=$G/zoo_ep20
mkdir -p logs out
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
python code/build_layer_rv_ep20.py $G/results_ep20/meta_gmn_unlearn_test.npz out/layer_rv_meta_gmn_unlearn_test.npz --relearn > logs/layer_rv_psi.txt 2>&1; tail -n 4 logs/layer_rv_psi.txt
python code/build_layer_rv_ep20.py $G/results_ep20/meta_gmn_oracle_unlearn_test.npz out/layer_rv_meta_gmn_oracle_unlearn_test.npz > logs/layer_rv_oracle.txt 2>&1; tail -n 3 logs/layer_rv_oracle.txt
ls -lh out; echo "end=$(date)"
