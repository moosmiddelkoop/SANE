#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
# Completion of E3a (Slurm 28924999, which hit its time limit after classes 0-68 + 4 edits of class 69): full-gradient SANE-W unlearning,
# psi-stop, identical protocol (eta 1e-3*||theta0||, Boost beta 0.1, stop psi_c <= 0.4 psi_c(theta0), <=1000 steps, 6/class, batch 8,
# delta_full_n 1, norm-matched controls, judge graph net s3), restricted to classes 85,86,87,88,89,90,91,92,93,94,95,96,97,98,99; relearn k=1,5,20 (edit-index keyed).
source $HOME/sane_zoo/.venv/bin/activate
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; R=$G/sane_ep20; ARCH=$G/results_ep20_sane
mkdir -p logs results compact $ARCH
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
export ROOT=$R RUNS=$G/runs_ep20 EDITOR_SEED=0 SANE_RUNS=$R/runs; cp $R/edit_sets.npz .
TAG=meta_sane_full_unlearn_test_clsB
T start $TAG
python code/run_edit_sane_gmn.py meta:sane_token_full unlearn --classes 85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 --n_per_class 6 --beta 0.1 --eta 1e-3 --max_steps 1000 --log_every 50 --batch 8 --judges gmn@s3 --delta_full_n 1 --time_limit 9000 --tag $TAG > logs/log_$TAG.txt 2>&1
grep -h -E "DONE|time limit|Error" logs/log_$TAG.txt | tail -n 3
T relearn
python code/relearn_resnet_fixidx.py results/$TAG.npz --n_per_class 1 --ks 1,5,20 --lr_mult 1e-3 > logs/log_relearn_$TAG.txt 2>&1; tail -n 1 logs/log_relearn_$TAG.txt
T archive; cp results/*.npz $ARCH/; python code/compact_results.py compact results/*.npz; cp logs/*.txt compact/; ls -lh compact
echo "end=$(date)"
