#SBATCH --time=__TIME__
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=<partition>
#SBATCH -C A4000
#SBATCH --gres=gpu:__NGPU__
__BEGIN__

. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.6/toolkit
module load cuDNN/cuda12.6
. /path/to/scratch/uv-env.sh
source /path/to/venv/bin/activate
echo "node=$(hostname) python=$(which python)"
nvidia-smi -L || { echo 'no GPU visible'; exit 3; }
NG=$(nvidia-smi -L | wc -l); echo "GPUs: $NG"
tar xzf code_sane.tgz
export ZOO=__ZOO__ ZOO_DEVICE=cuda OMP_NUM_THREADS=2 KMP_AFFINITY=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T() { echo "== $(date +%H:%M:%S) $*"; }
E=__EPOCHS__; NPC=__NPC__; NPC2=__NPC2__; MAXS=__MAXSTEPS__; CALN=__CALN__; CLS="__CLS__"; ETAS="__ETAS__"
mkdir -p runs results
for f in sane_perms.npy graph1_s0_e30.pt graph1_s1_e30.pt mlp_paper_s0_e30.pt; do [ -e $f ] && cp -L $f runs/$f; done
ls -lL zoo_theta.npy zoo_meta.npz edit_sets.npz __IMG__ runs/ || true
echo '{"skip_bias_line_for": ["sane_"]}' > job_overrides.json
# ---- 1. alignment permutations (shared cache), then SANE pretraining, seeds 0 (editor) and 1 (judge/monitor), one GPU each
T perms; python run_sane_pretrain.py 0 $E --perms_only 2>&1 | grep -v Warning | tail -n 3 || true
T pretrain
CUDA_VISIBLE_DEVICES=0 python run_sane_pretrain.py 0 $E --time_limit __PRELIMIT__ > runs/log_sane_s0.txt 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=$(( 1 % NG )) python run_sane_pretrain.py 1 $E --time_limit __PRELIMIT__ > runs/log_sane_s1.txt 2>&1 & P2=$!
wait $P1 || echo FAILED sane_s0
wait $P2 || echo FAILED sane_s1
grep -h -E "DONE|identity control|head |time limit" runs/log_sane_s0.txt runs/log_sane_s1.txt | cut -c1-400 | tail -n 12 || true
S0=sane_s0_e$E; S1=sane_s1_e$E
G=graph1_s0_e30; M=mlp_paper_s0_e30; JG=graph1_s1_e30
ls runs/$S0.pt runs/$S1.pt || { echo "FAILED: SANE checkpoints missing"; exit 4; }
# ---- 2. eta_z calibration on this zoo's validation split (SANE-Z, token head)
T calib
run_calib() { CUDA_VISIBLE_DEVICES=$(( SLOT % NG )) python run_edit.py zedit:$S0@token $2 --set val --n_per_class $CALN --max_steps $MAXS --eta_z $1 $CLS --tag calib_$2_$1 > results/log_calib_$2_$1.txt 2>&1 || echo "FAILED calib $2 $1"; }
export -f run_calib; export S0 CALN MAXS CLS NG
for eta in $ETAS; do for task in unlearn uplearn; do echo "$eta $task"; done; done | xargs -P $(( 2 * NG )) --process-slot-var=SLOT -L 1 bash -c 'run_calib $0 $1'
ETA=$(python pick_eta_z.py $ETAS); echo "eta_z = $ETA"; cat results/calib_summary.json | head -c 1500 || true
# ---- 3. SANE edits (test split), priority order; manifold-distance monitor = seed-1 SANE for every method
J="$S1@token,$JG,$M"; MON="--sane_monitor $S1"
cat > jobs_edit.txt <<EOF
meta:$S0@token unlearn --judges $J --save_delta --traj_all $MON
zedit:$S0@token unlearn --eta_z $ETA --judges $J --save_delta --traj_all $MON
meta_out:$S0@token unlearn --judges $J --save_delta --traj_all $MON --no_derived
zedit_dense:$S0@token unlearn --eta_z $ETA --judges $J --save_delta --traj_all $MON --no_derived
meta:$S0@token uplearn --judges $J --save_delta --traj_all $MON
zedit:$S0@token uplearn --eta_z $ETA --judges $J --save_delta --traj_all $MON
meta_out:$S0@token uplearn --judges $J --save_delta --traj_all $MON --no_derived
zedit_dense:$S0@token uplearn --eta_z $ETA --judges $J --save_delta --traj_all $MON --no_derived
meta:$G unlearn --judges $S1@token --traj_all --no_derived $MON --n_per_class $NPC2 --tag mon_graph_unlearn
meta:$M unlearn --judges $S1@token --traj_all --no_derived $MON --n_per_class $NPC2 --tag mon_mlp_unlearn
meta_out:$G unlearn --judges $S1@token --traj_all --no_derived $MON --n_per_class $NPC2 --tag mon_graphout_unlearn
zdec:$S0@token unlearn --eta_z $ETA --judges $J --save_delta --traj_all $MON --n_per_class $NPC2
zdec:$S0@token unlearn --eta_z $ETA --zpsi $G --judges $J --save_delta --traj_all $MON --n_per_class $NPC2
meta:$S0@linear unlearn --judges $J --save_delta --traj_all $MON --no_derived
meta:$G uplearn --judges $S1@token --traj_all --no_derived $MON --n_per_class $NPC2 --tag mon_graph_uplearn
meta:$M uplearn --judges $S1@token --traj_all --no_derived $MON --n_per_class $NPC2 --tag mon_mlp_uplearn
zdec:$S0@token uplearn --eta_z $ETA --judges $J --save_delta --traj_all $MON --n_per_class $NPC2
zdec:$S0@token uplearn --eta_z $ETA --zpsi $G --judges $J --save_delta --traj_all $MON --n_per_class $NPC2
EOF
T edits
run_edit_line() { echo "-- $(date +%H:%M:%S) start [gpu $(( SLOT % NG ))] $*"; CUDA_VISIBLE_DEVICES=$(( SLOT % NG )) python run_edit.py --n_per_class $NPC --max_steps $MAXS --log_every 20 $CLS $* > "results/log_$(echo $* | tr ' :@,' '____' | cut -c1-90).txt" 2>&1 || echo "FAILED edit: $*"; echo "-- $(date +%H:%M:%S) end $*"; }
export -f run_edit_line; export NPC MAXS
cat jobs_edit.txt | xargs -P __NPROC__ --process-slot-var=SLOT -I{} bash -c 'run_edit_line {}'
grep -h -E "DONE|FAILED|warn" results/log_*.txt | tail -n 40 || true
# ---- 4. robustness (relearning + linear probe) on the SANE result files with edit vectors
T relearn
run_relearn() { CUDA_VISIBLE_DEVICES=$(( SLOT % NG )) python relearn_cnn.py $1 > "results/log_relearn_$(basename $1 .npz).txt" 2>&1 || echo "FAILED relearn $1"; }
export -f run_relearn
ls results/*.npz | grep -E "sane_s0|zpsi" | grep -v -E "relearn_|calib_" | xargs -P $(( 2 * NG )) --process-slot-var=SLOT -I{} bash -c 'run_relearn {}'
grep -h -E "DONE|FAILED" results/log_relearn_*.txt | tail -n 20 || true
T done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
du -sh results runs || true
ls -lh results/*.npz runs/*.pt | head -n 80 || true
