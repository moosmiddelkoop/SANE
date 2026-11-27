tar xzf code.tgz
export ZOO=__ZOO__ ZOO_DEVICE=cuda OMP_NUM_THREADS=2 KMP_AFFINITY=disabled
T() { echo "== $(date +%H:%M:%S) $*"; }
ls -lL zoo_theta.npy zoo_meta.npz edit_sets.npz __IMG__
mkdir -p runs results
# ---- 1. metanets (two GPU processes at a time); seed 1 nets are the independent judges
T metanets_seed0
python run_meta.py mlp_paper 0 30 > runs/log_mlp_s0.txt 2>&1 & P1=$!
python run_meta.py graph1 0 30 > runs/log_graph_s0.txt 2>&1 & P2=$!
wait $P1 || echo FAILED mlp_s0; wait $P2 || echo FAILED graph_s0
tail -1 runs/log_mlp_s0.txt runs/log_graph_s0.txt
T metanets_seed1
python run_meta.py mlp_paper 1 30 > runs/log_mlp_s1.txt 2>&1 & P1=$!
python run_meta.py graph1 1 30 > runs/log_graph_s1.txt 2>&1 & P2=$!
OMP_NUM_THREADS=2 python baselines_meta.py > runs/log_baselines.txt 2>&1 & P3=$!
wait $P1 || echo FAILED mlp_s1; wait $P2 || echo FAILED graph_s1; wait $P3 || echo FAILED baselines
tail -1 runs/log_mlp_s1.txt runs/log_graph_s1.txt runs/log_baselines.txt
# ---- 2. edits, priority order; each line is independent and skips if its result exists
G=graph1_s0_e30; M=mlp_paper_s0_e30; JG=graph1_s1_e30,mlp_paper_s0_e30; JM=mlp_paper_s1_e30,graph1_s0_e30
cat > jobs_edit.txt <<EOF
meta:$G unlearn --judges $JG --save_delta --traj_all
meta:$M unlearn --judges $JM --save_delta --traj_all --no_derived
meta_out:$G unlearn --judges $JG --save_delta --traj_all
ga_real unlearn --save_delta
ft_retain unlearn --save_delta
meta:$G uplearn --judges $JG --save_delta --traj_all
meta:$M uplearn --judges $JM --save_delta --traj_all --no_derived
meta_out:$G uplearn --judges $JG --save_delta --traj_all
ga_real uplearn --save_delta
ft_retain uplearn --save_delta
meta_raw:$M unlearn --save_delta --no_derived
meta_raw:$M uplearn --save_delta --no_derived
EOF
T edits
run_edit_line() { echo "-- $(date +%H:%M:%S) start $*"; python run_edit.py $* --n_per_class 100 > "results/log_$(echo $* | tr ' :' '__' | cut -c1-80).txt" 2>&1 || echo "FAILED edit: $*"; echo "-- $(date +%H:%M:%S) end $*"; }
export -f run_edit_line
cat jobs_edit.txt | xargs -P __NPROC__ -I{} bash -c 'run_edit_line {}'
grep -h -E "DONE|FAILED|warn" results/log_*.txt | tail -30
# ---- 3. robustness: relearning attack + linear-probe recoverability on every result file with edit vectors
T relearn
ls results/*.npz | grep -v relearn_ | xargs -P __NPROC__ -I{} bash -c 'python relearn_cnn.py {} > "results/log_relearn_$(basename {} .npz).txt" 2>&1 || echo "FAILED relearn {}"'
grep -h -E "DONE|FAILED" results/log_relearn_*.txt | tail -20
T done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
du -sh results runs || true; ls -lh results/*.npz | head -40 || true