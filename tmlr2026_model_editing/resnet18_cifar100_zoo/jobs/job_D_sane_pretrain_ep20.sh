#SBATCH --partition=<gpu_partition>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:30:00
# Adapted-config SANE AE (d_model 1024, 4+4 gpt2 layers, 8 heads, latent 128, window 512, token width 289) on the epoch-20 ResNet-18 zoo,
# 700 train models of the GMN-study split, same args as the epoch-60 adapted run (sane_s0.json): seed 0, 50 epochs, batch 64, lr 1e-4 OneCycle,
# wd 3e-9, gamma 0.05, tau 0.1, noise 0.1, n_val 20, time_limit 9600 s.
source $HOME/sane_zoo/.venv/bin/activate   # scratch venv lost files to the /path/to/scratch clean-up (2026-11-05)
tar xzf sane_ep20_code.tgz
export OMP_NUM_THREADS=8 PYTHONPATH=$PWD/code ZOO_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/path/to/scratch/gmn_resnet; SRC=$G/zoo_ep20; R=$G/sane_ep20; ARCH=$G/results_ep20_sane
mkdir -p logs results compact
echo "node=$(hostname) start=$(date)"; nvidia-smi -L || { echo "no GPU"; exit 3; }
T() { echo "== $(date +%H:%M:%S) $*"; }
mkdir -p $R/runs
T tokens; python code/build_tokens_ep20.py $SRC $R > logs/tokens.txt 2>&1; tail -n 4 logs/tokens.txt; cp $R/tokens_info.json . 2>/dev/null || true
T split check
python - <<'PY'
import numpy as np, json, sys
m = np.load("/path/to/scratch/gmn_resnet/sane_ep20/zoo_meta.npz", allow_pickle=True); sp = m["split"]
te = np.where(sp == "test")[0]; ref = np.load("test_model_ids_ep20.npy")
ok = np.array_equal(np.sort(te), np.sort(ref)); cnt = {s: int((sp == s).sum()) for s in np.unique(sp)}
json.dump({"split_counts": cnt, "test_ids_match_gmn_ep20": bool(ok), "split_id": str(m["split_id"]), "epoch": int(m["epoch"])}, open("split_check.json", "w"), indent=1)
print(cnt, "test ids match GMN study:", ok, "epoch", int(m["epoch"])); sys.exit(0 if ok and cnt.get("train") == 700 else 4)
PY
T pretrain
python code/run_pretrain.py $R 0 50 --d_model 1024 --layers 4 --lat 128 --nhead 8 --window 512 --batch 64 --lr 1e-4 --wd 3e-9 --gamma 0.05 --temperature 0.1 --noise 0.1 --n_val 20 --time_limit 9600 2>&1 | grep --line-buffered -v Warning > logs/pretrain_s0.txt
tail -n 6 logs/pretrain_s0.txt
T recon eval
python code/eval_recon_test.py $R runs/sane_s0.pt sane_s0_recon_eval.json 2>&1 | grep --line-buffered -v Warning > logs/recon_eval.txt; tail -n 4 logs/recon_eval.txt
cp runs/sane_s0.pt runs/sane_s0.json runs/sane_s0_emb.npz sane_s0_recon_eval.json $R/runs/
cp runs/sane_s0.json .
python -c "import numpy as np; e=np.load('runs/sane_s0_emb.npz', allow_pickle=True); np.savez('sane_s0_zbar.npz', zbar=e['zbar'], zstd=e['zstd'], split=e['split'], recall=e['recall'])"
ls -lh runs/ $R/runs/ *.json *.npz || true
echo "end=$(date)"
