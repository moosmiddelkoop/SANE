import subprocess, os, time, sys, json
def run_pool(jobs, n_par=6, log="pool.log"):
    """jobs: list of (spec, seed, epochs, frac). Runs single-threaded workers in parallel."""
    env = dict(os.environ, KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NT="1")
    pending = list(jobs); running = []
    with open(log, "a") as lf:
        while pending or running:
            while pending and len(running) < n_par:
                j = pending.pop(0)
                out = f"{os.environ.get('RUNS','runs')}/{j[0]}_s{j[1]}_f{j[3]}.json"
                if os.path.exists(out): continue
                p = subprocess.Popen([sys.executable, "worker.py", j[0], str(j[1]), str(j[2]), str(j[3])],
                                     stdout=lf, stderr=lf, env=env)
                running.append((p, j))
            for p, j in list(running):
                if p.poll() is not None:
                    running.remove((p, j))
                    lf.write(f"DONE {j} rc={p.returncode}\n"); lf.flush()
            time.sleep(2)
