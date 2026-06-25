#!/bin/bash
# Submit a chain of sbatch jobs that pick up after each other (afterany = continue
# even if the previous job hit walltime). Requires resume="AUTO" in the pretrain
# script so each job continues from the last Ray checkpoint.
set -euo pipefail

JOB1=$(sbatch --parsable run_pretrain.sbatch)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 run_pretrain.sbatch)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 run_pretrain.sbatch)
echo "chain: $JOB1 -> $JOB2 -> $JOB3"
