#!/bin/bash
#SBATCH --job-name=fastddpm-profile
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=fastddpm-profile-%j.out
#SBATCH --error=fastddpm-profile-%j.err

# Phase 0 profile: parameter count, true GPU memory, data-loader speed, and where one training step's
# time goes (see scripts/phase0_profile.py). One GPU, a few minutes, no checkpoints written.
#   H100:  sbatch phase0_profile.sh
#   L40S:  BATCH=12 sbatch --partition=l40s --gres=gpu:l40s:1 phase0_profile.sh   (batch 16 does not fit)
# Variables go through the shell, not --export (which holds jobs on the h100 partition).
BATCH=${BATCH:-16}
STEPS=${STEPS:-60}
OUT=$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -oE 'L40S|H100|A100' | head -1)
TAG=${GPU:-GPU}_bs${BATCH}
echo "=== job $SLURM_JOB_ID | node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) | $(nproc) CPUs | tag $TAG ==="

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > $OUT/profile_smi_${TAG}.csv &
SMI=$!

python ~/ECE_551_problems_ML/scripts/phase0_profile.py --batch $BATCH --steps $STEPS 2>&1 | tee $OUT/phase0_profile_${TAG}.txt

kill $SMI

python - "$OUT/profile_smi_${TAG}.csv" <<'PY' | tee -a $OUT/phase0_profile_${TAG}.txt
import statistics, sys
rows = [tuple(int(v) for v in l.replace(",", " ").split()) for l in open(sys.argv[1]) if l.strip()]
peak = max(m for m, u in rows)
act = [u for m, u in rows if m >= 0.5 * peak]       # samples while the model was on the GPU
print(f"[nvidia-smi] GPU utilization while training: median {statistics.median(act):g}%, "
      f"min {min(act)}%, max {max(act)}% over {len(act)} one-second samples; peak memory {peak:,} MiB")
PY
