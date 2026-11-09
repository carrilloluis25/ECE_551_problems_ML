#!/bin/bash
#SBATCH --job-name=fastddpm-probe
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=easley053
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=fastddpm-probe-%j.out
#SBATCH --error=fastddpm-probe-%j.err

# What training loss does the AUTHORS' checkpoint sit at, under our training setup?
# Resumes training from their checkpoint for EXTRA_STEPS steps (batch 16) and reports the
# median/mean loss. Compare with our runs (2e-4: median ~3,250 and 2e-5: ~3,310 at step 10,000).
#   - similar level -> the loss plateau says nothing about quality; our runs are just earlier in training
#   - much lower   -> our training is missing something the authors' run had
# The authors' file is only symlinked, never written. Checkpoints made here are deleted at the end.
# Run from the slurm folder: sbatch authors_loss_probe.sh   (EXTRA_STEPS=300 by default)
EXP=~/carc-scratch/fastddpm-exp
DOC=authors_loss_probe
EXTRA=${EXTRA_STEPS:-300}
AUTHORS=$EXP/logs/ldfdct_smoke_test/ckpt_1.pth

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

[ -f "$AUTHORS" ] || { echo "ERROR: $AUTHORS not found"; exit 1; }
[ -f configs/lr_test_0.00002.yml ] || { echo "ERROR: configs/lr_test_0.00002.yml missing (made by lr_test.sh)"; exit 1; }
grep -q "max_steps" fast_ddpm_main.py || { echo "ERROR: apply patches/0001-training-controls.patch first"; exit 1; }

mkdir -p $EXP/logs/$DOC
ln -sfn $AUTHORS $EXP/logs/$DOC/ckpt.pth

N=$(python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)[3])" $AUTHORS)
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | authors' checkpoint is at step $N | running to step $((N + EXTRA)) ==="

python fast_ddpm_main.py \
  --config lr_test_0.00002.yml --dataset LDFDCT \
  --exp $EXP --doc $DOC \
  --scheduler_type uniform --timesteps 10 \
  --resume_training --max_steps $((N + EXTRA)) --no_first_ckpt

python - $EXP/logs/$DOC/stdout.txt <<'PY'
import re, statistics, sys
d = {int(m[1]): float(m[2]) for l in open(sys.argv[1]) if (m := re.search(r"step: (\d+), loss: ([0-9.eE+-]+)", l))}
v = list(d.values())
print(f"authors' checkpoint, {len(v)} steps ({min(d)}..{max(d)}): median loss {statistics.median(v):.1f}  mean {sum(v)/len(v):.1f}  max {max(v):.1f}")
PY

rm -f $EXP/logs/$DOC/ckpt_*.pth $EXP/logs/$DOC/ckpt.pth     # authors' file untouched; only our new copy and the link go
