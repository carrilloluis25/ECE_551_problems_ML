#!/bin/bash
#SBATCH --job-name=fastddpm-base
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=fastddpm-base-%j.out
#SBATCH --error=fastddpm-base-%j.err

# Baseline: Fast-DDPM LDFDCT denoising from scratch, paper settings, one H100.
#   batch 16, lr 2e-5 (the authors' checkpoint stores lr 2e-5), uniform scheduler, 10 time steps,
#   TOTAL=500000 steps (the authors' checkpoint is at step 500000).
# One job runs ONE chunk (CHUNK steps, ~12 h at 0.342 s/step) and stops cleanly at the chunk's end.
# Start it through submit_baseline.sh, which chains the chunks and the evaluations. Run by hand to
# restart a chunk: it resumes from <exp>/logs/<DOC>/ckpt.pth if that exists, otherwise starts fresh
# (a fresh start deletes an existing <DOC> folder that has no ckpt.pth).
# Needs patches/0001-training-controls.patch applied to the fast-ddpm fork.
TOTAL=${TOTAL:-500000}
CHUNK=${CHUNK:-125000}
DOC=${DOC:-baseline_bs16_lr2e-5}
EXP=~/carc-scratch/fastddpm-exp
LOGDIR=$EXP/logs/$DOC
CFG=baseline_bs16_lr2e-5.yml

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

# fail fast, with a clear message, if scratch refuses writes (group quota exceeded: chunks 3 and 4 died of this)
dd if=/dev/zero of=$EXP/.quota_probe bs=1M count=100 2>/dev/null && rm -f $EXP/.quota_probe || { rm -f $EXP/.quota_probe; echo "ERROR: cannot write to $EXP (scratch group quota exceeded?). Free space first."; exit 1; }

grep -q "max_steps" fast_ddpm_main.py || { echo "ERROR: apply patches/0001-training-controls.patch first"; exit 1; }

# 0.00002, not 2e-5: PyYAML reads 2e-5 as a string
sed "/^training:/,/^sampling:/ s/batch_size: .*/batch_size: 16/; /^training:/,/^sampling:/ s/snapshot_freq: .*/snapshot_freq: 25000/; /^optim:/,\$ s/^\( *lr:\) .*/\1 0.00002/" configs/ldfd_train_smoke.yml > configs/$CFG

RESUME=""
START=0
if [ -e "$LOGDIR/ckpt.pth" ]; then
  START=$(python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)[3])" "$LOGDIR/ckpt.pth")
  RESUME="--resume_training"
fi
if [ "$START" -ge "$TOTAL" ]; then echo "already at step $START >= $TOTAL, nothing to do"; exit 0; fi
TARGET=$(( (START / CHUNK + 1) * CHUNK ))
[ "$TARGET" -gt "$TOTAL" ] && TARGET=$TOTAL

echo "=== job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=$CUDA_VISIBLE_DEVICES | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) ==="
echo "=== doc $DOC | step $START -> $TARGET (total $TOTAL) | ${RESUME:-fresh start} ==="
grep -nE "^training:|^optim:|batch_size|snapshot_freq| lr:" configs/$CFG

python fast_ddpm_main.py \
  --config $CFG --dataset LDFDCT \
  --exp $EXP --doc $DOC \
  --scheduler_type uniform --timesteps 10 \
  --max_steps $TARGET --no_first_ckpt $RESUME
RC=$?     # the job's exit code must be python's, or --dependency=afterok would continue after a crash

# python's exit code alone is NOT enough: on 2026-11-10 a chunk crashed on "Disk quota exceeded" at step
# 350,511 and still reported exit 0, so the chain went on (the fork's main apparently catches exceptions).
# Trust the checkpoint instead: the chunk only counts as done if ckpt.pth reached the chunk's TARGET step.
if [ "$RC" -eq 0 ]; then
  END=$(python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)[3])" "$LOGDIR/ckpt.pth" 2>/dev/null)
  if [ "${END:-0}" -lt "$TARGET" ]; then
    echo "ERROR: python exited 0 but ckpt.pth is at step ${END:-unreadable}, short of the target $TARGET (crash swallowed?)"
    RC=1
  fi
fi

echo "=== python exit code $RC | $(grep 'step:' $LOGDIR/stdout.txt | tail -1) ==="
ls -lh $LOGDIR | tail -n +2
exit $RC
