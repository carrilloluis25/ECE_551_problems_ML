#!/bin/bash
#SBATCH --job-name=fastddpm-lr
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=fastddpm-lr-%A_%a.out
#SBATCH --error=fastddpm-lr-%A_%a.err

# Learning-rate test: the paper says 2e-4, the repo configs say 2e-5.
# Two short runs at the paper's batch size (16), identical except for the learning rate.
# Needs patches/0001-training-controls.patch applied to the fast-ddpm fork (--max_steps etc.).
LRS=(0.0002 0.00002)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
STEPS=${STEPS:-10000}
DOC=lr_test_${LR}

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

grep -q "max_steps" fast_ddpm_main.py || { echo "ERROR: apply patches/0001-training-controls.patch first"; exit 1; }

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) | lr $LR | steps $STEPS ==="

CFG=lr_test_${LR}.yml
sed "/^training:/,/^sampling:/ s/batch_size: .*/batch_size: 16/; /^training:/,/^sampling:/ s/snapshot_freq: .*/snapshot_freq: 5000/; /^optim:/,\$ s/^\( *lr:\) .*/\1 ${LR}/" configs/ldfd_train_smoke.yml > configs/$CFG
grep -nE "^training:|^optim:|batch_size|snapshot_freq| lr:" configs/$CFG

python fast_ddpm_main.py \
  --config $CFG --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp --doc $DOC \
  --scheduler_type uniform --timesteps 10 \
  --max_steps $STEPS --no_first_ckpt
