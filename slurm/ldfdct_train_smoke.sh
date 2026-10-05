#!/bin/bash
#SBATCH --job-name=fastddpm-ldfdct-train-smoke
#SBATCH --partition=interactive
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=fastddpm-ldfdct-train-smoke-%j.out
#SBATCH --error=fastddpm-ldfdct-train-smoke-%j.err

source ~/.bashrc
conda activate fastddpm

cd ~/ECE_551_problems_ML/fast-ddpm

python fast_ddpm_main.py \
  --config ldfd_train_smoke.yml \
  --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp \
  --doc ldfdct_train_smoke \
  --scheduler_type uniform \
  --timesteps 10
