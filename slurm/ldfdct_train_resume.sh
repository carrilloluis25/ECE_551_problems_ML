#!/bin/bash
#SBATCH --job-name=fastddpm-ldfdct-train-resume
#SBATCH --partition=interactive
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=fastddpm-ldfdct-train-resume-%j.out
#SBATCH --error=fastddpm-ldfdct-train-resume-%j.err

source ~/.bashrc
conda activate fastddpm

cd ~/ECE_551_problems_ML/fast-ddpm

python fast_ddpm_main.py \
  --config ldfd_train_smoke.yml \
  --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp \
  --doc ldfdct_train_smoke \
  --scheduler_type uniform \
  --timesteps 10 --resume_training
