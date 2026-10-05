#!/bin/bash
#SBATCH --job-name=fastddpm-ldfdct-smoke
#SBATCH --partition=interactive
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fastddpm-ldfdct-smoke-%j.out
#SBATCH --error=fastddpm-ldfdct-smoke-%j.err

source ~/.bashrc
conda activate fastddpm

cd ~/ECE_551_problems_ML/fast-ddpm

python fast_ddpm_main.py \
  --config ldfd_smoke_test.yml \
  --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp \
  --doc ldfdct_smoke_test \
  --sample --fid \
  --scheduler_type uniform \
  --timesteps 10
