#!/bin/bash
#SBATCH --job-name=hello-gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=hello-gpu-%j.out
#SBATCH --error=hello-gpu-%j.err

# The simplest GPU job: say which node and GPU we got, then exit. Needs no conda and no data.
# Defaults above are for ONE L40S. Pick the GPU in one of two ways (see docs/slurm_first_script.md):
#   edit the two lines   --partition=...   and   --gres=...
#   or override at submit time, with no edit:
#     L40S:  sbatch hello_gpu.sh
#     H100:  sbatch --partition=h100 --gres=gpu:h100:1 hello_gpu.sh
# Do NOT use  --export=ALL,VAR=...  on the H100 partition (jobs get held); set variables in the shell instead.
# All #SBATCH lines must come before the first command, or SLURM ignores them.

echo "job id:   $SLURM_JOB_ID"
echo "node:     $(hostname)"
echo "gpus:     $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo OK
