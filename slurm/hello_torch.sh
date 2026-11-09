#!/bin/bash
#SBATCH --job-name=hello-torch
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=hello-torch-%j.out
#SBATCH --error=hello-torch-%j.err

# Step 2 after hello_gpu.sh: the same job, plus activating the conda env and checking that PyTorch
# can use the GPU. This is the pattern every real script in this project follows.
# L40S:  sbatch hello_torch.sh
# H100:  sbatch --partition=h100 --gres=gpu:h100:1 hello_torch.sh

source ~/.bashrc            # a batch job starts a fresh shell: load conda first
conda activate fastddpm

echo "node: $(hostname)   gpus: $CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"{name} | capability {cap} | {mem:.1f} GiB")
# a tiny computation on the GPU, so "available" means "works"
x = torch.randn(2048, 2048, device="cuda")
print("matmul ok:", float((x @ x).abs().mean()) > 0)
PY
echo OK
