# HPC Quickstart: Fast-DDPM on Easley

One-page version of [hpc_tutorial.md](hpc_tutorial.md). Use this to get running;
use the full tutorial for the reasons behind each step and the troubleshooting log.

`<...>` marks a value to replace. Do not paste it literally into the shell.

## 0. What you need
- Easley login (UNM Alliance HPC), a GitHub account, an SSH key added to GitHub.
- GPU used: L40S (`--gres=gpu:l40s:1`). Partition for tests: `interactive` (4 hr cap).
  Partitions for long runs: `l40s` or `h100` (2 days). `general` has no GPUs.

## 1. Get the code (once)
```bash
git clone --recurse-submodules git@github.com:<you>/ECE_551_problems_ML.git
cd ECE_551_problems_ML
```
`fast-ddpm/` is a git submodule (a fork of mirthAI/Fast-DDPM). If it is empty:
`git submodule update --init --recursive`.

## 2. Build the environment (once, on the login node, no GPU needed)
```bash
conda create -n fastddpm python=3.10.6 -y
conda activate fastddpm
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm tensorboard tensorboardX scikit-image medpy pillow scipy pyyaml pandas lmdb matplotlib requests gdown
pip install "numpy<2" "opencv-python<4.10"
pip check
```
Why these pins: torch 2.2.2 needs NumPy 1.x, and recent opencv needs NumPy 2.
The paper's PyTorch 1.12.1 has no L40S/H100 support.

## 3. Stage data and checkpoint (once, on scratch)
Scratch: `~/carc-scratch` (symlink to `/carc/scratch/users/<netid>`). Never put data in the repo.
```bash
mkdir -p ~/carc-scratch/fastddpm-data ~/carc-scratch/fastddpm-exp/logs/ldfdct_smoke_test

# dataset bundle (7.3 GB) from the Fast-DDPM README's Google Drive link
gdown 1kF0g8fMR5XPQ2FTbutfTQ-hwG_mTqerx -O ~/carc-scratch/fastddpm-data/processed_dataset.zip
cd ~/carc-scratch/fastddpm-data
unzip processed_dataset.zip "data/LD_FD_CT_test/*" "data/LD_FD_CT_train/*" -d .

# pretrained denoising checkpoint (1.7 GB), renamed to what the sampler expects
curl -L -C - -o ~/carc-scratch/fastddpm-exp/logs/ldfdct_smoke_test/ckpt_1.pth \
  https://huggingface.co/SebastianJiang/FastDDPM/resolve/main/ckpt_LDFDCT.pth
```

## 4. Make a config
```bash
cd ~/ECE_551_problems_ML/fast-ddpm
cp configs/ldfd_linear.yml configs/ldfd_smoke_test.yml
```
Edit `configs/ldfd_smoke_test.yml`:
```yaml
data:
    train_dataroot: "/users/<netid>/carc-scratch/fastddpm-data/data/LD_FD_CT_train"
    sample_dataroot: "/users/<netid>/carc-scratch/fastddpm-data/data/LD_FD_CT_test"
sampling:
    ckpt_id: [1]          # we have one checkpoint, saved as ckpt_1.pth
sampling_fid:
    batch_size: 16        # paper uses 128 (80 GB A100); 128 runs out of memory on L40S
```

## 5. Run inference (verified working)
Save as `slurm/ldfdct_smoke_test.sh`, then `sbatch` it:
```bash
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
python fast_ddpm_main.py --config ldfd_smoke_test.yml --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp --doc ldfdct_smoke_test \
  --sample --fid --scheduler_type uniform --timesteps 10
```
```bash
sbatch slurm/ldfdct_smoke_test.sh
squeue -u $USER
tail -f fastddpm-ldfdct-smoke-<jobid>.err     # logs go to .err, not .out
```
Expect about 16 minutes. The last log line is `Average: PSNR 37.46, SSIM 0.9155`
(paper: 37.5 / 0.92).

## 6. Run training (not yet verified)
Same as step 5 with: a config copy that sets `training.snapshot_freq: 10`, no
`--sample --fid`, and a **different `--doc`** (e.g. `ldfdct_train_smoke`).
Reusing the inference `--doc` would delete the checkpoint folder.
To resume after a stop, add `--resume_training` with the same `--exp` and `--doc`.

## 7. SLURM cheat sheet
| Task | Command |
|---|---|
| Submit | `sbatch script.sh` |
| My jobs | `squeue -u $USER` |
| Cancel | `scancel <jobid>` |
| Job history | `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed` |
| Partitions | `sinfo` |
| GPU names | `scontrol show node <node> \| grep -i gres` |
| Interactive GPU | `srun --partition=interactive --gres=gpu:l40s:1 --time=00:30:00 --pty bash` |

Package installs do not need a GPU: do them on the login node.
`nvidia-smi` only works inside a GPU allocation.

## 8. Most common failures
| Symptom | Cause and fix |
|---|---|
| `Permission denied (publickey)` | Key not on GitHub, or not loaded: `ssh -vT git@github.com` |
| `bash: gpu_partition: No such file` | You pasted a `<placeholder>` literally |
| `ModuleNotFoundError: yaml/pandas/lmdb/...` | Missing dependency; rerun the step 2 pip lines |
| NumPy "compiled using NumPy 1.x" warning | `pip install "numpy<2"` |
| `CUDA out of memory` | Lower `batch_size` in the config |
| Job killed at `TIME LIMIT` | Raise `--time`, or use the `l40s` / `h100` partition for runs over 4 hr |
| Job fails instantly, `.out` empty | Read the `.err` file; Python errors land there |

## 9. Rules of thumb
- Test on a tiny run before a long one; queue time makes debugging at scale slow.
- Cancel jobs you no longer need (`scancel`); GPUs are shared.
- Keep data, checkpoints, and run output on scratch; keep only code and docs in git.
- Record every new error and its fix in the full tutorial's troubleshooting log.
