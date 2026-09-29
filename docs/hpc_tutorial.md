# Tutorial: Running Fast-DDPM on the UNM Alliance HPC Cluster

This document is written so someone else in the lab, starting from a clean
account, can reproduce this project's environment and results end to end.
Fill in the `<...>` placeholders as they're confirmed (partition names,
dataset paths, etc.) — this file should be updated as the project progresses,
not written once and left stale.

## 1. Cluster resources

| Cluster | Nodes | GPUs | Interconnect | Use for |
|---|---|---|---|---|
| Hopper | 61 | 37× NVIDIA A100 (80GB) | HDR 400 Gbps InfiniBand | Baseline reproduction (matches paper's training hardware) |
| Easley | 65 | 36× L40S + 8× H100 | NDR 800 Gbps InfiniBand | Hardware-comparison phase only (needs newer PyTorch/CUDA) |

**Why Hopper first:** Fast-DDPM pins PyTorch 1.12.1, which predates official
H100 (`sm_90`) support and likely lacks compiled kernels for L40S (`sm_89`).
A100 is a direct hardware match to what the paper used, so nothing about the
software stack needs to change to get a correct baseline.

## 2. Account setup: SSH keys and Git/GitHub auth

These are the steps we worked through getting `ECE_551_problems_ML` pushed to
GitHub from the cluster login node — documented here so it isn't rediscovered
from scratch next time.

**Set your git identity** (a fresh account auto-fills name/email from your
username + hostname, which you don't want on your commits):
```bash
git config --global user.name "Your Name"
git config --global user.email "the-email-your-GitHub-account-uses@example.com"
```

**GitHub no longer accepts account passwords over HTTPS git** (disabled in
2021). Use SSH keys instead:
```bash
ls -la ~/.ssh                          # check if a key already exists
ssh-keygen -t ed25519 -C "your_github_email@example.com"   # if not
cat ~/.ssh/id_ed25519.pub              # copy this output
```
Paste the copied key into GitHub: Settings → SSH and GPG keys → New SSH key.

**Load the key and verify:**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh -T git@github.com
# Expect: "Hi <username>! You've successfully authenticated, but GitHub does not provide shell access."
```

**Clone/push using the SSH remote**, not HTTPS:
```bash
git remote set-url origin git@github.com:<username>/ECE_551_problems_ML.git
git push -u origin main
```

If `ssh -T git@github.com` still fails after adding a key, check which key is
actually being offered:
```bash
ssh -vT git@github.com   # look for "Offering public key: ..." — confirm it matches the one on GitHub
```

## 3. Environment setup

> **TODO once run:** record the exact commands used (conda vs. Apptainer
> container), and note whether PyTorch 1.12.1 was used as-is or needed a
> CUDA-toolkit-matched rebuild for Hopper's driver version.

Target versions (from the Fast-DDPM paper, Section IV/Training Details):
- Python 3.10.6
- PyTorch 1.12.1
- torchvision (matching), OpenCV, NumPy, scikit-image, MedPy — via the
  repo's `requirements.txt`

Check GPU driver / CUDA compatibility before installing:
```bash
nvidia-smi   # run inside an interactive GPU job (see §4), not on the login node
```

## 4. SLURM basics for this cluster

> **TODO:** fill in actual partition/QOS names once confirmed
> (`sinfo` on the cluster will list them).

Interactive GPU session (for debugging, not long training runs):
```bash
srun --partition=<gpu_partition> --gres=gpu:1 --time=01:00:00 --pty bash
```

Batch job template (fill in the placeholders):
```bash
#!/bin/bash
#SBATCH --job-name=fastddpm-smoke
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

module load <cuda_module> <python_module>   # fill in from `module avail`
source activate fastddpm                    # or your conda env name

python fast_ddpm_main.py --config <task>.yml --dataset <TASK> --exp <exp_path> --doc <run_name> --sample --fid
```

**Storage note:** point `--exp` and any checkpoint output at scratch/work
storage, not `$HOME` — home directories on HPC systems typically have small
quotas.

## 5. Workflow: from clean checkout to reproduced results

1. **Inference smoke test** — pretrained checkpoint (Hugging Face
   `SebastianJiang/FastDDPM`) + one provided dataset, `--sample --fid` only,
   single GPU, short job. Confirms environment + data paths are correct
   before spending training compute.
2. **Training smoke test** — same dataset, a handful of iterations only.
   Confirms checkpoints save to the right place and that resuming from a
   checkpoint after a killed job works (needed since training runs
   (~26h reported in the paper) likely exceed a single job's walltime limit).
3. **Full reproduction run** — train to completion on the chosen task,
   evaluate PSNR/SSIM against the paper's Table I numbers.

## 6. Extending to a new dataset

The repo dispatches on `--dataset {LDFDCT|BRATS|PMUB}`, each backed by a
loader class in `datasets/`. To add a new one:

1. Preprocess the raw data into the same convention as the existing loaders
   expect (check how an existing loader reads e.g. `PMUB-test` as the
   template — same normalization, slicing, directory naming).
2. Write a new `Dataset` class mirroring an existing one's interface, and
   register it wherever `--dataset` strings are dispatched to a class.
3. Clone an existing YAML config (e.g. `LDCT.yml`) and adjust paths/image
   size/channels for the new dataset.
4. Run the smoke-test pattern (§5.1–5.2) on the new dataset before committing
   a full job to it.

## 7. Sweeping configurations (SLURM job arrays)

Each config variant (timestep count, `scheduler_type uniform|non-uniform`,
learning rate, resolution) = one array task, one GPU:

```bash
#!/bin/bash
#SBATCH --job-name=fastddpm-sweep
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-<N-1>
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

CONFIGS=(config_a.yml config_b.yml config_c.yml)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

python fast_ddpm_main.py --config $CONFIG --exp runs/sweep_${SLURM_ARRAY_TASK_ID} --doc sweep_${SLURM_ARRAY_TASK_ID}
```

Give every array task a unique `--exp`/`--doc` so outputs never collide.
Hopper's 37 A100s give real parallel capacity for this — but GPU count, not
node count, is the actual concurrency ceiling, and it's shared with other
cluster users.

## 8. Hardware comparison (Easley) — Semester 2

Once the Hopper baseline is solid: port the environment to Easley, upgrading
PyTorch/CUDA to a version with H100 (`sm_90`) and L40S (`sm_89`) kernel
support. Run the identical config across A100 / H100 / L40S and compare
training throughput and time-to-convergence.

## Troubleshooting log

> Append entries here as issues come up — this is the part that actually
> saves the next person time.

- **`error: remote origin already exists`** — use `git remote set-url origin
  <url>` instead of `git remote add`.
- **`git push` 403 with correct-looking password** — GitHub disabled password
  auth for git in 2021; use a Personal Access Token or SSH key (§2).
- **`Permission denied (publickey)` on clone/push** — no key loaded, or key
  never added to GitHub account. Run `ssh -vT git@github.com` and check the
  "Offering public key" line matches what's registered on GitHub.
- **`git init` run from `$HOME` instead of a project directory** — check
  `git status` before adding files broadly; relocate with `git mv` + `mv
  .git` into a proper project subdirectory rather than re-initializing.
