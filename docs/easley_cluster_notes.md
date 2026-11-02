# Easley Cluster Notes

Reference facts about the UNM Alliance HPC cluster as it applies to this project,
with the command that produced each one. Recorded 2026-11-02 unless noted.
Statuses: **confirmed** (seen in command output or a log), **computed** (derived
from confirmed numbers), **unverified** (inferred or not yet checked).

---

## 1. Hardware

From the cluster's published resource table (screenshot supplied by the user):

| Cluster | Nodes | CPU cores | GPUs | Interconnect | Notes |
|---|---|---|---|---|---|
| Easley | 65 | 4,160 | 36x L40S + 8x H100 | NVIDIA NDR 800 Gbps InfiniBand | Newest; 23.3 TB RAM total |
| Hopper | 61 | 2,176 | 37x A100 | NVIDIA HDR 400 Gbps InfiniBand | See section 6 |

Observed on Easley (`scontrol show node <node> | grep -i gres`, `nvidia-smi`):
- `easley058`: 4x L40S, 64 CPUs, ~257 GB RAM (**confirmed**). Not every node has
  GPUs: `easley001` shows no GPU resource (**confirmed**).
- GPU resource names: `gpu:l40s` (**confirmed**). `gpu:h100`: **unverified**,
  check before use.
- L40S: `nvidia-smi` reports 46,068 MiB; PyTorch reports 44.40 GiB usable
  (**confirmed**). Driver 570.211.01, CUDA 12.8.
- To my knowledge L40S cards have no NVLink (**unverified**). Check with
  `nvidia-smi topo -m` inside a multi-GPU allocation.

**H100 access (reported by the user, 2026-11-02):** permission was recently granted
for the H100 partition. Partition name, gres name (`gpu:h100` is assumed),
per-user limits, GPUs per node, and the H100 variant (PCIe vs SXM, which decides
NVLink) are all **unverified**. Commands to fill these in:
```bash
sinfo -o "%P %a %l %D %G %N"
sinfo -N -o "%N %P %G" | grep -i h100
sacctmgr show assoc user=$USER format=user,account,partition%20,qos%40
```
Then, inside an allocation on the H100 partition: `nvidia-smi`,
`nvidia-smi topo -m`, and
`python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`
(expect capability (9, 0); PyTorch 2.2.2 + cu121 should support it).
Each H100 has 80 GB, the same as the paper's A100s, so the paper's batch size of
16 may fit (**unverified**; the batch-size test will show it).

## 2. Partitions (`sinfo`)

| Partition | Time limit | Use |
|---|---|---|
| `general` (default) | 2 days | real runs |
| `interactive` | 4 hours | tests and debugging |
| `debug` | 1 hour | quick checks |
| `scavenger` | 2 days | lower priority, can be preempted |

All are backed by the same `easley[...]` nodes (**confirmed**).
`scontrol show partition general` shows: `MaxNodes=UNLIMITED`, `MaxTime=2-00:00:00`,
default QOS `easley-general`, allowed QOS `easley-general, extended, easley`.

## 3. QOS limits (`sacctmgr show qos ... format=...`)

| QOS | Per-user limits | Consequence |
|---|---|---|
| `easley-general` | `cpu=512`, `mem=2,059,472M` (~2 TB); no GPU or node cap listed | No explicit GPU limit. At 8 CPUs per GPU task, 512 CPUs allows ~64 GPU tasks (**computed**), more than the 36 L40S that exist, so GPU availability and queue priority are the real limit. Multi-node jobs are allowed. |
| `easley` | 512 submitted jobs | Large job arrays are fine. |
| `extended` | `cpu=1`, 1 job, 5-day walltime | One CPU: not usable for GPU training. |
| `interactive` | 1 node per user (`QOSMaxNodePerUserLimit` shown by `squeue`) | A job array on `interactive` runs one task at a time, whatever `%N` says. |

Caveats: `sacctmgr` shows only explicit limits. Fair-share priority can still
delay large jobs, and `GrpTRES` was blank for this account, so no group GPU cap
was visible. A GPU-hour allocation could still be tracked elsewhere
(**unverified**; ask the staff).

## 4. Storage

| Path | What it is | Notes |
|---|---|---|
| `/users/<netid>` | Home (`chama:/home/homes`, 133 TB total, 80 TB free at the time) | Per-user quota **unverified** (`quota -s` was not checked) |
| `/projects` | Shared project space (`chama:/home/carc_projects`) | Has `.snapshot` backups |
| `/carc/scratch` | Scratch (1.2 PB, 524 TB free at the time) | **Purge policy unverified** |
| `~/carc-scratch` | Symlink to `/carc/scratch/users/<netid>` | Created by the admins; set up in 2024 |

Layout used by this project, all under `~/carc-scratch`:
- `fastddpm-data/data/LD_FD_CT_{train,test}/`: extracted LDFDCT data (4.9 GB train,
  1.3 GB test); the zip is 7.3 GB.
- `fastddpm-exp/logs/<doc>/`: checkpoints (1.7 GB each) and `stdout.txt` logs.

Checkpoints are large and numerous; set `snapshot_freq` high and clean up test
runs.

## 5. Measured performance (single L40S, PyTorch 2.2.2)

| Item | Result | Status |
|---|---|---|
| Inference, 10 steps, batch 16 | ~3.6 s per batch; 3,501 images in ~16 min | confirmed |
| Inference quality, LDFDCT test set | PSNR 37.46, SSIM 0.9155 (paper: 37.5 / 0.92) | confirmed |
| Inference batch 128 | CUDA out of memory | confirmed |
| Training batch 16 | CUDA out of memory at step 2 (step 1 fits, then Adam + EMA state is allocated) | confirmed |
| Training batch 8 | ~0.507 s/step (~2 steps/s) | confirmed |
| Checkpoint save | ~1.7 GB per file; ~35 s stall per snapshot (two files written) | confirmed |
| Startup | ~10 s fresh; ~50 s when resuming (loads a 1.7 GB checkpoint) | confirmed |
| Resume | Continued from the saved step; loss stayed in range | confirmed |
| 400,000 steps at batch 8 | ~56 h at 0.507 s/step | computed, assumes speed holds |
| 800,000 steps at batch 8 (matches the paper's total images seen) | ~113 h | computed |
| Batch-size limit test (4, 8, 10, 12, 14) | **empty** (`slurm/bs_limit_test.sh` written; results pending) | |

Loss behavior (interpretation, consistent with the logged values): the raw loss
is a sum over pixels, and a batch containing an image at time step 0 adds about
65,536 / batch_size to the mean, because the network cannot predict 1% noise.
This produces clusters near 8k, 16k and 24k at batch 8. Log per-time-step or
smoothed loss for graphs.

## 6. Hopper

The account's SLURM associations (`sacctmgr show assoc user=$USER`) list QOS
names `hopper-condo`, `hopper-debug` and `hopper-genera+` (truncated, probably
`hopper-general`), plus other CARC clusters (`wheeler`, `xena`). This suggests the
shared accounting database associates the account with Hopper (**unverified**).
It does not prove login or job-submission access. Earlier attempts to find a
Hopper login node found none.

To test:
```bash
sacctmgr show cluster format=cluster%12,controlhost,controlport
sinfo -M all 2>&1 | head
```
Or ask the staff whether the account has Hopper access and how to log in.
If it does, Hopper's A100s (80 GB) would fit the paper's batch size of 16.

## 7. Commands used to learn the above

| Question | Command |
|---|---|
| Partitions and node states | `sinfo` |
| GPU type on a node | `scontrol show node <node> \| grep -i gres` |
| Partition limits and QOS | `scontrol show partition general` |
| QOS limits | `sacctmgr show qos easley-general,extended,easley format=name%16,maxtrespu%40,maxtrespj%30,maxwall,maxjobspu,maxsubmitpu` |
| Account associations | `sacctmgr show assoc user=$USER format=user,account,qos%40,maxtrespu%30,grptres%30` |
| Why a job is pending | `squeue -u $USER` (the `NODELIST(REASON)` column) |
| Storage mounts | `df -h \| grep -i carc`, `readlink -f ~/carc-scratch` |
| GPU driver / CUDA | `nvidia-smi` inside a GPU allocation |

## 8. Questions for the cluster staff

- Do I have access to Hopper, and how do I log in?
- Per-user GPU limits and any GPU-hour allocation for my account?
- Can I run multi-node GPU jobs on `easley-general`?
- Scratch purge policy and per-user quota?
- Gres name for the H100 cards?
