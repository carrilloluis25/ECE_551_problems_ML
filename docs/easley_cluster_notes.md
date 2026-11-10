# Easley Cluster Notes

Reference facts about the UNM Alliance HPC cluster as it applies to this project,
with the command that produced each one. Recorded 2026-11-02 unless noted.
Statuses: **confirmed** (seen in command output or a log), **computed** (derived
from confirmed numbers), **unverified** (inferred or not yet checked).
For choosing how to run something and the limits that apply, see
[run_options_and_bounds.md](run_options_and_bounds.md); this file holds the raw facts.

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
- GPU resource names: `gpu:l40s` and `gpu:h100` (**confirmed**: both are used in
  `#SBATCH --gres` lines of jobs that ran; `nvidia-smi` reports "NVIDIA H100 NVL").
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
Memory per H100 is **95,830 MiB total (93.1 GiB usable to PyTorch)**, not 80 GB (`nvidia-smi` and
`torch.cuda.get_device_properties` in `hello_gpu.sh` / `hello_torch.sh`, 2026-11-09). A training run read 94,133 MiB in use
(see the batch-size test). The paper's batch size of 16 fits (confirmed).

## 2. Partitions (`sinfo`)

| Partition | Time limit | Use |
|---|---|---|
| `general` (default) | 2 days | **CPU only**: nodes easley001-048 have no GPUs |
| `h100` | 2 days | 4 nodes (easley051-054), 2x H100 each = **8 H100** |
| `l40s` | 2 days | 3 nodes (easley055-057), 4x L40S each = **12 L40S** |
| `interactive` | 4 hours | CPU nodes plus the H100 and the 3 L40S nodes; tests and debugging |
| `debug` | 1 hour | same nodes; quick checks |
| `scavenger` | 2 days | preemptible; also includes L40S nodes easley058-063, so all 9 L40S nodes (36 cards) are reachable only here |

(`sinfo -o "%P %a %l %D %G %N"`, **confirmed**.) Two consequences: **GPU jobs go to
`h100`, `l40s` or `interactive`, never `general`**, and only 12 L40S plus 8 H100
sit in non-preemptible partitions. An H100 node holds only 2 GPUs, so a 4-GPU H100
run needs 2 nodes; an L40S node holds 4.
`scontrol show partition general` shows: `MaxNodes=UNLIMITED`, `MaxTime=2-00:00:00`,
default QOS `easley-general`, allowed QOS `easley-general, extended, easley`.
The QOS and limits of the `h100` and `l40s` partitions have **not** been checked:
`scontrol show partition h100` and `scontrol show partition l40s`.

## 3. QOS limits (`sacctmgr show qos ... format=...`)

| QOS | Per-user limits | Consequence |
|---|---|---|
| `easley-general` | `cpu=512`, `mem=2,059,472M` (~2 TB); no GPU or node cap listed | Applies to the CPU-only `general` partition, so it says **nothing** about GPU limits. The GPU partitions' QOS is unverified. |
| `easley` | 512 submitted jobs | Large job arrays are fine. |
| `extended` | `cpu=1`, 1 job, 5-day walltime | One CPU: not usable for GPU training. |
| `interactive` | 1 node per user (`QOSMaxNodePerUserLimit` shown by `squeue`) | A job array on `interactive` runs one task at a time, whatever `%N` says. |

Caveats: `sacctmgr` shows only explicit limits. Fair-share priority can still
delay large jobs, and `GrpTRES` was blank for this account, so no group GPU cap
was visible. A GPU-hour allocation could still be tracked elsewhere
(**unverified**; ask the staff).

**Added 2026-11-03** (`sacctmgr show qos format=Name,MaxWall,MaxTRESPU,MaxJobsPU,GrpTRES`,
`sacctmgr show assoc user=$USER format=Account,Partition,QOS,GrpTRES,GrpTRESMins,MaxTRES,MaxTRESMins,MaxWall,MaxJobs`):
- The account's associations show **no** `GrpTRESMins`, `MaxTRESMins`, `MaxWall` or `MaxJobs`
  (**confirmed**), so no GPU-hour cap is set in SLURM.
- Per-user QOS CPU caps on the Easley QOS rows: `easley-h1+` 64, `easley-l4+` 96,
  `easley-in+` 64, `easley-de+` 128, `easley-ge+` 512. The names are truncated, so the
  mapping to h100 / l40s / interactive / debug / general is **assumed**.
- The **GPU part** (`gres/...`) of the per-user limits was cut off by column width, so the
  per-user GPU cap on `h100` and `l40s` is still **unverified**. Rerun:
  `sacctmgr show qos format=Name%22,MaxWall,MaxTRESPU%70,MaxJobsPU,GrpTRES%40 | grep -iE "name|easley"`.
- Other QOS rows seen: `large-alloc` (3 days, `cpu=1024`) and `extended` (5 days, `cpu=1`, 1 job).
- `sshare -u $USER` is the correct form (`-u $lc2025` fails with an empty variable).

## 4. Storage

| Path | What it is | Notes |
|---|---|---|
| `/users/<netid>` | Home (`chama:/home/homes`, 133 TB total, 80 TB free at the time) | Quota (`quota -s`, 2026-11-03): **100 GB soft, 200 GB hard**, about 4.3 M files; 36 GB and 191k files used (**confirmed**) |
| `/projects` | Shared project space (`chama:/home/carc_projects`) | Has `.snapshot` backups |
| `/carc/scratch` | Scratch (1.2 PB, 509 TB free on 2026-11-03) | **Group quota: 100 GiB per group (`lc2025`), confirmed 2026-11-10** by the login banner (`exceeded (101.23GiB/100.00GiB)`) and a refused write (`Disk quota exceeded`); it counts everything the group owns on scratch, so keep `du -sh ~/carc-scratch/` (trailing slash: it is a symlink) under it. Freed space was credited back a few minutes after deleting. **Purge policy still unverified** (`lfs` is not installed) |
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
| Batch-size limit test (4, 8, 10, 12, 14) | Limit is batch 14 in fp32; ~15.6 images/s at every size (table below) | confirmed |

Loss behavior (interpretation, consistent with the logged values): the raw loss
is a sum over pixels, and a batch containing an image at time step 0 adds about
65,536 / batch_size to the mean, because the network cannot predict 1% noise.
This produces clusters near 8k, 16k and 24k at batch 8. Log per-time-step or
smoothed loss for graphs.

### Batch-size limit test (L40S, fp32, `slurm/bs_limit_test.sh`, 2026-11)

Training `batch_size` set per task, 4-minute runs, memory from `nvidia-smi`
sampled every 2 s (this is memory held by the process including PyTorch's cache,
so it slightly overstates what is strictly needed).

| Batch | s/step | images/s | Peak GPU memory (MiB) |
|---:|---:|---:|---:|
| 4 | 0.260 | 15.4 | 15,425 |
| 8 | 0.509 | 15.7 | 28,131 |
| 10 | 0.641 | 15.6 | 34,489 |
| 12 | 0.766 | 15.7 | 40,819 |
| 14 | 0.894 | 15.7 | 43,225 |
| 16 | out of memory at step 2 (earlier run) | | |

Findings:
- **Limit on one L40S in fp32: batch 14.** The paper's batch of 16 does not fit.
- **Throughput is flat at ~15.6 images/s at every batch size.** The GPU is already
  compute-saturated at batch 4, so a larger batch buys no speed. Speedups must
  come from compute efficiency (bf16, `channels_last`, `torch.compile`) or from
  more GPUs.
- **Step time is linear in batch size**, about 0.063 s per image with almost no
  fixed overhead (**computed** from the table).
- **Memory is linear too:** about 3.2 GB per image plus about 2.7 GB fixed
  (**computed**, from batches 4 to 12; the 14 point is capped by the card). The
  fixed part is consistent with weights + gradients + Adam + EMA at ~20 bytes per
  parameter for roughly 100M parameters (**interpretation**; Phase 0's parameter
  count will confirm).
- **Extrapolation (computed, untested):** batch 16 needs about 53 GB, which
  explains the out-of-memory error. On the H100 the same slope held from batch 14 to 28 (see the H100 section below), and the paper's batch 16 fits there.
- **Cost estimate (computed):** the paper's 400,000 steps at batch 16 take about
  112 hours on one L40S (400,000 x 16 x 0.063 s).

### Batch-size test on H100 (easley053, default PyTorch settings)

Sizes 4 to 14 ran first with the original (untagged) script, which overwrote the L40S
result files (the L40S table above is the saved original; files were later renamed
`bs_result_<GPU>_bs<N>.txt`). Sizes 16 to 32 ran with the GPU-tagged script (array
1398837). Two tasks could share the node's 2 H100s.

| Batch | s/step | images/s | Peak (MiB, nvidia-smi) | Throughput vs L40S |
|---:|---:|---:|---:|---:|
| 4 | 0.105 | 38.1 | 15,933 | 2.5x |
| 8 | 0.182 | 44.0 | 46,493 | 2.8x |
| 10 | 0.223 | 44.8 | 49,657 | 2.9x |
| 12 | 0.261 | 46.0 | 41,629 | 2.9x |
| 14 | 0.303 | 46.2 | 46,709 | 2.9x |
| 16 | 0.342 | 46.8 | 52,743 | (L40S out of memory) |
| 20 | 0.426 | 46.9 | 65,269 | |
| 24 | 0.500 | 48.0 | 77,697 | |
| 28 | 0.585 | 47.9 | 89,917 | |
| 32 | 0.658 | 48.6 | 94,133 | |

Findings:
- **No out-of-memory up to batch 32.** The H100's ceiling is above 32 and still
  unmeasured. The paper's batch of 16 fits with room to spare (about 53 GB).
- **The cards have more than 80 GB.** `nvidia-smi` reported 94,133 MiB in use at batch 32.
  **Confirmed 2026-11-09:** the cards are **H100 NVL** with 95,830 MiB total (93.1 GiB usable
  to PyTorch), measured by `slurm/hello_gpu.sh` and `slurm/hello_torch.sh`. An earlier note
  here assuming 80 GB was wrong.
- **The H100 trains about 2.5x to 2.9x faster than the L40S** at the same batch size
  (images/s, **computed**).
- **Throughput saturates around batch 12 to 16** at about 47 images/s. Going from batch 16 to
  32 adds only about 4%, so a larger batch is not a speed lever, only a memory/semantics choice.
- **Memory is linear in batch size from 14 up:** about 3.1 GB per image
  (52.7 GB at 16 to 89.9 GB at 28), matching the L40S slope (**computed**). Small batches
  (8, 10) read high, likely from PyTorch's caching and cuDNN benchmark mode
  (**interpretation, unverified**), and batch 32 sits at the card's limit. The `nvidia-smi`
  column is only a rough guide; `torch.cuda.max_memory_allocated()` is the reliable number.
- **Cost estimate (computed):** the paper's 400,000 steps at batch 16 take about
  400,000 x 0.342 s = 38 hours on one H100, plus roughly 1 hour of checkpoint writes if
  saved every 5,000 steps. That fits inside one 2-day `h100` job. It assumes the speed holds
  and the job is not interrupted (against about 112 hours on one L40S).
  **Update 2026-11-03:** the authors' released checkpoint is at **500,000** steps, and the
  baseline run targets that: 500,000 x 0.342 s = **47.5 hours**, which leaves almost no
  margin in a 48-hour limit, so it runs as four chained 125,000-step jobs. The 10,000-step
  learning-rate runs measured 29 min per 5,000 steps (0.348 s/step), matching 0.342.

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
- What are the per-user limits on the `h100` and `l40s` partitions, and can I run multi-node jobs there?
- Scratch purge policy? Can the 100 GiB group quota (hit on 2026-11-10) be raised for a 500,000-step run?
- Gres name for the H100 cards? (`gpu:h100` worked in job scripts; `nvidia-smi` reports "NVIDIA H100 NVL".)
- Why are H100 jobs submitted with `--export=ALL,VAR=...` held with "user env retrieval failed
  requeued held"? (Tested 2026-11-08 on `easley051`: the same job without `--export` ran; a
  variable set in the shell instead of `--export` also ran. We work around it.)
- Is there a GPU-hour allocation tracked outside SLURM (a CARC portal)?
