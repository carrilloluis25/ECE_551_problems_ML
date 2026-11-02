# Project Roadmap: Fast Diffusion Models for Medical Image Translation on HPC

**Repo:** ECE_551_problems_ML
**Source models surveyed:** [pattichis/AIM](https://github.com/pattichis/AIM)
**Primary target model:** [Fast-DDPM](https://github.com/mirthAI/Fast-DDPM) (Jiang et al., JBHI 2025)
**Cluster:** UNM Alliance HPC — Easley (36× L40S + 8× H100). Hopper (37× A100
80GB) was the original target but turned out not to be reachable from this
account (no separate Hopper login found) — see "Known technical constraint"
below for the pivot this caused.
**Duration:** 2 semesters

## Goal

Reproduce Fast-DDPM's published results for medical image-to-image diffusion on
the lab's HPC cluster, establish a reusable HPC workflow (environment, job
scripts, checkpointing), then extend the work along two axes: new
datasets/configurations, and a hardware comparison (L40S vs. H100).
Document the process as a tutorial so the workflow is reproducible by others
in the lab.

**Chosen direction (added Nov 2, 2026):** make Fast-DDPM training and inference
as fast as possible on Easley, while holding quality at the reproduced
**37.46 dB PSNR / 0.9155 SSIM**. This is organized as a 7-phase HPC roadmap
(profiling, sharded inference, single-GPU efficiency, DDP, multi-node scaling,
experiments at scale, 3D). See "Direction: Fast-DDPM HPC Roadmap" below. The
source document is `Fast-DDPM HPC Roadmap.pdf`.

## Why Fast-DDPM

- Public, working code with a clear structure (`configs/`, `datasets/`,
  `models/`, `runners/`).
- Pretrained checkpoints available (Hugging Face `SebastianJiang/FastDDPM`) —
  lets us validate the pipeline via inference before spending training
  compute.
- Paper reports exact training hardware (4× A100 80GB node, batch size 16,
  400K iterations, ~26h for the super-resolution task) — gives a concrete
  budget to plan SLURM jobs against. (Originally this was going to be a
  direct hardware match via Hopper's A100s; since we're on Easley's L40S/H100
  instead, treat this as a rough time budget rather than an exact match —
  see "Known technical constraint" below.)
- Reports PSNR/SSIM against ground truth for 3 tasks (Table I in the paper) —
  gives us an objective reproduction target instead of eyeballing outputs.
- Simplest architecture among the diffusion models surveyed in AIM (plain
  U-Net denoiser + DDPM/DDIM sampling) — no latent space, no cascaded
  GAN+diffusion stages, no video/temporal component.

## Known technical constraint (and the pivot it caused)

Fast-DDPM pins Python 3.10.6 / PyTorch 1.12.1, which predates official
support for H100 (`sm_90`) and lacks compiled kernels for L40S (`sm_89`).
The original plan was to baseline on Hopper's A100s (a direct hardware match
to the paper) and defer any PyTorch/CUDA upgrade to a later
hardware-comparison phase.

**That plan changed once we confirmed Hopper isn't reachable from this
account** — only Easley is. Resolution: build the environment on Easley from
the start using a modern PyTorch build (**2.2.2**, cu121) that supports both
L40S and H100. Confirmed working as of the environment-setup session (see
`docs/hpc_tutorial.md` §4 for the full recipe and the dependency conflicts
hit along the way — NumPy 2.x/torch ABI mismatch, opencv-python needing
NumPy ≥2 while torch needs <2).

**Consequence:** results won't be a bit-for-bit reproduction of the paper's
Table I (different torch version, different GPU architecture) — should land
close, but state this explicitly in any writeup rather than claiming exact
reproduction.

---

## Semester 1 — Reproduce and build HPC fluency

| Weeks | Goal | Deliverable / exit criteria | Status |
|---|---|---|---|
| 1–2 | Read DDPM fundamentals + Fast-DDPM paper closely; understand why 10 steps suffice (their step-selection scheme) | Written summary of the method (for advisor / own notes) | |
| 3–4 | Get cluster fluency: SSH, git+GitHub auth, SLURM basics (`sbatch`/`srun`, interactive jobs, GPU requests) on Easley | First successful GPU job on Easley (`nvidia-smi` inside an `srun` allocation) | ✅ Done |
| 5–6 | Stand up the environment (conda) — pivoted to modern PyTorch 2.2.2 since Hopper's unreachable; get repo (fork + submodule) + one pretrained checkpoint + one dataset onto the cluster | **Inference smoke test**: generate a sample image from a pretrained checkpoint, single GPU, no training | ✅ Done — LDFDCT, full 3,501-image test set: **PSNR 37.46 / SSIM 0.9155** vs. paper's Table II `37.5 / 0.92`. Confirms pipeline correctness (checkpoint loading, data staging, sampling math, eval code) using the authors' weights — does **not** yet confirm our own training reproduces this (next milestone). |
| 7–9 | Training smoke test: train for a handful of iterations, confirm checkpoints land on scratch storage (not home), confirm resume-from-checkpoint works after a killed job | Working checkpoint/resume cycle | ✅ Done. Training loop runs (loss ~72,000 to a few hundred); batch 16 runs out of memory on the L40S, batch 8 works at ~0.51 s/step; checkpoints are ~1.7 GB with a ~35 s stall per save; resume continued from the saved step with the loss in range. |
| 10–13 | Full reproduction run on one task (pick one: multi-image super-resolution, low-dose CT denoising, or T1w→T2w translation) with our own PSNR/SSIM eval script | Numbers within reasonable range of Table I in the paper (see pivot caveat above) | Now the gate of Phase 5 in the HPC roadmap below: a from-scratch run reaching ~37.5 dB, done on the parallel pipeline. |
| 14–16 | Write up Semester 1: reproduction results, environment/job scripts finalized, gaps encountered | Semester 1 report/poster + first draft of the cluster tutorial | |

## Direction: Fast-DDPM HPC Roadmap

**Goal:** cut Fast-DDPM's training and inference time on Easley as far as
possible while keeping denoising quality at the reproduced 37.46 dB PSNR /
0.9155 SSIM. Each phase has a gate that must be met before the next phase
relies on it. Phases 1 and 2 can run side by side after Phase 0. Phase 6 needs
only Phase 3, so the 3D work can start before the scaling study finishes.

| Phase | What | Gate | Status |
|---|---|---|---|
| 0 | **Baseline and profiling.** Inference baseline; short training runs at several batch sizes (seconds/step, images/s, `max_memory_allocated`); parameter count; `torch.profiler` split of data loading, forward, backward, optimizer, EMA; GPU utilization; L40S cards per node and partition limits. Changes no code. | Baseline table (time, throughput, memory, utilization) plus a profile showing where the time goes | 🔶 Partial. Known: inference ~3.6 s per batch of 16 (~16 min for 3,501 images); training ~0.51 s/step at batch 8; batch 16 runs out of memory in fp32; one node had 4 L40S. L40S batch-limit test done: ceiling is batch 14 (fp32), ~15.6 images/s flat from batch 4 to 14 (compute-bound), ~3.2 GB per image + ~2.7 GB fixed. H100 (batch 4 to 14): ~2.5x to 2.9x the L40S throughput, ~44 to 46 images/s; its batch ceiling and true memory use are unmeasured. Empty: profiler, utilization, parameter count |
| 1 | **Sharded inference.** Add `--shard` / `--num_shards` (keep indices where `index % num_shards == shard`); per-image PSNR/SSIM to a CSV per shard; Slurm job array, one L40S each; merge from all 3,501 rows, not from per-shard averages; repeat at `--timesteps` 5 and 2 | Merged PSNR/SSIM equal to the single-GPU run (37.46 / 0.9155), wall-clock near 1/N | Empty |
| 2 | **Single-GPU efficiency.** bf16 autocast, `channels_last`, `torch.compile`, data-loader tuning (`pin_memory`, `persistent_workers`, `prefetch_factor`, pre-converted data), then re-measure the largest batch that fits | Steps/s table for each optimization added one at a time, with no quality loss | Empty |
| 3 | **DDP training on one node.** Replace `DataParallel` in `sg_train` with `DistributedDataParallel` via `torchrun`; `DistributedSampler` with `set_epoch`; per-rank seeds; rank-0-only checkpointing, logging and folder deletion; keep strict checkpoint loading; decide batch semantics (16 global vs 16 per GPU) after settling the learning rate | Matching loss curves at 1, 2 and 4 GPUs (same global batch) and a measured speedup over the Phase 2 single-GPU run | Empty |
| 4 | **Multi-node training and scaling study.** `srun` + `torchrun` across nodes; NCCL all-reduce benchmark first; strong scaling (fixed global batch, 1, 2, 4, 8 GPUs) and weak scaling (fixed per-GPU batch); scaling efficiency = speedup / GPUs; profile compute vs. communication; project the paper's 400,000 iterations against its 26 h on 4 A100s | Strong- and weak-scaling plots with efficiency per GPU count and a stated point where more GPUs stop paying off | Empty |
| 5 | **Experiments at scale.** Learning rate 2e-4 vs 2e-5; step-count ablation (3, 5, 10, 50 steps) as a job array, evaluated with sharded inference; uniform vs non-uniform scheduler (Table V, denoising row); iteration cap; small validation pass | A from-scratch run reaching ~37.5 dB, plus the ablation tables, all produced on the parallel pipeline | Empty |
| 6 | **3D and the memory wall.** `Conv2d` to `Conv3d`; find the largest crop and batch on one GPU; activation checkpointing; FSDP; spatial parallelism (stretch); latent-space diffusion (the paper's alternative) | For each technique, the largest volume trainable and its time per step | Empty |

### Metrics recorded in every phase
Same columns for every run so results compare across phases:
time per training step (s, wall clock over 100+ steps after warm-up);
throughput (images/s, global batch / step time); peak GPU memory
(`torch.cuda.max_memory_allocated()`); GPU utilization (%); communication share
(%, from the profiler); scaling efficiency (speedup / GPUs); PSNR / SSIM (sharded
inference on the 3,501 test images); GPU-hours (GPUs x wall-clock hours).

### Additions after reviewing the roadmap against the code
These come from reading the Fast-DDPM source and our measurements, not from the
original roadmap document.
- **EMA and DDP.** `EMAHelper` in `models/ema.py` only unwraps `nn.DataParallel`.
  Under DDP the EMA weights would be stored under different names and
  checkpoints would stop loading. Unwrap both wrappers (Phase 3).
- **Move the iteration cap earlier.** `n_iters` is never checked, and every
  benchmark phase needs a clean stop after N steps. Add a `--max_steps` stop in
  Phase 0 (we have been using `timeout`). Also add a flag to skip the
  always-saved step-1 checkpoint (it costs ~35 s).
- **Atomic checkpoint saves** (write to a temp file, then rename) before long
  multi-GPU runs. `torch.save` overwrites in place, and we observed `ckpt.pth`
  lagging the newest numbered checkpoint after a kill.
- **Batch 16 and 32 in Phase 0** do not fit in fp32 on one L40S. Record the
  out-of-memory result, then re-measure after bf16 in Phase 2.
- **`--timesteps 5` and `2`** (Phase 1) are checked to be subsets of the trained
  levels (0, 99, 199, ..., 999): they sample {0, 199, 399, 599, 799, 999} and
  {0, 499, 999}. Other counts, such as 20, are not subsets and would not be a
  fair test of the pretrained checkpoint.
- **Interconnect (Phase 4).** To my knowledge L40S cards have no NVLink, so
  intra-node gradient traffic goes over PCIe: check with `nvidia-smi topo -m`.
  The model has roughly 100M parameters (inferred from checkpoint size), so each
  all-reduce moves about 400 MB. Communication should be small next to a
  ~0.5 s step but becomes visible as steps get faster after Phase 2.
- **GPU topology (confirmed by `sinfo`).** An H100 node has only 2 GPUs, so
  single-node DDP on H100 tops out at 2 GPUs and a 4-GPU H100 run needs 2 nodes.
  An L40S node has 4 GPUs, so single-node DDP can use 4 L40S. Phase 3 on H100 is
  therefore 2-GPU; Phase 4 starts at 4 GPUs there.
- **3D data (Phase 6).** The processed LDFDCT folders are named like
  `C227_216_fd.png` (case, slice). Consecutive slices per patient could be
  stacked into pseudo-volumes without new downloads. Not yet checked: whether
  every slice of each case is present.

### Open questions and risks
- **Learning rate:** paper 2e-4 vs config 2e-5. Blocks any from-scratch
  comparison; settle it early.
- **Easley capacity:** non-preemptible GPUs are 12 L40S (3 nodes x 4) and 8 H100
  (4 nodes x 2); all 36 L40S are reachable only through preemptible `scavenger`.
  Still unknown: how many GPUs one user can hold at once and whether multi-node
  jobs are allowed on the GPU partitions. These decide how far Phase 4 can go.
- **Queue time:** multi-GPU and multi-node jobs may wait much longer than
  single-GPU ones. Keep smoke tests on `interactive` and budget queue time.
- **Batch size vs quality:** a larger global batch speeds training but can change
  results; any change from the paper's 16 needs its own quality check.
- **Reproducibility:** bf16 and `torch.compile` change numerics slightly. Fix
  seeds and compare against the fp32 baseline, not against exact numbers.
- **Storage:** sharded outputs, checkpoints from many runs and pre-converted data
  all land on scratch. Check the quota before Phase 5.

## Additional extension options (after the HPC roadmap)

Pick based on how the phases above land; suggested order:

1. **Scaling / hardware study.** Largely covered by Phase 4. What remains as an
   add-on is the H100 comparison. Narrowed from the
   original A100/H100/L40S 3-way to **L40S vs. H100** (both on Easley) since
   Hopper isn't reachable. The Semester 1 environment already supports both
   architectures — this is mostly a matter of re-running the same config with
   `--gres=gpu:h100:1` and comparing throughput/time-to-convergence.
   Independent of any ML novelty — a legitimate systems contribution on its
   own, and reuses everything built in Semester 1.
2. **New dataset.** Apply Fast-DDPM to a modality/dataset not in the original
   paper, e.g. CUBS carotid ultrasound
   ([source](https://data.mendeley.com/datasets/m7ndn58sv6/1)). Requires
   writing a new `Dataset` class + config (see tutorial's "Adding a new
   dataset" section).
3. **Comparative-methods study.** Bring in
   [GDM-VE](https://github.com/mirthAI/GDM-VE) (latent-space geodesic
   diffusion) on the same dataset/task and compare quality-vs-compute against
   Fast-DDPM's pixel-space approach — same problem domain, different core
   mechanism, so it's an apples-to-apples comparison rather than a fresh
   start.

Final deliverable: consolidated report covering reproduction + chosen
extension, plus a finished cluster tutorial (see `docs/hpc_tutorial.md`) that
would let a new lab member repeat the whole pipeline from a clean account.

## Project themes checklist

Themes the project should cover to round it out. Status: done / partial / empty
(empty = not determined yet, fill in as the semester progresses).

### HPC usage
- [ ] **Permanent vs. temporary data storage** — partial. Known: datasets/checkpoints
      live on `~/carc-scratch` (`/carc/scratch`), home is `/users`, a separate
      `/projects` mount exists. Scratch purge policy, quotas, and what counts as
      permanent: empty.
- [ ] **Allocation** — empty (GPU-hour budget, per-user limits, queue priority).
- [ ] **SLURM** — partial. Done: `srun`, `sbatch`, partitions, GPU requests. Empty:
      job arrays, dependency-chained resubmission for checkpoint/resume.
- [ ] **Exploit HPC / change parameters for optimization (needs HPC)** — empty.
      Plan: array-job sweeps over `--timesteps`, `--scheduler_type`, batch size,
      learning rate, model width.
- [ ] **Find the limits** — partial. Found: L40S OOMs at `sampling_fid.batch_size: 128`,
      works at 16; training batch 16 runs out of memory, batch 8 works. L40S batch-limit
      test done (`slurm/bs_limit_test.sh`): ceiling is batch 14 in fp32, throughput
      ~15.6 images/s at every size. Empty: H100 ceiling, max image size, 3D limits
      (Phase 0 and Phase 6).
- [ ] **Demonstrate setup for experiment** — partial (tutorial, SLURM scripts, env recipe).

### Training and models
- [ ] **Something that trains quickly** — partial. Fast-DDPM's 10-step design is the
      basis; measured ~0.51 s/step at batch 8 on one L40S. Speedups are Phases 2
      to 4 of the HPC roadmap.
- [ ] **Training vs. inference** — partial. Inference measured (~3.6 s per batch of
      16, full test set in ~16 min). Training: ~0.51 s/step at batch 8, a ~35 s stall per
      1.7 GB checkpoint save; memory profile: empty (Phase 0).
- [ ] **Distillation processes** — empty. (Interpreted as step-count distillation;
      the Fast-DDPM paper discusses progressive distillation as the alternative it
      avoids. Confirm the intended meaning.)
- [ ] **Lightweight models and lightweight training** — empty (smaller `ch`/`ch_mult`,
      fewer iterations, mixed precision).
- [ ] **Expose the common pieces (saving and loading models)** — partial. Known:
      checkpoint is a list `[model, optimizer, epoch, step, EMA]`, sampling expects
      `<exp>/logs/<doc>/ckpt_<N>.pth`, resume reads `ckpt.pth`. Write-up: empty.
- [ ] **What it takes to go from 2D to 3D** — empty. Write-up of memory and compute
      scaling, what changes in the model and data loaders.
- [ ] **Validation and training loss graphs** — empty. Training loss is logged to
      TensorBoard by the existing code; validation loss appears not to be wired up
      (`validation_freq` is effectively disabled), to be confirmed.

### Deliverables
- [ ] **Write the tutorial and balance it with innovation** — partial (tutorial in
      progress; innovation direction empty).
- [ ] **Share the struggles in the tutorial** — partial (troubleshooting log exists
      and grows with each issue).
- [ ] **Inference mode in Google Colab / interface with Colab** — empty. Idea: a
      notebook that loads a pretrained checkpoint and runs sampling.
- [ ] **Image datasets on students' own computers** — empty.
- [ ] **Implement an interface** — empty (form of the interface not defined).

### Unclear items (need clarification before they can be planned)
- [ ] "xtern concept" — empty
- [ ] "get equations from MIT book" — empty (which book/chapter?)
- [ ] "show depth" — empty (depth of the math, depth of the experiments, or 3D depth?)

## Open decisions to revisit

- [ ] Which of the three Fast-DDPM tasks to reproduce first (super-resolution
      / denoising / translation) — pick based on which public dataset is
      fastest to obtain access to.
- [x] ~~SLURM partition/QOS names~~ — confirmed: `h100` (4 nodes x 2 H100) and
      `l40s` (3 nodes x 4 L40S), both 2-day; `interactive` (4hr); `debug` (1hr);
      `scavenger` (2-day, preemptible). `general` is CPU-only. Gres names
      `gpu:l40s:4` and `gpu:h100:2` per node confirmed by `sinfo`.
- [ ] H100 access was granted (2026-11-02, user-reported). Fill in: partition
      name, gres name, per-user limits, GPUs per node, PCIe vs SXM (NVLink). The
      L40S vs H100 comparison is now runnable; the batch-limit script takes a
      partition/gres override and tags results by GPU (see `easley_cluster_notes.md`).
- [ ] Extension after the HPC roadmap (H100 comparison vs. new dataset vs.
      GDM-VE comparison) — revisit once Phase 4 results are in.
- [ ] Learning rate: 2e-4 (paper) or 2e-5 (config)? Needed before Phase 3's
      batch-semantics decision and Phase 5's from-scratch run.
- [ ] GPU allocation. Known: `interactive` allows 1 node per user; the
      `easley-general` QOS (512 CPUs per user) applies only to the CPU-only
      `general` partition. Empty: the QOS and limits of the `h100` and `l40s`
      partitions (`scontrol show partition h100`), any GPU-hour allocation, and
      whether multi-node GPU jobs are allowed. Ask the staff.
- [ ] Hopper access. The account lists `hopper-condo`, `hopper-debug` and
      `hopper-genera+` QOS entries, so it may be associated with Hopper (80 GB
      A100s, matching the paper). Unverified: test with `sacctmgr show cluster` and
      `sinfo -M all`, or ask the staff. If available, it changes the batch-size
      and baseline plans.
- [ ] Reference notes: see `docs/easley_cluster_notes.md` for partitions, QOS
      limits, storage layout, measured numbers and the commands behind each.
- [ ] Scratch purge policy and quota (checkpoints are ~1.7 GB each).
