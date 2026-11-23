# Multi-GPU in Fast-DDPM: what the code does today, and what DDP adds

This page explains how the Fast-DDPM code uses more than one GPU **today**, how that differs from
the standard way (`DistributedDataParallel`, started by `torchrun`), and what we would add to the
model code to use it. It is background for Phases 1 to 4 of [ROADMAP.md](ROADMAP.md).

Each statement says where it comes from: **[code]** read in the Fast-DDPM source (the upstream
repository, which our fork follows; patch 0001 is the only change we have made so far),
**[measured]** from a job on Easley, **[PyTorch]** how PyTorch works in general (not yet observed
on this project), or **[unverified]** an expectation we still have to test.

## 1. What is implemented today

### 1.1 One GPU is the normal case
`fast_ddpm_main.py` builds one model on one device and trains it with a normal loop. A job given
one GPU runs exactly that. All the results so far (the 500,000-step baseline, Phase 0) are
single-GPU. [measured]

### 1.2 `torch.nn.DataParallel` is already in the code
Both training and inference wrap the model in `DataParallel`:

| Where | What the code does | Source |
|---|---|---|
| `Diffusion.sg_train` (training) | `model = torch.nn.DataParallel(model)` | [code] |
| `Diffusion.sg_sample` (inference) | `model = torch.nn.DataParallel(model)` before loading weights | [code] |

With one GPU the wrapper does nothing useful. With **two or more GPUs visible to the job**, it
splits every batch across them with no change to the code. This is the "free" multi-GPU mode, and
[scaling_bench.sh](../slurm/scaling_bench.sh) measures it. It has **not been run yet**, so we do not
know how much it speeds training up. [unverified]

### 1.3 What else the current loop does (all [code] unless noted)
- **Data:** `DataLoader(batch_size=..., shuffle=True, num_workers=8, pin_memory=True)`. One loader,
  one process.
- **Every step:** `loss.item()` (for the log line) and a TensorBoard write. Both wait for the GPU,
  which costs under about 2% of a step. [measured]
- **Gradient clipping is never applied.** It is inside `try/except: pass` and the config has no
  `grad_clip` key; the profile confirmed 0 clips in 75 steps. [measured]
- **EMA:** `EMAHelper` (decay 0.999) keeps a second copy of the weights. It only knows how to
  unwrap `DataParallel`.
- **Checkpoint:** a list `[model, optimizer, epoch, step, EMA]`; the model keys start with `module.`
  because of the `DataParallel` wrapper; files are 1.7 GB. [measured]
- **Seeds:** one global seed (`--seed`, default 1244) set once at the start.
- **Failures:** `main` catches exceptions and returns exit code 0, so a crash looks like a clean
  finish. We check the checkpoint's step instead. [measured]
- **Inference:** `DataLoader(shuffle=False)`, noise drawn from the global generator with
  `torch.randn`, images named `<ckpt>_<img_id>_pt.png` where `img_id` starts at the number of
  files already in the output folder. Only the averages of PSNR and SSIM are logged.

### 1.4 What patch 0001 added (our only change to the fork)
[patches/0001-training-controls.patch](../patches/0001-training-controls.patch):
- `--max_steps N`: stop after N steps and save a final checkpoint (0 = no limit).
- `--no_first_ckpt`: skip the step-1 checkpoint.
- Atomic checkpoint saves: write `ckpt_<step>.pth.tmp`, rename it, then repoint `ckpt.pth`, so a
  kill leaves a stray `.tmp` file rather than a truncated checkpoint.

None of this is multi-GPU specific. It makes fixed-length benchmark runs possible.

## 2. How `DataParallel` works (and where it is slow)

One Python process owns the model, whose weights live on GPU 0. [PyTorch]

**Every forward pass:**
1. **Scatter:** the batch is cut along its first dimension. A batch of 16 on 2 GPUs becomes 8 and 8.
2. **Replicate:** the model is copied from GPU 0 to every other GPU. This happens on every call.
3. **Parallel apply:** one Python thread per GPU runs the forward pass on its chunk.
4. **Gather:** the outputs are copied back and joined on GPU 0.

**Backward and update:** the loss is computed on GPU 0. The gradients from the replicas are summed
into the parameters on GPU 0, and `optimizer.step()` and the EMA update run there. The next forward
pass replicates the new weights again.

What follows from this: [PyTorch], to be checked against `scaling_bench.sh` output
- GPU 0 does extra work (gather, loss, optimizer, EMA, Adam state), so it uses more memory and
  the other GPUs may wait for it.
- The model (113.7M parameters, about 455 MB in fp32) is copied to each extra GPU on every forward
  pass.
- Single machine only.
- If the batch does not divide evenly, the GPUs get unequal chunks (16 on 3 GPUs gives 6, 5, 5).

## 3. How DDP works, and what `torchrun` does

`DistributedDataParallel` (DDP) uses **one process per GPU**. [PyTorch]

`torchrun` starts the N processes and gives each one environment variables:

| Variable | Meaning |
|---|---|
| `RANK` | the process's id across the whole job (0 to N-1) |
| `LOCAL_RANK` | its id on its own machine; used to choose the GPU |
| `WORLD_SIZE` | the total number of processes |
| `MASTER_ADDR`, `MASTER_PORT` | where the processes find each other at start-up |

**Each step:**
1. Each rank reads **its own** slice of the data (`DistributedSampler` makes the slices disjoint).
2. Each rank runs forward, loss and backward on its own GPU.
3. During backward, DDP **averages the gradients** across ranks with an all-reduce (NCCL, over
   NVLink or PCIe). Gradients are grouped into buckets, and each bucket's all-reduce starts as soon as
   its gradients are ready, so communication overlaps with the rest of the backward pass.
4. Every rank runs the optimizer on the same averaged gradients. Because every rank starts from the
   same weights (DDP broadcasts rank 0's when it is built), the copies stay identical. No weights
   are copied between steps.

Only the gradients cross the link, once per step: about **455 MB in fp32** (113.7M parameters, [measured]).
At an assumed PCIe speed of 50 GB/s that is about 10 ms against 226 ms of backward ([computed]; the
link type is [unverified] until `scaling_bench.sh` prints `nvidia-smi topo -m`).

## 4. The two side by side

| | `DataParallel` (in the code today) | DDP with `torchrun` (to add) |
|---|---|---|
| Processes | 1, threads inside it | 1 per GPU |
| Launch | `python fast_ddpm_main.py ...` | `torchrun --nproc_per_node=N fast_ddpm_main.py ...` |
| What moves each step | model copies out, outputs back, gradients to GPU 0 | gradients only (all-reduce) |
| Who does the update | GPU 0 | every rank, identically |
| Data loading | one loader feeds all GPUs | one loader per rank, disjoint slices |
| Multi-node | no | yes (`srun` + `torchrun`, untested here) |
| Code change | none | yes (section 5) |
| Expected scaling | below the ideal | closer to the ideal [unverified] |
| Best fixed-global-batch speed-up on 2 H100 | not measured | at most ~1.88x ([computed]: batch 8 takes 0.182 s, batch 16 takes 0.342 s) |

The 1.88x ceiling comes from the hardware, not from either method: two H100 each running half a
batch of 16 cannot be 2x faster, because a batch of 8 is not twice as fast as a batch of 16.
Reaching 2x or more needs a larger global batch, which is a different recipe.

## 5. DDP is an addition to the model code, not a rewrite

The model (the U-Net), the loss, the noise schedule, the data and the sampler stay the same. DDP
adds a thin layer around the training loop. This is the list for the Phase 3 patch (a second patch
file after 0001):

| Change | Why |
|---|---|
| `dist.init_process_group("nccl")`, `torch.cuda.set_device(LOCAL_RANK)` | start the process group |
| `DistributedDataParallel(model, device_ids=[LOCAL_RANK])` instead of `DataParallel` | the wrapper |
| `DistributedSampler` and `sampler.set_epoch(epoch)` instead of `shuffle=True` | disjoint data per rank, reshuffled each epoch |
| per-rank seed: `--seed + rank` for the noise and timestep draws | otherwise every rank draws identical noise |
| per-rank batch = global batch / world size | keep the recipe (global batch 16) unchanged |
| rank-0-only checkpoint saving, logging, TensorBoard, and folder deletion | otherwise ranks overwrite each other |
| `EMAHelper` unwraps DDP as well as `DataParallel` | otherwise EMA keys and checkpoints mismatch |
| keep strict checkpoint loading; a DDP checkpoint must load in single-GPU eval | so evaluation tools keep working |
| check the checkpoint step instead of the exit code | `main` returns 0 after a crash; a dead rank can also hang the others |
| logged loss averaged across ranks (one small all-reduce, or rank 0's loss only) | the log is comparable with single-GPU runs |

Inference is simpler: it needs no gradients, so [ROADMAP.md](ROADMAP.md) Phase 1 splits the test
set into shards (job array, or one `torchrun` launch with one shard per GPU) and merges per-image
scores afterwards. It has its own list of fixes (global image ids, per-image seeds, per-image CSV).

## 6. How we will know it works

1. **Hello test:** a tiny script, separate from Fast-DDPM, where every rank prints its rank,
   hostname and GPU and does an all-reduce with a predictable answer (rank 0 holds 1, rank 1
   holds 2, both end with 3). It shows how many nodes and GPUs the job really got.
2. **Baselines:** `scaling_bench.sh` on 1 H100 and on 2 H100 gives the single-GPU and
   `DataParallel` numbers.
3. **Same recipe, different hardware:** the DDP loss curve at 1 and 2 GPUs (same global batch)
   should match the single-GPU curve.
4. **Speed-up and efficiency:** speed-up = (1-GPU seconds per step) / (N-GPU seconds per step);
   efficiency = speed-up / N. Compare against the 1.88x bound.
5. **Quality:** a DDP-trained checkpoint scored by the single-GPU evaluation (37.457 dB / 0.9155 is the
   baseline to match within the run-to-run spread, which is not yet measured).

## 7. Limits on Easley

Per-user caps (2026-11-21): 2 H100 or 3 L40S at a time, 1 GPU on `interactive`. So single-node DDP
here means 2 H100 or up to 3 L40S, and a 4-GPU job is not possible on the normal partitions. The
`scavenger` partition lists no cap but its jobs can be preempted (untested). See
[easley_cluster_notes.md](easley_cluster_notes.md) and [phase0_findings.md](phase0_findings.md).
