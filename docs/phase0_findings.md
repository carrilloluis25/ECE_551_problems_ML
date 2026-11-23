# Phase 0 findings: the baseline and what it tells us

Phase 0 of [ROADMAP.md](ROADMAP.md) asks for a baseline table (time, throughput, memory, utilization)
and a profile of where a training step's time goes, without changing any model code. This page
collects what was measured on Easley (2026-11), what it implies for the later phases, what was
wrong along the way, and what is still open. Every number says whether it was **measured**,
**computed** (arithmetic on measured numbers) or **unverified**. The sources are
[easley_cluster_notes.md](easley_cluster_notes.md) (the tables) and
[run_options_and_bounds.md](run_options_and_bounds.md) (limits and the baseline run).

## 1. The baseline table

| | L40S (46 GB) | H100 NVL (96 GB) |
|---|---|---|
| Usable memory (PyTorch) | 44.4 GiB | 93.1 GiB |
| **fp32 training batch ceiling** | **14** (16 fails at step 2) | **33** (34 fails at step 2) |
| Training speed at batch 8 | 0.507 s/step, 15.7 images/s | 0.182 s/step, 44.0 images/s |
| Training speed at batch 16 (the paper's) | does not fit | 0.342 s/step, 46.8 images/s |
| Throughput vs batch size | flat at ~15.6 images/s from batch 4 | flat at ~47 images/s from batch ~12 |
| Memory per image | ~3.2 GB (+2.7 GB fixed), from `nvidia-smi` | ~3.1 GB from `nvidia-smi` (includes allocator cache); **2.6 GiB steady state** from the profiler (section 4) |
| Inference (10 steps, batch 16) | ~3.6 s per batch; 3,501 images in 16.6 min | not measured |
| **Where a step goes** (profile, batch 16) | not profiled | backward 65%, forward 30%, EMA 3%, optimizer 1.5%, data wait 0% |
| Parameters | 113,669,761 (same model) | 113,669,761 |
| GPUs per node | 4 | 2 |
| **Per-user cap** | **3** (`l40s`), **1** (`interactive`) | **2** (`h100`) |

(all **measured**, except "memory per image", which is **computed** from the batch-size tables; the profile rows are from section 4)

The from-scratch baseline (batch 16, lr 2e-5, 500,000 steps, one H100): **37.457 dB / 0.9155 on the full
test set** (the authors' checkpoint: 37.457 / 0.9155; paper 37.46 / 0.9155). It took about **48.6
H100-hours** at 0.345 to 0.359 s/step (**measured**, from `sacct`). Checkpoints are 1.7 GB.

## 2. Findings that shape the next phases

1. **Throughput is flat in batch size on both cards**, so a bigger batch is not a speed lever. It
   also means that splitting a batch across GPUs has a floor: on the H100, batch 8 per card runs at
   0.182 s against 0.342 s for batch 16, so two H100 at a fixed global batch of 16 can give at most
   about **1.88x** before any communication cost (**computed**). Reaching 2x needs a larger global batch,
   which is a different recipe.
2. **The per-user GPU caps limit what can be studied:** 2 H100 or 3 L40S at a time, and 1 GPU on
   `interactive`. A 1/2/4/8-GPU scaling study cannot run on the normal partitions. A 2-node job with
   one GPU per node is within the stated limits (**untested**).
3. **Memory:** the H100 ceiling (33) is about 2.4x the L40S ceiling (14). At 33 the card is 98.9% full
   and the step is about 1.7x slower than extrapolated (1.18 s vs ~0.68 s). The likely cause is the
   allocator working at the limit (**unverified**; the profile reports `num_alloc_retries`).
4. **A large share of the smoke test may be checkpoint writing.** The smoke config saves every 100
   steps, and each save stalled the L40S for ~35 s (two 1.7 GB files, measured before the atomic-save
   patch). 100 steps of compute is ~51 s, so about **40% of the wall time** may be saving
   (**computed**). Saving every 100 steps is also pointless for a smoke test. The patch now writes one
   file plus a link, which should roughly halve the stall (**unverified**).
5. **The "10-minute smoke test" is a time limit, not a workload.** `ldfd_train_smoke.yml` sets no
   step cap, so the job trains until `--time=00:10:00` ends it. Speed comparisons need a fixed number
   of steps (`--max_steps`), as `scaling_bench.sh` does.
6. **What the training loop does** (read from `sg_train`; the profile confirmed the first two points):
   - it already wraps the model in `DataParallel`, so a job given 2 GPUs uses both with no code change;
   - `loss.item()` (in the log line) and `tb_logger.add_scalar` force a GPU sync every step, which
     costs under about 2% of a step (GPU kernel time is ~0.331 s of a 0.338 s step; **measured**);
   - **gradient clipping is never applied**: it ran 0 times and was silently skipped on all 75 measured
     steps, because the config has no `grad_clip` key. The baseline config was derived from this file,
     so the model that matched the paper trained without clipping (**measured**);
   - every process would get the same seed under DDP, so the ranks would draw identical noise and
     timesteps: each rank needs its own seed;
   - `main` catches exceptions and returns exit code 0, so a crash looks like a clean finish.
7. **Quality versus steps:** 250,000 steps already gives 37.23 dB (0.24 dB below the final
   37.46); the subset score passed the pass bar at 225,000 (**measured**, stride-10 subset).
8. **Operations:** scratch has a **100 GiB group quota** (hit on 2026-11-10); `--export=ALL,VAR=...`
   holds jobs on the H100 partition (set variables in the shell instead); easley052 was down on
   2026-11-08 and ran jobs again on 2026-11-21.

## 3. What we got wrong, and when we found out

| Claim | What was true | How we found out |
|---|---|---|
| The H100 has 80 GB | 95,830 MiB (H100 NVL) | `hello_torch.sh` |
| 400,000 steps for the baseline | the authors' checkpoint is at 500,000 | reading its checkpoint |
| Held H100 jobs mean a bad node | the cause was `--export=ALL,VAR=...` | an A/B test |
| A 25 dB cutoff at 50,000 steps | the run was healthy and rising 3 to 4 dB per 25k | the run itself |
| The curve would flatten near 37.3 dB | it reached 37.46 | the finished run |
| The H100 batch ceiling was above 32 | exactly 33 | the 2026-11-21 array |
| A job's exit code shows success | the fork's `main` returns 0 after a crash | the quota failure |
| 4 L40S in one job is possible | the cap is 3 (1 on `interactive`) | `sacctmgr` |

## 4. Profile results (2026-11-22, H100, batch 16, job 1445699)

`slurm/phase0_profile.sh`, one H100 (**measured**; the 5-step `torch.profiler` pass adds its own overhead).

| Measurement | Result |
|---|---|
| Parameters | **113,669,761** (up path 62.5%, down 27.5%, mid 9.7%); gradients 455 MB in fp32 |
| Checkpoint size implied by 4 fp32 copies | 1.694 GiB (the files are 1.736 GiB) |
| Natural step | **0.3376 s = 47.4 images/s** (the baseline run: 0.345 s with logging and snapshots) |
| Time waiting for data | 0.1 ms per step (0.0%) |
| Data loader alone (8 workers) | 155 images/s, about 3.3x what one H100 consumes |
| GPU utilization (`nvidia-smi`) | median 99% (coarse: a kernel was running, not how efficiently) |
| Allocator retries at batch 16 | 0 |
| Gradient clipping | skipped on every step (no `grad_clip` in the config) |

Where a step goes (a synchronize between phases, so the sum is 0.3464 s against the natural 0.3376 s):

| Phase | ms | % |
|---|---:|---:|
| backward | 225.8 | 65.2% |
| forward + loss | 103.8 | 30.0% |
| EMA update | 10.3 | 3.0% |
| optimizer step | 5.2 | 1.5% |
| host-to-device copy | 1.0 | 0.3% |
| data wait, noise and timestep sampling, logging | ~0.2 | ~0.0% |

GPU time by operation (5 profiled steps): convolution backward 40.5%, forward convolutions ~14%
(the kernels are TF32 `sm90` kernels, so convolutions already use the tensor cores), elementwise
`mul` 8.7% and `add_` 7.9%, **`nchwToNhwc` layout conversions 8.8%**, GroupNorm backward 7.0%.

What this means:
1. **A step is 95% forward and backward compute**; data, optimizer, EMA and logging are about 5%
   together. One H100 is saturated, so a single-GPU speedup must reduce GPU work (bf16,
   `channels_last`, `torch.compile`), and more GPUs divide it.
2. **`channels_last` has a measured target:** the 8.8% spent converting layouts for every convolution.
   That is a ceiling, not a promise.
3. **The elementwise operations (about 16%) are not the EMA** (the EMA is 3% of the step). They are
   mostly the model's own operations, a target for `torch.compile` fusion.
4. **Multi-GPU should scale well on compute:** the gradient all-reduce is 455 MB per step. At an
   assumed PCIe bandwidth of about 50 GB/s that is roughly 10 ms against 226 ms of backward, and DDP
   overlaps it with the backward pass. The 1.88x bound for 2 H100 at a global batch of 16 (section 2)
   stands; the link speed is an assumption until `scaling_bench.sh` prints the topology.
5. **The data loader is not a limit** for one GPU, and can feed about 3 GPUs' worth of images per
   second from 8 workers. Under DDP each process has its own loader (8 workers each).

**Node variation: negligible.** Three batch-16 profile runs on different nodes gave 0.3376, 0.3388 and
0.3399 s/step (a 0.7% spread), with the same phase split (backward 65.2 to 65.3%, forward 30.0%) and 99%
GPU utilization (**measured**).

**Memory (batch 16, measured with the updated profiler, 2026-11-22):**

| | MiB | GiB |
|---|---:|---:|
| Peak allocated during the first 15 steps | 59,661 | 58.3 |
| **Peak allocated in steady state (after warm-up)** | **45,129** | **44.1** |
| Peak reserved by the allocator since warm-up | 51,836 | 50.6 |
| Allocated at the end of a step (weights + Adam + EMA + one batch) | 2,283 | 2.2 |

The first steps peak **14.5 GiB above steady state**, which fits `cudnn.benchmark = True` trying
workspace-hungry algorithms and then releasing them. `nvidia-smi` agrees once this is accounted for: its
usual 52,755 MiB is the allocator's reserved 51,836 MiB plus about 0.9 GiB of CUDA context, and the odd
57,991 MiB reading in one job was the short-lived peak caught by the 1-second sampler. The steady-state
memory per image is about **2.6 GiB** ((45,129 - 2,283) / 16 MiB, **computed** and assuming linear
scaling), lower than the ~3.1 GB per image from `nvidia-smi`, which includes the allocator's cache.
The weights, Adam state and EMA take only 1.7 GiB of the 44.1 GiB; the rest is activations and
temporary workspaces.

**Batch 33 on the same profiler (job 1445727).** It is the same model at the card's edge:

| | Batch 16 | Batch 33 | If it scaled linearly |
|---|---:|---:|---:|
| Natural step | 0.339 s | **1.064 s** (31.0 images/s) | ~0.70 s |
| Forward + loss | 104 ms | 222 ms | 214 ms |
| Backward | 226 ms | **837 ms** | 466 ms |
| Allocator retries | 0 | **7** | |
| Peak allocated / reserved | 58.3 / 60.4 GiB | 89.4 / 91.8 GiB | |

The slowdown is entirely in the backward pass (1.8x slower than linear scaling would give), while the
forward pass scales normally (1.04x). A likely cause (**unverified**): at the memory limit cuDNN can only
use convolution-backward algorithms with small temporary workspaces, which are slower. Retries alone (7
over the whole run) cannot explain a slowdown at every step. Treat batch 32 as the practical maximum:
batch 33 runs, but at 31 images/s it is slower per image than batch 16 (47 images/s).

## 5. Still open in Phase 0

- Optional: test the cuDNN-algorithm explanation for the slow backward pass at batch 33 (for example with
  `torch.backends.cudnn.benchmark = False`, or by logging the chosen algorithms). It does not affect the plan.
- Done: parameter count, step-time split, loader speed, utilization, gradient clipping, node variation,
  steady-state memory, and the batch-33 behaviour. **Phase 0 is complete.**
