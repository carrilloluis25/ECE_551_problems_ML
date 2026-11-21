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
| Memory per image | ~3.2 GB (+2.7 GB fixed) | ~3.1 GB |
| Inference (10 steps, batch 16) | ~3.6 s per batch; 3,501 images in 16.6 min | not measured |
| GPUs per node | 4 | 2 |
| **Per-user cap** | **3** (`l40s`), **1** (`interactive`) | **2** (`h100`) |

(all **measured**, except "memory per image", which is **computed** from the batch-size tables)

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
6. **What the training loop does** (read from `sg_train`, **not yet profiled**):
   - it already wraps the model in `DataParallel`, so a job given 2 GPUs uses both with no code change;
   - `loss.item()` (in the log line) and `tb_logger.add_scalar` force a GPU sync every step;
   - gradient clipping is inside `try/except: pass` and the config has no `grad_clip` key, so it is
     probably skipped (the profile checks this);
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

## 4. Still open in Phase 0

`slurm/phase0_profile.sh` (not yet run) measures, in one H100 job of a few minutes:

| Item | Why it matters |
|---|---|
| Parameter count (total and per block) | confirms the ~116M figure derived from the checkpoint size, and sizes the gradient all-reduce |
| True peak memory (`max_memory_allocated`) | replaces the `nvidia-smi` readings, which overstate and are not monotonic |
| Time per phase of a step (data wait, host-to-device, forward, backward, clip, optimizer, EMA, logging) | decides which speedups are worth building |
| Data-loader speed alone | whether 8 workers can feed 1, 2 or 3 GPUs |
| `torch.profiler` top operations | where the GPU and the Python overhead go |
| GPU utilization | whether the GPU is waiting |
| Allocator retries at batch 16 and 33 | tests the explanation for the slow batch 33 |

When those results exist, section 1 gets a "where the time goes" row, and Phase 0 is done.
