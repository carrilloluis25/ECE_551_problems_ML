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
| 7–9 | Training smoke test: train for a handful of iterations, confirm checkpoints land on scratch storage (not home), confirm resume-from-checkpoint works after a killed job | Working checkpoint/resume cycle | |
| 10–13 | Full reproduction run on one task (pick one: multi-image super-resolution, low-dose CT denoising, or T1w→T2w translation) with our own PSNR/SSIM eval script | Numbers within reasonable range of Table I in the paper (see pivot caveat above) | |
| 14–16 | Write up Semester 1: reproduction results, environment/job scripts finalized, gaps encountered | Semester 1 report/poster + first draft of the cluster tutorial | |

## Semester 2 — Extend

Pick based on how Semester 1 lands; suggested order:

1. **Scaling / hardware study (primary recommendation).** Narrowed from the
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

## Open decisions to revisit

- [ ] Which of the three Fast-DDPM tasks to reproduce first (super-resolution
      / denoising / translation) — pick based on which public dataset is
      fastest to obtain access to.
- [x] ~~SLURM partition/QOS names~~ — confirmed: `general` (2-day, real runs),
      `interactive` (4hr, debugging), `debug` (1hr), `scavenger` (2-day,
      preemptible). GPU gres: `gpu:l40s` confirmed, `gpu:h100` assumed but
      not yet verified.
- [ ] Verify the H100 gres resource name before the Semester 2 hardware study
      (`scontrol show node <h100-node> | grep -i gres`).
- [ ] Semester 2 direction (L40S/H100 scaling study vs. new dataset vs.
      GDM-VE comparison) — revisit at the Semester 1 writeup checkpoint.
