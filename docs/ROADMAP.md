# Project Roadmap: Fast Diffusion Models for Medical Image Translation on HPC

**Repo:** ECE_551_problems_ML
**Source models surveyed:** [pattichis/AIM](https://github.com/pattichis/AIM)
**Primary target model:** [Fast-DDPM](https://github.com/mirthAI/Fast-DDPM) (Jiang et al., JBHI 2025)
**Cluster:** UNM Alliance HPC — Hopper (37× A100 80GB) and Easley (36× L40S + 8× H100)
**Duration:** 2 semesters

## Goal

Reproduce Fast-DDPM's published results for medical image-to-image diffusion on
the lab's HPC cluster, establish a reusable HPC workflow (environment, job
scripts, checkpointing), then extend the work along two axes: new
datasets/configurations, and a hardware comparison (A100 vs. H100 vs. L40S).
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
  budget to plan SLURM jobs against, and matches Hopper's A100s directly.
- Reports PSNR/SSIM against ground truth for 3 tasks (Table I in the paper) —
  gives us an objective reproduction target instead of eyeballing outputs.
- Simplest architecture among the diffusion models surveyed in AIM (plain
  U-Net denoiser + DDPM/DDIM sampling) — no latent space, no cascaded
  GAN+diffusion stages, no video/temporal component.

## Known technical constraint

Fast-DDPM pins Python 3.10.6 / PyTorch 1.12.1, which predates official
support for H100 (`sm_90`) and likely lacks compiled kernels for L40S
(`sm_89`). **Hopper (A100) is the baseline target**; Easley (H100/L40S)
requires upgrading the PyTorch/CUDA stack first and is deliberately deferred
to the hardware-comparison phase, not attempted during initial reproduction.

---

## Semester 1 — Reproduce and build HPC fluency

| Weeks | Goal | Deliverable / exit criteria |
|---|---|---|
| 1–2 | Read DDPM fundamentals + Fast-DDPM paper closely; understand why 10 steps suffice (their step-selection scheme) | Written summary of the method (for advisor / own notes) |
| 3–4 | Get cluster fluency: SSH, git+GitHub auth (done), module system, SLURM basics (`sbatch`/`srun`, interactive jobs, GPU requests) on Hopper | First successful GPU job on Hopper (`nvidia-smi` inside a batch job) |
| 5–6 | Stand up the environment (conda or Apptainer container) matching Fast-DDPM's pinned versions; get repo + one pretrained checkpoint + one dataset onto the cluster | **Inference smoke test**: generate a sample image from a pretrained checkpoint, single GPU, no training |
| 7–9 | Training smoke test: train for a handful of iterations, confirm checkpoints land on scratch storage (not home), confirm resume-from-checkpoint works after a killed job | Working checkpoint/resume cycle |
| 10–13 | Full reproduction run on one task (pick one: multi-image super-resolution, low-dose CT denoising, or T1w→T2w translation) with our own PSNR/SSIM eval script | Numbers within reasonable range of Table I in the paper |
| 14–16 | Write up Semester 1: reproduction results, environment/job scripts finalized, gaps encountered | Semester 1 report/poster + first draft of the cluster tutorial |

## Semester 2 — Extend

Pick based on how Semester 1 lands; suggested order:

1. **Scaling / hardware study (primary recommendation).** Port the environment
   to Easley (upgrade PyTorch/CUDA to support H100/L40S), then benchmark
   training throughput and time-to-convergence for the identical
   config across A100 / H100 / L40S. Independent of any ML novelty — a
   legitimate systems contribution on its own, and reuses everything built in
   Semester 1.
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
- [ ] SLURM partition/QOS names on Hopper and Easley (fill in once known —
      see tutorial placeholders).
- [ ] Semester 2 direction (scaling study vs. new dataset vs. GDM-VE
      comparison) — revisit at the Semester 1 writeup checkpoint.
