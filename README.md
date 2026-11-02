# ECE_551_problems_ML

Two-semester project: reproducing and extending **Fast-DDPM**, a fast diffusion
model for medical image-to-image generation, on the UNM Alliance HPC cluster
(Easley). A second goal is to document the whole process as a tutorial so the
experiments can be replicated by others on the same cluster.

## Background

- Source list of models and datasets: [pattichis/AIM](https://github.com/pattichis/AIM)
- Base model: [Fast-DDPM](https://github.com/mirthAI/Fast-DDPM) (Jiang et al., JBHI 2025)
  - Trains and samples with 10 diffusion time steps instead of 1,000.
  - Three tasks: CT denoising (LDFDCT), T1w-to-T2w MRI translation (BRATS),
    multi-image MRI super-resolution (PMUB).
- Why this model: working public code, pretrained checkpoints, published
  benchmark numbers to reproduce, and the simplest architecture of the
  diffusion models surveyed in AIM (plain U-Net denoiser).

## Goals

1. Reproduce Fast-DDPM's published results on our own cluster environment.
2. Build a reusable HPC workflow (environment, SLURM scripts, checkpoint/resume).
3. Make Fast-DDPM training and inference as fast as possible on Easley while
   holding quality at the reproduced 37.46 dB PSNR / 0.9155 SSIM. This is a
   7-phase HPC roadmap (profiling, sharded inference, single-GPU efficiency, DDP,
   multi-node scaling, experiments at scale, 3D):
   [docs/ROADMAP.md](docs/ROADMAP.md).
4. Write a tutorial that lets a new lab member repeat the pipeline from a clean
   account: [docs/hpc_tutorial.md](docs/hpc_tutorial.md) (full) and
   [docs/hpc_quickstart.md](docs/hpc_quickstart.md) (one page).

## Compute environment

| Item | Value |
|---|---|
| Cluster | Easley (UNM Alliance HPC) |
| GPUs used so far | NVIDIA L40S (46 GB) |
| H100 usage | **empty** |
| Hopper (A100) | No login found yet; the account lists Hopper QOS names, so access is **unverified** |
| Python / PyTorch | 3.10.6 / 2.2.2 (cu121), conda env `fastddpm` |
| Scheduler | SLURM. GPU partitions: `h100` (4 nodes x 2 H100) and `l40s` (3 nodes x 4 L40S); `interactive` (4 h, 1 node per user), `debug`, `scavenger`. `general` is CPU-only |
| Scratch | `~/carc-scratch` (data and checkpoints live here, not in the repo) |

The paper pins PyTorch 1.12.1 on A100s. That version has no kernels for L40S or
H100, so this project uses PyTorch 2.2.2. Details and the dependency fixes this
required are in the tutorial. Scheduler limits, storage layout and measured
numbers are in [docs/easley_cluster_notes.md](docs/easley_cluster_notes.md).

## Status

| Milestone | Status |
|---|---|
| Cluster access, SSH/git auth, repo and Fast-DDPM fork added as submodule | Done |
| Conda environment working on L40S | Done |
| Inference reproduction (LDFDCT, pretrained checkpoint, full test set) | Done |
| Training smoke test (loop runs, checkpoints save) | Done: batch 8 at ~0.51 s/step; batch 16 runs out of memory on the L40S |
| Resume-from-checkpoint test | Done: continued from the saved step, loss in range |
| Batch-size limit test (L40S) | Done: limit is batch 14 in fp32; ~15.6 images/s at every batch size. H100 trains ~2.5x to 2.9x faster (44 to 46 images/s); H100 batch ceiling: **empty** |
| HPC roadmap phases 0-6 | Phase 0 partial; Phases 1-6 **empty** |
| Full training reproduction | **empty** |
| Extension (new dataset / hardware comparison / GDM-VE) | **empty** |

### Results

| Experiment | PSNR | SSIM | Paper (Table II) |
|---|---|---|---|
| LDFDCT inference, pretrained checkpoint, 3,501 test images | 37.46 | 0.9155 | 37.5 / 0.92 |
| LDFDCT trained from scratch on Easley | **empty** | **empty** | 37.5 / 0.92 |
| Other tasks (BRATS, PMUB) | **empty** | **empty** | see paper |

The inference result validates the pipeline (checkpoint loading, data staging,
sampling, evaluation) using the authors' weights. It does not yet show that our
own training reproduces the paper.

## Repository layout

```
ECE_551_problems_ML/
├── README.md
├── docs/
│   ├── ROADMAP.md              # plan, 7-phase HPC roadmap, themes checklist, open decisions
│   ├── hpc_tutorial.md         # full cluster setup, workflow, troubleshooting log
│   ├── hpc_quickstart.md       # one-page version of the tutorial
│   ├── easley_cluster_notes.md # partitions, QOS limits, storage, measured numbers
│   ├── fast_ddpm_theory.md     # theory notes on the paper
│   ├── fast_ddpm_code_notes.pdf    # 24-slide code walkthrough
│   └── code_notes_src/build.py     # generates the slides
├── slurm/
│   ├── ldfdct_smoke_test.sh        # inference run on the full LDFDCT test set
│   ├── ldfdct_train_smoke.sh       # short training run
│   ├── ldfdct_train_resume.sh      # resume test (same run plus --resume_training)
│   └── bs_limit_test.sh            # batch-size limit job array
└── fast-ddpm/              # git submodule: our fork of mirthAI/Fast-DDPM
```

Datasets, checkpoints, and run outputs are stored on scratch, not in this repo.

## Quick start

Full instructions are in [docs/hpc_tutorial.md](docs/hpc_tutorial.md). Short version:

```bash
git clone --recurse-submodules git@github.com:carrilloluis25/ECE_551_problems_ML.git
conda create -n fastddpm python=3.10.6 -y
conda activate fastddpm
# install pinned dependencies: see tutorial section 4
sbatch slurm/ldfdct_smoke_test.sh
```

## Notes

- Fast-DDPM's configs ship with batch sizes tuned for an 80 GB A100. They were
  lowered for the L40S (for example `sampling_fid.batch_size` 128 to 16).
- Adding a new dataset means editing several `if/elif` dispatch points in the
  Fast-DDPM code. The exact locations are in the tutorial.

## Open questions

- Learning rate: the paper says 2e-4, the repo configs say 2e-5. Which one: **empty**
- Extension after the HPC roadmap: **empty**
- GPU-hour allocation and multi-node limits on Easley: **empty**
- Hopper access: **empty**
- Advisor / course deliverable format: **empty**

## Acknowledgements

Fast-DDPM code and checkpoints by Jiang et al. (mirthAI). Model and dataset
index from Prof. Marios S. Pattichis's AIM repository.
