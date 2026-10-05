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
3. Extend the work. Direction for the extension: **empty**
   (candidates in [docs/ROADMAP.md](docs/ROADMAP.md)).
4. Write a tutorial that lets a new lab member repeat the pipeline from a clean
   account: [docs/hpc_tutorial.md](docs/hpc_tutorial.md).

## Compute environment

| Item | Value |
|---|---|
| Cluster | Easley (UNM Alliance HPC) |
| GPUs used so far | NVIDIA L40S (46 GB) |
| H100 usage | **empty** |
| Hopper (A100) | Not reachable from this account |
| Python / PyTorch | 3.10.6 / 2.2.2 (cu121), conda env `fastddpm` |
| Scheduler | SLURM (`interactive`, `general`, `debug`, `scavenger` partitions) |
| Scratch | `~/carc-scratch` (data and checkpoints live here, not in the repo) |

The paper pins PyTorch 1.12.1 on A100s. That version has no kernels for L40S or
H100, so this project uses PyTorch 2.2.2. Details and the dependency fixes this
required are in the tutorial.

## Status

| Milestone | Status |
|---|---|
| Cluster access, SSH/git auth, repo and Fast-DDPM fork added as submodule | Done |
| Conda environment working on L40S | Done |
| Inference reproduction (LDFDCT, pretrained checkpoint, full test set) | Done |
| Training smoke test (checkpoint saving) | **empty** |
| Resume-from-checkpoint test | **empty** |
| Full training reproduction | **empty** |
| Extension (new dataset / sweep / hardware comparison / GDM-VE) | **empty** |

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
│   ├── ROADMAP.md          # two-semester plan and open decisions
│   └── hpc_tutorial.md     # cluster setup, workflow, troubleshooting log
├── slurm/
│   ├── ldfdct_smoke_test.sh        # inference run on the full LDFDCT test set
│   └── ldfdct_train_smoke.sh       # short training run for the training smoke test
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

- Which task to use for the full training reproduction: **empty**
- Semester 2 direction: **empty**
- Compute budget / allocation limits on Easley: **empty**
- Advisor / course deliverable format: **empty**

## Acknowledgements

Fast-DDPM code and checkpoints by Jiang et al. (mirthAI). Model and dataset
index from Prof. Marios S. Pattichis's AIM repository.
