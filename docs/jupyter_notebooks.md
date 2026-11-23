# Notebooks: Colab and Easley JupyterHub

How to look at what Fast-DDPM produces, in a notebook, on either Google Colab or Easley's JupyterHub. It
includes the one-time **kernel** setup (so a notebook runs in the `fastddpm` conda environment), the
JupyterHub server profiles, and the problems we ran into. Status tags: **[ran]** seen working in this
project, **[tested on CPU]** the logic was run on a CPU with a tiny model only, **[unverified]** not run yet.

## 1. The notebooks

| Notebook | What it does | GPU needed | Status |
|---|---|---|---|
| [`notebooks/fast_ddpm_viewer.ipynb`](../notebooks/fast_ddpm_viewer.ipynb) | Runs the pretrained model (the authors' LDFDCT checkpoint, EMA weights, 10-step DDIM) on a few low-dose CT slices and shows **input \| generated \| ground truth \| error**, with PSNR and SSIM per image | 1 | **[ran]** on Colab (T4) and on the Easley hub (1 L40S) |
| [`notebooks/fast_ddpm_parallel_inference.ipynb`](../notebooks/fast_ddpm_parallel_inference.ipynb) | Extends it to several GPUs: throughput by batch size, time and quality at 10, 5 and 2 steps, a batch split across GPUs by hand, test-set sharding with a merge check, and a chart of the cluster's `scale_result_*.txt` files | 1 or more (multi-GPU sections need 2+) | **[tested on CPU]**; **[unverified]** on a GPU |

What the viewer does **not** do: it handles a handful of images (8 by default), not the 3,501-image test
set; it measures no speed; it only knows this one task (single-channel low-dose to full-dose CT
denoising, 256 by 256 grayscale); and it never trains or changes a checkpoint.

**Which images.** Any `NAME_ld.png` (low dose) with an optional `NAME_fd.png` (full dose) partner. Slices
from the **test** folder (`LD_FD_CT_test`) show how the model does on unseen patients. Slices from the
**training** folder (`LD_FD_CT_train`) flatter it, because the model saw them while learning. Without a
`_fd` file you get an output but no score or error map. The images are the authors' processed dataset
(Google Drive bundle named in the Fast-DDPM README). They are not in this git repository; check the dataset's
usage terms before sharing them publicly.

## 2. Option A: Google Colab

1. Open the notebook: use the "Open in Colab" badge at its top on GitHub, or upload the `.ipynb` yourself
   (File → Upload notebook). If the repository is private, Colab asks you to authorize GitHub.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the cells in order. The first cell clones the Fast-DDPM code and downloads the pretrained checkpoint
   from Hugging Face (about 1.7 GB, a few minutes the first time).
4. When the upload cell asks, upload a `.zip` of `_ld.png` / `_fd.png` pairs (section 4 shows how to make one).
5. The last cell shows one row per image.

A free Colab runtime has **one GPU**, so the parallel notebook's multi-GPU sections run their pieces one
after another there: that checks the splitting and merging are correct, but the wall-clock time does not
shrink. Real multi-GPU timings come from Easley.

## 3. Option B: Easley JupyterHub

### 3.1 One-time: register the `fastddpm` environment as a kernel
A notebook runs in whatever Python its **kernel** points at. The hub's default kernel does not have
PyTorch or the Fast-DDPM dependencies, so register the environment from
[hpc_tutorial.md](hpc_tutorial.md). On the **login node**:

```bash
conda activate fastddpm
```

```bash
conda install conda-forge::ipykernel -y
```

```bash
python -m ipykernel install --user --name fastddpm --display-name "Python (fastddpm)"
```

Check the packages the notebooks import (the environment was built for training, so `pandas` or
`matplotlib` may be missing; install any missing one with `pip install <name>`):

```bash
python -c "import torch, yaml, skimage, matplotlib, pandas, PIL; print(torch.__version__)"
```

In this project the environment is Python 3.10 with `torch 2.2.2+cu121` **[ran]**.

### 3.2 Start a server **with a GPU**
Open `https://easley.alliance.unm.edu/jupyter/hub/spawn` and sign in with your Easley (CARC) account. The page
asks you to **select a job profile**. The first entry, and the default, has **no GPU**:

| Profile | GPU | Use it for |
|---|---|---|
| 1 hour, 1 core, 4GB RAM (default) | none | editing files only: **the notebooks will fail** |
| 4 hours, 32 cores, 120GB RAM | none | |
| 4 hours, 64 cores, 235GB RAM | none | |
| **8 hours, 16 cores, 55GB RAM, 1 L40s GPU** | 1 L40S | **the viewer** |
| **8 hours, 48 cores, 165GB RAM, 3 L40s GPU** | 3 L40S | **the parallel-inference notebook** (sections 2 and 3 then use 3 cards at the same time) |
| 8 hours, 32 cores, 500GB RAM, 1 H100 GPU | 1 H100 | the viewer, faster |
| 8 hours, 32 cores, 950GB RAM | none | |

There is no profile with 2 H100. The 3-L40S profile uses your **whole L40S allowance** (3 GPUs) for up to
8 hours, and a server holds its GPUs until it ends, so **stop it when you are done**: File → Hub Control
Panel → Stop My Server. (Per-user caps: [easley_cluster_notes.md](easley_cluster_notes.md).)

### 3.3 Open the notebook and choose the kernel
1. Get the notebook onto Easley: `git pull` in `~/ECE_551_problems_ML` (it only brings in **committed**
   files), or use the hub file browser's Upload button.
2. Open it, then Kernel → Change Kernel → **Python (fastddpm)**.
3. In the first cell set `CKPT` to a checkpoint that exists. The default
   (`~/carc-scratch/fastddpm-exp/logs/ldfdct_smoke_test/ckpt_1.pth`) stops the cell with an assertion error if
   the file is missing. Any `ckpt_<step>.pth` from one of your runs works, which shows what **your** model
   produces.
4. In the viewer, `INPUT_DIR` already points at the LDFDCT test folder on Easley, so nothing needs uploading;
   it takes 8 slices spread across the folder.

### 3.4 Check that the server really has a GPU
Run this in a cell before anything else:

```python
import os, subprocess, sys, torch
print("python      :", sys.executable)
print("torch       :", torch.__version__, "| built for CUDA:", torch.version.cuda)
print("cuda found  :", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("host        :", os.uname().nodename)
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True))
```

A working result **[ran]** lists a GPU and `cuda found: True`, for example `GPU 0: NVIDIA L40S` on
`easley057`. From a terminal you can also see what the server job requested:

```bash
squeue --me -o "%.10i %.12P %.20j %.10b %.8T %N"
```

The `%b` column should show a GPU (like `gres:gpu:1`).

## 4. Making a sample of image pairs (for Colab)
On Easley, this picks 8 pairs spread across the test set. Swap `LD_FD_CT_test` for `LD_FD_CT_train` to
sample training slices:

```bash
cd ~/carc-scratch/fastddpm-data/data/LD_FD_CT_test
find . -name "*_ld.png" | sort | awk 'NR%400==1' | head -8 | while read f; do echo "$f"; echo "${f%_ld.png}_fd.png"; done | zip ~/sample_pairs.zip -@
```

Then copy it to your computer:

```bash
scp lc2025@easley.alliance.unm.edu:~/sample_pairs.zip .
```

## 5. Problems we hit

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: No GPU` from the first cell, but `import torch` worked | The server was started on the default profile (1 core, 4GB, **no GPU**). The diagnostic cell showed host `easley001`, `nvidia-smi` failing with "couldn't communicate with the NVIDIA driver", and `CUDA_VISIBLE_DEVICES: None`. The kernel and the environment were fine. | Stop the server, start a new one on a profile with a GPU (section 3.2). The host then becomes a GPU node and `nvidia-smi -L` lists a GPU. |
| The notebook runs but PSNR and SSIM are **not printed** above the "generated" panel, and the "generated" titles sit higher than the others | The old code put the scores in the panel's axis label, and `axis("off")` hides axis labels. | Edit the **helper cell** (the one that defines `show`): replace the `set_xlabel(...)` and `set_label_position("top")` lines with `axes[r][1].set_title(f"generated\nPSNR {psnr(out, gt):.2f} dB   SSIM {ssim(gt, out, data_range=1.0):.4f}", fontsize=10)`. **Run that cell first**, then the display cell: rerunning only the display cell keeps the old `show` in memory. |
| "Python (fastddpm)" is missing from the kernel list | The `ipykernel install` step was not run, or was run in another environment | Rerun it (section 3.1) with `fastddpm` activated, then refresh the page |
| `torch.cuda.is_available()` is `False` in Jupyter but `True` on a job | The server has no GPU | Section 3.2 |
| Assertion error naming `CKPT` | The default checkpoint file does not exist on your account | Set `CKPT` to a real `ckpt_<step>.pth` |
| Other jobs of yours wait while the hub is open | The server holds its GPUs for up to 8 hours, and per-user caps count them | Stop the server when finished |
| `Could not solve for environment specs` when creating a new environment | `pytorch-cuda=3.9` mixes up the CUDA and Python versions | Use the `pip` command in section 6 |

## 6. Appendix: a generic PyTorch kernel (the earlier setup notes)
These are the steps we first used to get any PyTorch-with-GPU environment into Jupyter, kept here because
they also explain the pieces above. For this project use `fastddpm` instead; it already matches the
training code.

1. Create and activate an environment on the login node. Always **pin the Python version**; otherwise
   conda picks the newest, which PyTorch may not support yet:

   ```bash
   conda create -n torch_0 python=3.11 -y
   ```

   ```bash
   conda activate torch_0
   ```

2. Install PyTorch with CUDA from the wheel index (not `conda install pytorch-cuda`, see the table above):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Register it as a kernel:

   ```bash
   conda install conda-forge::ipykernel -y
   ```

   ```bash
   python -m ipykernel install --user --name torch_0 --display-name "Python (torch_0)"
   ```

4. Test it **on a GPU node**, not the login node (`cuda.is_available()` is `False` there):

   ```bash
   srun --partition=interactive --gres=gpu:1 --time=00:15:00 --pty bash
   ```

   ```bash
   conda activate torch_0 && python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   The notes expected `2.5.1+cu121 True NVIDIA L40S`. Type `exit` to release the GPU.

5. Start a hub server with a GPU (section 3.2), open a notebook, and choose Kernel → Change Kernel →
   "Python (torch_0)".
