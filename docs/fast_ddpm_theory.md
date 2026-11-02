# Fast-DDPM: Theory Notes

Study notes on *Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for
Medical Image-to-Image Generation* (Jiang, Imran, Zhang, Zhou, Liang, Gong, Shao;
JBHI 2025; arXiv:2405.14802). Written to build a theoretical understanding of the
model and to connect each idea to the code we run on Easley.

How each statement is sourced:
- **[paper]** stated in the paper.
- **[code]** read directly from the `mirthAI/Fast-DDPM` source.
- **[background]** standard diffusion-model material (DDPM, DDIM, score-based
  models) that the paper builds on but does not derive.
- **[computed]** numbers I computed from the formulas, not quoted from the paper.
- **[verify]** something I noticed that should be checked before relying on it.

---

## 1. The problem

Fast-DDPM does **conditional image-to-image generation**: given an input image
(the *condition* $c$), generate a target image $x_0$. [paper]

| Task | Condition $c$ | Target $x_0$ | Dataset | Code name |
|---|---|---|---|---|
| Multi-image super-resolution | previous and next MRI slice | the middle slice | Prostate-MRI-US-Biopsy | `PMUB` |
| Denoising | low-dose CT slice | full-dose CT slice | LDCT-and-Projection-data | `LDFDCT` |
| Translation | T1-weighted MRI | T2-weighted MRI | BraTS 2018 | `BRATS` |

The model is **2D**: it works slice by slice. The paper lists 3D as future work
because 3D diffusion training takes "weeks to months." [paper]

Why diffusion instead of a GAN? The paper argues diffusion gives higher-quality
samples, avoids GAN mode collapse and the balancing act between generator and
discriminator, and is less sensitive to hyperparameters. [paper, Discussion C]

---

## 2. Diffusion models from first principles

### 2.1 Forward process: destroy the image with noise
Start with a clean target $x_0$. Define a ladder of noisy versions
$x_1, \dots, x_T$ where each is the previous one plus a little Gaussian noise:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\right)$$

$\beta_t$ is the *noise schedule*. Fast-DDPM uses the standard linear schedule:
$\beta_t$ rises linearly from $10^{-4}$ to $0.02$ over $T=1000$ steps. [paper, code]

Because each step is Gaussian, the whole chain collapses into a single
closed-form jump from $x_0$ to any $x_t$. [background]

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t) I\right),
\qquad \bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$$

Equivalently, with $\epsilon \sim \mathcal{N}(0, I)$:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

This is the equation behind training: to make a noisy training example you do
not simulate $t$ steps, you jump straight to step $t$.

**Notation map.** The paper writes the same thing in continuous time:
$q(x(t)\mid x_0)=\mathcal{N}(\alpha(t)x_0,\sigma^2(t)I)$ with
$\alpha^2(t)+\sigma^2(t)=1$ (Eq. 1). So the paper's $\alpha^2(t)$ is the DDPM
$\bar\alpha_t$, and the code's `at`/`atm1` variables are $\bar\alpha$ values.
The **signal-to-noise ratio** is $\mathrm{SNR}(t)=\alpha^2(t)/\sigma^2(t)$ and
falls monotonically from $+\infty$ (clean) to about $0$ (pure noise). [paper]

### 2.2 Reverse process: learn to undo the noise
Generation runs the chain backward: start from pure noise $x_T\sim\mathcal{N}(0,I)$
and repeatedly denoise. Each reverse step needs $p(x_{t-1}\mid x_t)$, which is
unknown, so a neural network learns it. [background]

Parameterization used here: a **U-Net** $\epsilon_\theta(x_t, c, t)$ takes the noisy
image, the condition image, and the time step, and **predicts the noise** that
was added. [paper]

### 2.3 The training objective
Sample a data pair $(x_0, c)$, a time step $t$, and noise $\epsilon$. Build $x_t$
with the closed form above. Ask the network to recover $\epsilon$:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,(x_0,c),\epsilon}\;
\big\|\epsilon - \epsilon_\theta\!\big(\alpha(t)x_0+\sigma(t)\epsilon,\; c,\; t\big)\big\|^2
\quad \text{(paper Eq. 9)}$$

Plain mean-squared error, with equal weight on every $t$. Training is regression,
not adversarial, which is why it is stable.

**Why predicting noise works (score view).** The paper also frames this as
score matching (Eqs. 2-6). The *score* of the noisy distribution is the gradient
of the log-density, $\nabla_x\log p(x(t))$, which points toward more likely
images. For the Gaussian forward process the score has a closed form
(Eq. 5):

$$\nabla_{x(t)}\log p(x(t)\mid x_0) = -\frac{x(t)-\alpha(t)x_0}{\sigma^2(t)} = -\frac{\epsilon}{\sigma(t)}$$

So predicting $\epsilon$ is the same as predicting the score up to a known scale,
and the reverse-time SDE (Eq. 3) uses that score to move samples toward the
data. [paper Eq. 2-6, background]

### 2.4 Sampling
Once $\epsilon_\theta$ is trained, sampling is a loop. Two versions matter here.

**DDPM (ancestral) sampling.** Takes all $T=1000$ steps, adding fresh noise at
each. Slow. [background]

**DDIM sampling** (Song et al.). A deterministic (or partly stochastic) update
that lets you **skip steps**. At each step, first estimate the clean image:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}$$

then jump to the next chosen time step $t'$:

$$x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{x}_0 + \sqrt{1-\bar\alpha_{t'}-\sigma^2}\;\epsilon_\theta + \sigma z,
\qquad \sigma = \eta\sqrt{\tfrac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}}\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_{t'}}}$$

With $\eta=0$ (the code default) $\sigma=0$ and sampling is fully deterministic:
the same starting noise gives the same image. [background]

[code] This is exactly `sg_generalized_steps` in `functions/denoising.py`:
`x0_t` is $\hat{x}_0$, `c1` is $\sigma$, and `c2` is
$\sqrt{1-\bar\alpha_{t'}-\sigma^2}$. The comment in the source says
"Equation (12)", referring to the DDIM paper.

---

## 3. The cost problem Fast-DDPM attacks

Standard DDPM needs about 1000 network evaluations per generated image, and
training spreads effort over all 1000 time steps. The paper reports
[paper]:

- DDPM training "takes days to weeks"; sampling an image volume "takes minutes
  to hours."
- Fast samplers (DDIM, DPM-Solver++) cut sampling to about 10-100 steps, but
  the network was still **trained on all 1000 steps**.

**The paper's key observation.** If you sample with 100-step DDIM, the network is
only ever *queried* at 100 of the 1000 time points. The other 900 were trained
for nothing. Training and sampling are *misaligned*. [paper, Introduction]

Two further facts about the schedule explain why this waste is large:

1. A schedule with too few steps never reaches pure noise at the end; one with
   too many reaches pure noise early and the late steps are redundant. The paper
   shows (Fig. 2) that a good $T$ lies roughly between 500 and 2000, which is why
   they keep the standard $T=1000$ schedule as the underlying curve. [paper]
2. [computed] With the linear $\beta$ schedule, $\bar\alpha_t < 0.01$ from
   $t\approx 673$ onward, and $\bar\alpha_{999}\approx 4\times10^{-5}$. More
   than a third of the 1000 steps are already almost pure noise (signal amplitude
   $\sqrt{\bar\alpha_t}$ below 0.1).

---

## 4. The Fast-DDPM idea

**Train and sample on the same small set of time steps.** [paper]

Instead of training on all 1000 time points and sampling on a subset, pick about
10 time points on the standard noise curve and use *only those* for both training
and sampling. The training loop (Algorithm 1) draws $t$ from this small set; the
sampling loop (Algorithm 2) walks down the same set.

The paper's justification is an analogy with diffeomorphic image registration,
where about 10 discrete time points suffice to solve an ODE accurately. Their
hypothesis is that at sampling time only about 10 points are visited, so the
denoiser only needs to be accurate there. [paper]

A second argument comes from the Discussion section: in *conditional* generation
the condition image already carries most of the structure, so the model refines
rather than invents. It can then spend its capacity on the noise levels that
matter instead of "being diluted across unnecessary diffusion steps." [paper]

Claimed result: training time $0.2\times$ and sampling time $0.01\times$ of DDPM,
with equal or better quality. [paper]

### 4.1 Choosing which 10 time points: the two noise schedulers
The underlying curve is $\alpha^2(t)$ from the $T=1000$ linear-$\beta$ schedule
(Eq. 8). Fast-DDPM samples it at ~10 points in two ways. [paper]

**Uniform:** evenly spaced points on $[0,1]$.

**Non-uniform:** a heuristic that puts **60% of the points where $t>699$**
(the high-noise region) and 40% where $t\le 699$. The authors cite prior work
suggesting later steps contribute more to perceptual quality. They do not give a
closed-form rule: "this heuristic approach." [paper]

[code] `sg_sample_image` / `sg_train` in `runners/diffusion.py`:

```
uniform:      skip = 1000 // timesteps ; seq = [0, 99, 199, ..., 999]
non-uniform:  seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
```

[computed] The noise level at each point (these are the only noise levels the
network ever sees during training):

| $t$ | $\bar\alpha_t$ | signal $\sqrt{\bar\alpha_t}$ | noise $\sqrt{1-\bar\alpha_t}$ | SNR |
|---:|---:|---:|---:|---:|
| 0 | 0.99990 | 0.9999 | 0.0100 | 9999 |
| 99 | 0.89702 | 0.9471 | 0.3209 | 8.71 |
| 199 | 0.65904 | 0.8118 | 0.5839 | 1.93 |
| 299 | 0.39642 | 0.6296 | 0.7769 | 0.657 |
| 399 | 0.19515 | 0.4418 | 0.8971 | 0.243 |
| 499 | 0.07859 | 0.2803 | 0.9599 | 0.0853 |
| 599 | 0.02588 | 0.1609 | 0.9870 | 0.0266 |
| 699 | 0.00697 | 0.0835 | 0.9965 | 0.00702 |
| 799 | 0.00153 | 0.0391 | 0.9992 | 0.00153 |
| 899 | 0.00028 | 0.0166 | 0.9999 | 0.000275 |
| 999 | 0.00004 | 0.0064 | 1.0000 | 0.00004 |

Rows 0 to 999 in steps of 100 (shifted by one) are the uniform set. The
non-uniform set is 0, 199, 399, 599, 699, 799, 849, 899, 949, 999.

Note how little the late rows differ from one another: past $t\approx 700$ the
image is essentially noise. The non-uniform scheduler spends five of its ten
points there; the uniform one spends about four.

---

## 5. How the conditioning works in code

[code] The condition is injected by **channel concatenation**. The network
input is the condition image stacked with the noisy target along the channel axis.

`functions/losses.py`:

```python
def sg_noise_estimation_loss(model, x_img, x_gt, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1,1,1,1)   # alpha-bar_t
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()                    # noisy target
    output = model(torch.cat([x_img, x], dim=1), t.float())       # condition + noisy target
    return (e - output).square().sum(dim=(1,2,3)).mean(dim=0)     # Eq. 9
```

- `x_img` is the condition (for LDFDCT: the low-dose slice, dict key `'LD'`).
- `x_gt` is the target (full-dose slice, dict key `'FD'`).
- `e` is $\epsilon$; `a` is $\bar\alpha_t$.
- The loss is the **sum** of squared errors over channels and pixels, averaged
  over the batch, so its absolute value scales with image size.

Three task variants map to three loss functions in `loss_registry`:

| `model.type` | Loss | Network input channels | Used by |
|---|---|---|---|
| `simple` | `noise_estimation_loss` | 1 (noisy image only, unconditional) | baseline |
| `sg` | `sg_noise_estimation_loss` | 2 = 1 condition + 1 noisy target | LDFDCT, BRATS |
| `sr` | `sr_noise_estimation_loss` | 3 = 2 neighbor slices + 1 noisy target | PMUB |

This is why `in_channels` is 2 or 3 in the configs.

**Timestep sampling during training** [code]: not plain uniform sampling. The loop
uses *antithetic* sampling: draw random indices `idx_1` into the set of chosen
time points, pair each with its mirror `len - idx_1 - 1`, and use both. This
balances low-noise and high-noise steps within a batch and reduces gradient
variance. The paper does not mention this detail.

**Architecture** [code, config]: a U-Net following the DDIM codebase
(`models/diffusion.py`): base width `ch: 128`, per-level multipliers
`ch_mult: [1,1,2,2,4,4]`, 2 residual blocks per level, self-attention at 16x16
resolution, no dropout. Weights are tracked with an **exponential moving average**
(`ema_rate: 0.999`); sampling loads the EMA weights, not the raw ones. The time
step enters through a sinusoidal (sin/cos) embedding like positional encoding in
transformers (`get_timestep_embedding`), passed through a two-layer MLP of width
`4*ch`, then projected and **added inside every residual block**
(`h = h + temb_proj(...)`). Blocks use GroupNorm with 32 groups. This is how one
network handles every noise level: the block activations are shifted according to
the current $t$.

**Checkpoints** [code]: a Python list
`[model_state, optimizer_state, epoch, step, ema_state]`. Sampling needs
`states[0]` and `states[-1]`; resuming training needs all five.

---

## 6. Experimental setup in the paper

[paper, Sections IV-V]

- **Hardware:** one node with 4 NVIDIA A100 (80 GB). PyTorch 1.12.1, Python 3.10.6.
- **Training:** Adam, batch size 16, 400,000 iterations for Fast-DDPM versus
  2,000,000 for the DDPM baseline. $\beta$: $10^{-4}\to0.02$, $T=1000$.
- **Scheduler choice per task:** non-uniform for super-resolution; uniform for
  denoising and translation.
- **Data:** 256x256, normalized to $[-1,1]$. Super-resolution: 6979 training
  triplets / 580 test triplets. Denoising: 13,211 training and 3,501 test pairs
  (38 / 10 patients). Translation: 5760 training / 768 test pairs.

### Metrics
**PSNR** (peak signal-to-noise ratio), in decibels, higher is better:

$$\mathrm{PSNR} = 20\log_{10}\!\left(\frac{\mathrm{MAX}_I}{\sqrt{\mathrm{MSE}}}\right)$$

It is a log-scaled pixel-error measure.
[paper Eq. 10]

**SSIM** (structural similarity), in $[0,1]$, higher is better. It compares
local mean, variance, and covariance between prediction $\hat x$ and ground
truth $x$ rather than raw pixel error: [paper Eq. 11]

$$\mathrm{SSIM}(x,\hat x)=\frac{(2\mu_x\mu_{\hat x}+c_1)(2\sigma_{x\hat x}+c_2)}{(\mu_x^2+\mu_{\hat x}^2+c_1)(\sigma_x^2+\sigma_{\hat x}^2+c_2)}$$

---

## 7. Results

### Paper results (Tables I-III) [paper]
| Task | Method | PSNR | SSIM | Train time | Inference |
|---|---|---|---|---|---|
| Super-resolution | DDPM | 25.3 | 0.83 | 136 h | 3.7 min |
| | Fast-DDPM | 27.1 | 0.89 | 26 h | 2.3 s |
| Denoising | DDPM | 35.4 | 0.87 | 141 h | 21.4 min |
| | Fast-DDPM | **37.5** | **0.92** | 26 h | 12.5 s |
| Translation | DDPM | 26.3 | 0.89 | 135 h | 22.2 min |
| | Fast-DDPM | 26.3 | 0.89 | 27 h | 13.2 s |

Reading these: on translation Fast-DDPM only *matches* DDPM quality; the gains
there are speed. On super-resolution and denoising it is also *better*.

### Our reproduction (inference, pretrained checkpoint)
| Task | PSNR | SSIM | Hardware / software |
|---|---|---|---|
| Denoising (LDFDCT, 3,501 test images) | 37.46 | 0.9155 | Easley L40S, PyTorch 2.2.2 |

Matches the paper's 37.5 / 0.92 despite different GPU and PyTorch version. This
validates the pipeline; it does not validate our own training.

---

## 8. The ablations: what they teach

### Number of time steps (Table IV) [paper]
| Steps | Super-res PSNR | Denoising PSNR | Translation PSNR |
|---:|---:|---:|---:|
| 3 | 24.8 | 32.7 | 25.3 |
| 5 | 26.9 | 37.4 | 25.8 |
| 10 | **27.1** | **37.5** | 26.3 |
| 50 | 26.2 | 36.9 | 26.5 |
| 100 | 25.8 | 36.5 | **26.6** |
| 1000 | 25.3 | 35.4 | 26.3 |

Pattern: **more steps is not better.** Super-resolution and denoising peak near
10 steps and *decline* with more. Translation peaks at 100.

The paper's explanation: super-resolution and denoising start from an input that
already has strong structure, so excess diffusion steps "introduce unnecessary
noise or blurring." Translation *creates* a new appearance (T1 to T2 contrast), so
it benefits from more refinement steps, up to a point. [paper] This is a
hypothesis supported by the data, not a proven mechanism.

### Scheduler (Table V) [paper]
| Scheduler | Super-res PSNR / SSIM | Denoising | Translation |
|---|---|---|---|
| Uniform | 26.6 / 0.88 | **37.5 / 0.92** | **26.3 / 0.89** |
| Non-uniform | **27.1 / 0.89** | 37.3 / 0.91 | 26.1 / 0.88 |

The best scheduler depends on the task. The authors list adaptive, learned
schedulers as future work. [paper]

---

## 9. Limitations and open questions (from the paper)

[paper, Discussion D]
1. Noise schedulers are **hand-picked**, not learned.
2. The model is **2D**. They propose 3D via diffusion in a small latent space
   learned by an autoencoder.
3. Optimal step count is task-dependent and chosen by ablation.

[verify] Items I noticed that deserve checking before we cite them:

1. **Learning rate.** Paper says $2\times10^{-4}$; the configs in the repo say
   `lr: 0.00002` ($2\times10^{-5}$). One of them is a typo or the configs were
   changed. Matters for any from-scratch reproduction.
2. **Uniform scheduler length.** In code, `range(-1, 1000, 100)` with the first
   element set to 0 gives **11** points (0, 99, ..., 999), not 10. [computed] The
   non-uniform list has exactly 10. Confirm whether this off-by-one is intentional.
3. **`n_iters` is unused.** `training.n_iters: 5000000` in the config is never
   checked in the training loop I read; the loop runs until `n_epochs`. The paper's
   400,000-iteration figure is not enforced by the code, so a full reproduction
   needs a manual stop.
4. **Validation loop.** `validation_freq` is set to an effectively infinite value
   and I found no validation code in `sg_train`, so only training loss is logged.
5. **`DataParallel`.** Training wraps the model in `torch.nn.DataParallel`, not
   `DistributedDataParallel`. On a single GPU this is harmless; it is the older,
   slower multi-GPU API.
6. **Numerical comparability.** Our PyTorch 2.2.2 / L40S results differ from the
   paper's 1.12.1 / A100 setup. Close agreement on inference does not guarantee
   identical training behavior.

---

## 10. Paper-to-code map

| Concept | Paper | File / function |
|---|---|---|
| Noise schedule $\beta_t$, $\bar\alpha_t$ | Eq. 7-8 | `runners/diffusion.py::get_beta_schedule`, `Diffusion.__init__` |
| Training loss | Eq. 9, Alg. 1 | `functions/losses.py::sg_noise_estimation_loss` |
| Training loop, time-step set | Alg. 1 | `runners/diffusion.py::sg_train` |
| DDIM sampling | Alg. 2 | `functions/denoising.py::sg_generalized_steps` |
| Choosing the sampling set | Fig. 3 | `runners/diffusion.py::sg_sample_image` |
| U-Net | Sec. IV-C | `models/diffusion.py::Model` |
| EMA weights | (not in paper) | `models/ema.py` |
| PSNR / SSIM | Eq. 10-11 | `functions/losses.py::calculate_psnr`, `skimage.metrics` |
| Metric aggregation | Tables I-III | `runners/diffusion.py::sg_sample_fid` (prints `Average: PSNR..., SSIM...`) |

---

## 11. Experiments that would build understanding

Each one ties theory to something we can run on Easley.

1. **Watch the forward process.** Take one CT slice, apply
   $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ at the 11 uniform
   $t$ values, and plot them. Confirms the table in section 4.1 visually.
2. **Reproduce Table IV.** Run inference with `--timesteps 3, 5, 10, 50, 100` on
   the pretrained denoising checkpoint. [verify] The checkpoint was trained for 10
   steps, so inference at other step counts tests something different from the
   paper's ablation (which retrained for each step count). Decide which one we mean.
3. **Visualize $\hat x_0$ along the sampling path.** `sg_generalized_steps` already
   returns `x0_preds`; plotting them shows how the estimate sharpens step by step.
4. **Uniform vs. non-uniform on LDFDCT.** Reproduce the Table V row for denoising.
5. **Check the learning-rate discrepancy** by training briefly at both values and
   comparing loss curves.
6. **Memory and time scaling.** Measure seconds per step and GPU memory at several
   batch sizes; this answers "find the limits" and gives the 2D-to-3D argument
   concrete numbers.

---

## 12. Equation reference

Source book / chapter for the equations: **empty** (add once confirmed).

---

## 13. Glossary

- **Forward / noising process:** fixed, no learning; adds Gaussian noise.
- **Reverse / denoising process:** learned; removes noise step by step.
- **$\bar\alpha_t$ (alpha-bar):** cumulative product of $(1-\beta)$; the fraction of
  signal variance left at step $t$. The paper's $\alpha^2(t)$.
- **$\epsilon$-prediction:** the network predicts the added noise, not the image.
- **Score:** gradient of the log-density; points toward more likely images.
- **DDIM:** deterministic sampler that allows skipping time steps.
- **$\eta$:** DDIM stochasticity knob; 0 means deterministic.
- **EMA:** exponential moving average of weights; used at sampling time.
- **Condition:** the input image the generated output is conditioned on.
- **Antithetic sampling:** pairing each random draw with its mirror to reduce variance.
- **PSNR / SSIM:** pixel-error and structural-similarity quality metrics.
