#!/usr/bin/env python3
"""Generates slides.html for the Fast-DDPM code notes. Print to PDF with headless Chrome (see README at bottom of this file)."""
import html, math, re

# ---------- helpers ----------
KW = r"\b(def|class|return|if|elif|else|for|in|import|from|with|not|and|or|True|False|None|try|except|assert)\b"

WARN = []
def _split(line):
    stripped = line.lstrip()
    if stripped.startswith("#") and not stripped.startswith("#SBATCH") and not stripped.startswith("#!"):
        return "", line
    m = re.search(r"\s+#(\s.*|$)", line)
    return (line[:m.start()], line[m.start():]) if m else (line, "")

def _vis(left, cmt):
    bo = re.fullmatch(r"\s*#?\s*(⟦\d⟧)\s*", cmt) if cmt else None
    return len(left) + 5 if bo else len(re.sub(r"⟦\d⟧", "   ", left + cmt))

def code(src, size=None, colw=694, cap=20):
    lines = src.strip("\n").split("\n")
    maxvis = max(_vis(*_split(l)) for l in lines)
    fit = (colw - 34) / (maxvis * 0.602)
    size = max(11, min(cap, int(fit * 2) / 2))
    out = []
    for line in lines:
        left, cmt = _split(line)
        left = re.sub(KW, r'<span class="k">\1</span>', html.escape(left))
        badge_only = re.fullmatch(r"\s*#?\s*(⟦\d⟧)\s*", cmt) if cmt else None
        if badge_only:
            s = left + "  " + badge_only.group(1)
        else:
            s = left + (f'<span class="c">{html.escape(cmt)}</span>' if cmt else "")
        s = re.sub(r"⟦(\d)⟧", r'<b class="n">\1</b>', s)
        out.append(s)
    return f'<pre class="code" style="font-size:{size}px">' + "\n".join(out) + "</pre>"

def legend(items):
    rows = "".join(f'<li><b class="n">{i+1}</b><span>{t}</span></li>' for i, t in enumerate(items))
    return f'<ol class="legend">{rows}</ol>'

slides = []
def slide(title, body, src=None, draft=False, cls=""):
    slides.append(dict(title=title, body=body, src=src, draft=draft, cls=cls))

def eq(s):
    return f'<div class="eq">{s}</div>'

def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    r = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><tr>{h}</tr>{r}</table>'

# ---------- numbers computed from the real schedule ----------
T = 1000
betas = [1e-4 + (0.02 - 1e-4) * i / (T - 1) for i in range(T)]
abar, p = [], 1.0
for b in betas:
    p *= (1 - b); abar.append(p)
uni = list(range(-1, 1000, 100)); uni[0] = 0
non = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]

def curve_svg():
    W, H, L, B, Tp = 560, 300, 60, 40, 15
    pw, ph = W - L - 15, H - B - Tp
    X = lambda t: L + pw * t / 999
    Y = lambda a: Tp + ph * (1 - a)
    pts = " ".join(f"{X(t):.1f},{Y(abar[t]):.1f}" for t in range(0, 1000, 5))
    s = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="plot">']
    for a in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{L}" x2="{W-15}" y1="{Y(a):.1f}" y2="{Y(a):.1f}" stroke="#e3e3e3"/>')
        s.append(f'<text x="{L-8}" y="{Y(a)+4:.1f}" text-anchor="end" font-size="12" fill="#555">{a:g}</text>')
    for t in (0, 200, 400, 600, 800, 999):
        s.append(f'<text x="{X(t):.1f}" y="{H-B+18}" text-anchor="middle" font-size="12" fill="#555">{t}</text>')
    s.append(f'<text x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle" font-size="13" fill="#222">time step t</text>')
    s.append(f'<text x="14" y="{Tp+ph/2:.0f}" text-anchor="middle" font-size="13" fill="#222" transform="rotate(-90 14 {Tp+ph/2:.0f})">ᾱ<tspan font-size="10" dy="3">t</tspan></text>')
    s.append(f'<polyline points="{pts}" fill="none" stroke="#1f5fbf" stroke-width="2.5"/>')
    for t in uni:
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(abar[t]):.1f}" r="5" fill="#d9480f"/>')
    s.append('</svg>')
    return "".join(s)

def unet_svg():
    levels = [("256²", 128), ("128²", 128), ("64²", 256), ("32²", 256), ("16²", 512), ("8²", 512)]
    W, H = 640, 400
    s = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="plot" font-family="Helvetica, Arial, sans-serif">']
    n = len(levels)
    rowh = 56
    for i, (res, ch) in enumerate(levels):
        y = 20 + i * rowh
        w = 150 - i * 4
        att = " + attention" if res == "16²" else ""
        for side, x in (("down", 30), ("up", W - 30 - w)):
            fill = "#dbe8fb" if side == "down" else "#fde6d8"
            s.append(f'<rect x="{x}" y="{y}" width="{w}" height="38" rx="5" fill="{fill}" stroke="#555"/>')
            s.append(f'<text x="{x+w/2}" y="{y+16}" text-anchor="middle" font-size="13" font-weight="bold">{res} × {ch}ch</text>')
            s.append(f'<text x="{x+w/2}" y="{y+31}" text-anchor="middle" font-size="11" fill="#444">{"2 ResBlocks"+att if side=="down" else "3 ResBlocks"+att}</text>')
        s.append(f'<line x1="{30+w}" x2="{W-30-w}" y1="{y+19}" y2="{y+19}" stroke="#2b8a3e" stroke-width="1.6" stroke-dasharray="5,3"/>')
        if i < n - 1:
            s.append(f'<text x="{30+w/2}" y="{y+52}" text-anchor="middle" font-size="12" fill="#555">↓ downsample</text>')
            s.append(f'<text x="{W-30-w/2}" y="{y+52}" text-anchor="middle" font-size="12" fill="#555">↑ upsample</text>')
    s.append(f'<text x="{W/2}" y="{20+0*rowh+14}" text-anchor="middle" font-size="12" fill="#2b8a3e">skip connection: torch.cat([h, hs.pop()], dim=1)</text>')
    my = 20 + (n - 1) * rowh + 46
    s.append(f'<text x="{W/2}" y="{my-6}" text-anchor="middle" font-size="13" fill="#444">bottleneck at 8²: ResBlock → Attention → ResBlock</text>')
    s.append('</svg>')
    return "".join(s)

# =====================================================================
# 1 Title
slide("", f"""
<div class="titlebox">
  <div class="big">Fast-DDPM: Code Notes</div>
  <div class="sub">Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation</div>
  <div class="meta">ECE 551 project &nbsp;|&nbsp; Luis Carrillo<br>
  Code: github.com/mirthAI/Fast-DDPM &nbsp;|&nbsp; Run on UNM Easley (L40S, PyTorch 2.2.2)</div>
  <div class="note">Draft. Code is quoted from the repository; slides marked DRAFT depend on experiments not yet run.</div>
</div>""", cls="title")

# 2 References
slide("References", """
<ul class="refs">
<li><b>Paper.</b> H. Jiang, M. Imran, T. Zhang, Y. Zhou, M. Liang, K. Gong, W. Shao, “Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation,” IEEE JBHI, 2025. arXiv:2405.14802</li>
<li><b>Code.</b> github.com/mirthAI/Fast-DDPM (adapted by the authors from the DDIM codebase, github.com/ermongroup/ddim)</li>
<li><b>Sampler.</b> J. Song, C. Meng, S. Ermon, “Denoising Diffusion Implicit Models,” 2020 (arXiv:2010.02502)</li>
<li><b>Base method.</b> J. Ho, A. Jain, P. Abbeel, “Denoising Diffusion Probabilistic Models,” NeurIPS 2020</li>
<li><b>Textbook.</b> A. Torralba, P. Isola, W. T. Freeman, <i>Foundations of Computer Vision</i>, MIT Press, 2024 (visionbook.mit.edu), diffusion chapter: <i>chapter/equation numbers: empty</i></li>
<li><b>Index of models and datasets.</b> github.com/pattichis/AIM</li>
<li><b>Format reference and 3D example.</b> M. S. Pattichis, MLSP 2026 tutorial, github.com/pattichis/MLSP2026</li>
</ul>
<div class="note2">Equations on these slides are written out directly; nothing is copied from the book. Long code lines are wrapped for display; the logic is unchanged.</div>
""")

# 3 The task
slide("The task: conditional image-to-image generation", f"""
<p class="lead">Given a <b>condition</b> image <i>c</i>, generate the <b>target</b> image <i>x<sub>0</sub></i>.</p>
{table(["Task","Condition c","Target x<sub>0</sub>","Dataset","--dataset"],[
 ["Denoising","low-dose CT slice","full-dose CT slice","LDCT-and-Projection-data","<code>LDFDCT</code>"],
 ["Translation","T1-weighted MRI","T2-weighted MRI","BraTS 2018","<code>BRATS</code>"],
 ["Super-resolution","previous + next MRI slice","middle MRI slice","Prostate-MRI-US-Biopsy","<code>PMUB</code>"]])}
<ul class="tight">
<li>The model is <b>2D</b>: it processes one 256×256 slice at a time.</li>
<li>These slides follow <b>LDFDCT</b> (CT denoising), the task we reproduced.</li>
<li>Same machinery for all three; they differ in how many condition channels go in (1 or 2).</li>
</ul>
""", src="paper Sec. IV-A; README.md; runners/diffusion.py")

# 4 Two boxes
def flow(steps, cls):
    return '<div class="flow">' + '<span class="arr">→</span>'.join(f'<div class="box {cls}"><span>{s}</span></div>' for s in steps) + "</div>"
slide("Diffusion in two phases", f"""
<div class="h2">Training</div>
{flow(["Pair<br>(c, x<sub>0</sub>)","Pick a time step t","Jump to noisy x<sub>t</sub>","U-Net predicts the noise ε<br>from (c, x<sub>t</sub>, t)","Squared error;<br>update weights"],"tr")}
<div class="h2" style="margin-top:34px">Deployment (sampling)</div>
{flow(["Condition c<br>+ random noise x<sub>T</sub>","U-Net predicts ε","DDIM step to the<br>next chosen t","repeat ~10 times","Generated<br>target x̂<sub>0</sub>"],"dp")}
<div class="callout">Fast-DDPM's change: <b>train and sample on the same ~10 time steps</b>, instead of training on 1,000 and sampling on a subset.</div>
""", src="paper Sec. III, Algorithms 1 and 2")

# 5 Data
slide("Training data: a pair of images", f"""
<div class="cols">
<div>{code('''
class LDFDCT(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        self.img_ld_path, self.img_fd_path = \\
            get_paths_from_images(dataroot)                          # ⟦1⟧

    def __getitem__(self, index):
        img_LD = Image.open(self.img_ld_path[index]).convert("L")   # ⟦2⟧
        img_FD = Image.open(self.img_fd_path[index]).convert("L")
        img_LD = img_LD.resize((self.img_size, self.img_size))      # ⟦3⟧
        img_FD = img_FD.resize((self.img_size, self.img_size))
        [img_LD, img_FD] = transform_augment(
            [img_LD, img_FD], split=self.split, min_max=(-1, 1))     # ⟦4⟧
        return {'FD': img_FD, 'LD': img_LD, 'case_name': case_name}  # ⟦5⟧
''', 15)}</div>
<div>{legend([
 "Finds <code>*ld.png</code> and <code>*fd.png</code> files and sorts both lists, so index <i>i</i> in each is the same slice.",
 "Grayscale, one channel.",
 "Resized to 256×256 (<code>image_size</code> in the config).",
 "Scaled to [−1, 1]; training also applies a random horizontal flip to the pair.",
 "Keys <code>'LD'</code> (condition) and <code>'FD'</code> (target) are hard-coded in the training loop."])}</div>
</div>""", src="datasets/LDFDCT.py, datasets/sr_util.py")

# 6 Noise schedule
slide("The noise schedule", f"""
<div class="cols">
<div>
{eq("β<sub>t</sub> rises linearly from 10<sup>−4</sup> to 0.02, &nbsp; t = 1 … 1000")}
{eq("ᾱ<sub>t</sub> = ∏<sub>s≤t</sub> (1 − β<sub>s</sub>)")}
{code('''
betas = np.linspace(beta_start, beta_end,
                    num_diffusion_timesteps)   # ⟦1⟧
alphas = 1.0 - betas
alphas_cumprod = alphas.cumprod(dim=0)         # ⟦2⟧
''', 15)}
{legend(["Linear β schedule (config: 0.0001 → 0.02, 1000 steps).", "<b>ᾱ<sub>t</sub></b>: the fraction of the original signal left at step t. The paper writes it α²(t)."])}
</div>
<div>{curve_svg()}
<div class="cap">ᾱ<sub>t</sub> for the real schedule (computed). Orange dots: the 11 time steps used by the <b>uniform</b> set. By t ≈ 673, ᾱ &lt; 0.01: the image is almost pure noise.</div></div>
</div>""", src="runners/diffusion.py::get_beta_schedule, Diffusion.__init__")

# 7 forward jump
slide("The forward process in one jump", f"""
{eq("x<sub>t</sub> = √ᾱ<sub>t</sub> · x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) · ε, &nbsp;&nbsp; ε ~ N(0, I)")}
<div class="cols">
<div>{code('''
def sg_noise_estimation_loss(model, x_img, x_gt, t, e, b,
                             keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)   # ⟦1⟧
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()                      # ⟦2⟧
''', 14.5)}</div>
<div>{legend([
 "<code>a</code> is ᾱ<sub>t</sub> for each image in the batch (one t per image), reshaped to broadcast over pixels.",
 "The noisy <b>target</b> <code>x</code>. Note it noises <code>x_gt</code> (the full-dose image), not the condition. <code>e</code> is ε."])}</div>
</div>
<div class="callout">No loop over t: because every step is Gaussian, any noise level is reachable directly from x<sub>0</sub>. This is why training can pick any t cheaply.</div>
""", src="functions/losses.py::sg_noise_estimation_loss")

# 8 time steps
rows = [[t, f"{abar[t]:.4f}", f"{math.sqrt(abar[t]):.3f}", f"{math.sqrt(1-abar[t]):.3f}"] for t in uni]
slide("Fast-DDPM's change: which time steps?", f"""
<div class="cols">
<div>{code('''
if self.args.scheduler_type == 'uniform':
    skip = self.num_timesteps // self.args.timesteps          # ⟦1⟧
    t_intervals = torch.arange(-1, self.num_timesteps, skip)
    t_intervals[0] = 0                                         # ⟦2⟧
elif self.args.scheduler_type == 'non-uniform':
    t_intervals = torch.tensor(
        [0, 199, 399, 599, 699, 799, 849, 899, 949, 999])      # ⟦3⟧
''', 14)}
{legend(["1000 // 10 = 100, so every 100th step.",
 "<code>arange(−1, 1000, 100)</code> gives −1, 99, …, 999; setting the first to 0 leaves <b>11</b> values (see “To verify”).",
 "Non-uniform: more points at high noise (t ≥ 699). Paper: better for super-resolution; uniform better for denoising/translation."])}</div>
<div>{table(["t","ᾱ<sub>t</sub>","signal √ᾱ","noise √(1−ᾱ)"], rows, "small")}
<div class="cap">Uniform set, computed from the schedule. Training and sampling both use <i>only</i> these noise levels.</div></div>
</div>""", src="runners/diffusion.py::sg_train, sg_sample_image")

# 9 antithetic
slide("Picking t for each training batch", f"""
<div class="cols">
<div>{code('''
idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))   # ⟦1⟧
idx_2 = len(t_intervals) - idx_1 - 1                              # ⟦2⟧
idx = torch.cat([idx_1, idx_2], dim=0)[:n]                        # ⟦3⟧
t = t_intervals[idx].to(self.device)                              # ⟦4⟧

loss = loss_registry[config.model.type](
    model, x_img, x_gt, t, e, b)                                      # ⟦5⟧
''', 15)}</div>
<div>{legend([
 "Random indices into the small set of chosen time steps.",
 "Each index is paired with its mirror (first ↔ last, …): if one image gets a low-noise step, another gets a high-noise step.",
 "Concatenate and trim to the batch size <i>n</i>.",
 "Map indices to actual time steps.",
 "<code>config.model.type</code> = <code>'sg'</code> selects the loss on the previous slide."])}</div>
</div>
<div class="callout">This is <i>antithetic sampling</i>: it balances noise levels inside every batch and lowers gradient variance. The paper does not mention it; it is in the code.</div>
""", src="runners/diffusion.py::sg_train")

# 10 time embedding
slide("Telling the network which t it is seeing", f"""
{eq("emb(t) = [ sin(t·ω₁), …, sin(t·ω<sub>k</sub>), cos(t·ω₁), …, cos(t·ω<sub>k</sub>) ], &nbsp; ω<sub>j</sub> = 10000<sup>−j/(k−1)</sup>")}
<div class="cols">
<div>{code('''
half_dim = embedding_dim // 2
emb = math.log(10000) / (half_dim - 1)
emb = torch.exp(
    torch.arange(half_dim, dtype=torch.float32) * -emb)                # ⟦1⟧
emb = timesteps.float()[:, None] * emb[None, :]                        # ⟦2⟧
emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)               # ⟦3⟧

# in Model.forward
temb = get_timestep_embedding(t, self.ch)
temb = self.temb.dense[0](temb)                                         # ⟦4⟧
temb = nonlinearity(temb)
temb = self.temb.dense[1](temb)
''', 15)}</div>
<div>{legend([
 "A ladder of frequencies from 1 down to 1/10000.",
 "Multiply each time step by every frequency.",
 "Sine and cosine of those products: a unique vector for each t (the same idea as positional encoding in transformers).",
 "Two linear layers expand it to width 4·ch (512)."])}</div>
</div>""", src="models/diffusion.py::get_timestep_embedding, Model.forward")

# 11 resblock
slide("Residual block: where t enters", f"""
<div class="cols">
<div>{code('''
def forward(self, x, temb):
    h = x
    h = self.norm1(h)
    h = nonlinearity(h)                    # SiLU: x * sigmoid(x)
    h = self.conv1(h)

    h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]   # ⟦1⟧

    h = self.norm2(h)
    h = nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
        x = self.nin_shortcut(x)                                    # ⟦2⟧
    return x + h                                                    # ⟦3⟧
''', 14)}</div>
<div>{legend([
 "The time embedding is projected to one number per channel and <b>added to every pixel</b> of the feature map. This is how one network behaves differently at different noise levels.",
 "A 1×1 convolution matches channel counts when they differ.",
 "Residual connection: the block learns a correction to its input."])}
<div class="cap">Normalization is GroupNorm with 32 groups. Dropout is 0.0 in the configs.</div></div>
</div>""", src="models/diffusion.py::ResnetBlock.forward")

# 12 unet
slide("The U-Net", f"""
<div class="cols">
<div>{unet_svg()}</div>
<div>{code('''
# downsampling: save every output for later
hs = [self.conv_in(x)]
...
# upsampling: concatenate the saved features
h = self.up[i_level].block[i_block](
    torch.cat([h, hs.pop()], dim=1), temb)     # ⟦1⟧
''', None, 448)}
{legend(["Skip connection: fine detail from the way down is handed back on the way up. Every ResBlock also receives <code>temb</code>."])}
<div class="cap">Widths and sizes derived from the config (<code>ch=128</code>, <code>ch_mult=[1,1,2,2,4,4]</code>, <code>attn_resolutions=[16]</code>, 2 blocks per level) and the code structure; not printed from a run.</div></div>
</div>""", src="models/diffusion.py::Model; configs/ldfd_linear.yml")

# 13 conditioning
slide("Conditioning: stack the condition with the noisy target", f"""
<div class="cols">
<div>{code('''
output = model(torch.cat([x_img, x], dim=1), t.float())   # ⟦1⟧
''', None, 694)}
{legend(["Concatenate along the channel axis (<code>dim=1</code>), then run the U-Net."])}</div>
<div class="callout" style="margin-top:0">The condition is never noised. The network sees it clean at every step and learns to use it as a guide.</div>
</div>
<div class="shapes">
<div class="sh">x_img (LD)<br><b>B × 1 × 256 × 256</b></div><span>+</span>
<div class="sh">x<sub>t</sub> (noisy FD)<br><b>B × 1 × 256 × 256</b></div><span>→</span>
<div class="sh hl">U-Net input<br><b>B × 2 × 256 × 256</b></div><span>→</span>
<div class="sh">predicted ε<br><b>B × 1 × 256 × 256</b></div>
</div>
<div style="margin-top:22px">{table(["model.type","loss","input channels","tasks"],[
 ["<code>sg</code>","sg_noise_estimation_loss","2 = 1 condition + 1 noisy target","LDFDCT, BRATS"],
 ["<code>sr</code>","sr_noise_estimation_loss","3 = 2 neighbouring slices + 1 noisy target","PMUB"],
 ["<code>simple</code>","noise_estimation_loss","1 (unconditional)","baseline"]])}</div>
""", src="functions/losses.py (loss_registry); configs/*.yml (in_channels)")

# 14 loss
slide("The training objective", f"""
{eq("L(θ) = E<sub>t, (x<sub>0</sub>, c), ε</sub> ‖ ε − ε<sub>θ</sub>( √ᾱ<sub>t</sub> x<sub>0</sub> + √(1−ᾱ<sub>t</sub>) ε , c , t ) ‖²")}
<div class="cols">
<div>{code('''
return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)   # ⟦1⟧
''', 15)}
{legend(["Squared error between true noise <code>e</code> and predicted noise <code>output</code>, <b>summed</b> over channels and pixels, then averaged over the batch. (Paper Eq. 9.)"])}</div>
<div class="bul">
<ul class="tight">
<li>Plain regression: no discriminator, no adversarial balance. This is the paper's argument for training stability versus GANs.</li>
<li>Every chosen t has equal weight.</li>
<li>Because the loss is a <i>sum</i>, its absolute value scales with image size; compare loss curves only at the same resolution.</li>
</ul></div>
</div>""", src="functions/losses.py; paper Eq. 9")

# 15 training loop (DRAFT)
slide("Training loop", f"""
<div class="cols w">
<div>{code('''
x_img = x['LD'].to(self.device)
x_gt  = x['FD'].to(self.device)
e = torch.randn_like(x_gt)                                       # ⟦1⟧
...  # choose t (previous slides)
loss = loss_registry[config.model.type](model, x_img, x_gt, t, e, b)
optimizer.zero_grad()
loss.backward()                                                  # ⟦2⟧
optimizer.step()
if self.config.model.ema:
    ema_helper.update(model)                                     # ⟦3⟧
if step % self.config.training.snapshot_freq == 0 or step == 1:
    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
    if self.config.model.ema:
        states.append(ema_helper.state_dict())
    torch.save(states, os.path.join(
        self.args.log_path, "ckpt_{}.pth".format(step)))
    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))  # ⟦4⟧
''', None, 755)}</div>
<div>{legend([
 "Fresh random noise every step.",
 "Standard backprop; optimizer is Adam (<code>functions/__init__.py</code>).",
 "Update the moving-average copy of the weights (next slides).",
 "A checkpoint at step 1 and every <code>snapshot_freq</code> steps; <code>ckpt.pth</code> always holds the latest and is what <code>--resume_training</code> reads."])}
<div class="cap">Loop ends only when <code>n_epochs</code> is reached; <code>n_iters</code> in the config is not checked (see “To verify”).</div></div>
</div>""", src="runners/diffusion.py::sg_train", draft=True)

# 16 EMA
slide("EMA: a smoothed copy of the weights", f"""
{eq("θ<sub>EMA</sub> ← (1 − μ) · θ + μ · θ<sub>EMA</sub>, &nbsp;&nbsp; μ = 0.999")}
<div class="cols">
<div>{code('''
def update(self, module):
    ...
    for name, param in module.named_parameters():
        if param.requires_grad:
            self.shadow[name].data = ((1. - self.mu) * param.data
                + self.mu * self.shadow[name].data)          # ⟦1⟧

def ema(self, module):      # copy shadow weights INTO the model
    ...
    param.data.copy_(self.shadow[name].data)                    # ⟦2⟧
''', 15)}</div>
<div>{legend([
 "Each step moves the shadow copy 0.1% toward the current weights, so it averages roughly the last 1,000 updates.",
 "At sampling time <code>ema(model)</code> overwrites the model with the smoothed weights. <b>Generated images come from the EMA weights, not the raw ones.</b>"])}
<div class="cap">Why: single-step weights are noisy; the average gives steadier, usually better samples. The paper does not discuss EMA; it comes with the DDIM codebase.</div></div>
</div>""", src="models/ema.py")

# 17 checkpoints
slide("Saving and loading models", f"""
<div class="cols">
<div>{code('''
# saved (training):
states = [model.state_dict(), optimizer.state_dict(),
          epoch, step, ema_state]

# loaded (sampling):
states = torch.load(
    os.path.join(self.args.log_path, f"ckpt_{ckpt_idx}.pth"),
    map_location=self.config.device)                      # ⟦1⟧
model.load_state_dict(states[0], strict=True)             # ⟦2⟧
ema_helper.load_state_dict(states[-1])                    # ⟦3⟧
ema_helper.ema(model)                                     # ⟦4⟧
''', 15)}</div>
<div>{legend([
 "Path = <code>&lt;exp&gt;/logs/&lt;doc&gt;/ckpt_&lt;N&gt;.pth</code>, with N taken from the list <code>sampling.ckpt_id</code> in the config.",
 "Index 0: the raw model weights.",
 "Last index: the EMA weights.",
 "Overwrite the model with the EMA weights, then sample."])}
<div class="callout">Practical consequence: the authors' single file <code>ckpt_LDFDCT.pth</code> only loads if it is renamed <code>ckpt_1.pth</code> and the config sets <code>ckpt_id: [1]</code>. The <code>--use_pretrained</code> flag does <i>not</i> do this; it targets unrelated CIFAR/LSUN models.</div></div>
</div>""", src="runners/diffusion.py::sg_sample, sg_train")

# 18 DDIM
slide("Sampling: the DDIM step", f"""
{eq("x̂<sub>0</sub> = ( x<sub>t</sub> − √(1−ᾱ<sub>t</sub>) · ε<sub>θ</sub> ) / √ᾱ<sub>t</sub> &nbsp;&nbsp;→&nbsp;&nbsp; x<sub>t′</sub> = √ᾱ<sub>t′</sub> · x̂<sub>0</sub> + √(1−ᾱ<sub>t′</sub>−σ²) · ε<sub>θ</sub> + σ z")}
<div class="cols w">
<div>{code('''
for i, j in zip(reversed(seq), reversed(seq_next)):      # ⟦1⟧
    at      = compute_alpha(b, t.long())
    at_next = compute_alpha(b, next_t.long())            # ⟦2⟧
    xt = xs[-1].to('cuda')
    et = model(torch.cat([x_img, xt], dim=1), t)         # ⟦3⟧
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()       # ⟦4⟧
    c1 = kwargs.get("eta", 0) * (
        (1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()    # σ
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = (at_next.sqrt() * x0_t
               + c1 * torch.randn_like(x) + c2 * et)             # ⟦5⟧
''', None, 755)}</div>
<div>{legend([
 "Walk the chosen time steps from the noisiest to 0 (about 10 iterations). <code>seq_next</code> is the same list shifted by one, ending with −1.",
 "<code>compute_alpha</code> prepends a zero to β, so t = −1 gives ᾱ = 1: the last step lands on a fully clean image.",
 "The same U-Net call as in training: condition stacked with the current noisy image.",
 "Estimate the clean image from the predicted noise.",
 "Re-noise x̂<sub>0</sub> to the next level. <code>eta=0</code> (default): σ = 0, fully <b>deterministic</b>."])}</div>
</div>""", src="functions/denoising.py::sg_generalized_steps")

# 19 evaluation
slide("Evaluation, and our result", f"""
<div class="cols">
<div>{code('''
x_gt = x_gt*255
x = x*255
psnr_temp = calculate_psnr(x[i,:,:], x_gt[i,:,:])              # ⟦1⟧
ssim_temp = ssim(x_gt[i,:,:], x[i,:,:], data_range=255)        # ⟦2⟧
...
logging.info('Average: PSNR {}, SSIM {}, time {}'.format(
    avg_psnr, avg_ssim, avg_time))
''', 14.5)}
{legend(["PSNR = 20·log<sub>10</sub>(255 / √MSE), in dB. Higher is better.", "SSIM from scikit-image: compares local mean, variance and covariance; 1.0 is identical."])}</div>
<div>{table(["LDFDCT, 3,501 test images","PSNR","SSIM"],[
 ["Paper, Table II (Fast-DDPM)","37.5","0.92"],
 ["<b>Ours</b> (pretrained weights, Easley L40S, PyTorch 2.2.2)","<b>37.46</b>","<b>0.9155</b>"],
 ["Paper, DDPM (1000 steps)","35.4","0.87"]])}
<div class="callout" style="margin-top:18px">Our run uses the authors' trained checkpoint, so it validates the pipeline (data, sampler, metrics) but <b>not</b> training. It matched despite a different GPU and PyTorch version.</div>
<div class="cap">Each log line is a batch of 16 images; the final line averages all 3,501.</div></div>
</div>""", src="runners/diffusion.py::sg_sample_fid; functions/losses.py::calculate_psnr")

# 20 config map
slide("Config keys that control behavior", table(["Key (configs/ldfd_linear.yml)","Value","Effect"],[
 ["<code>data.train_dataroot / sample_dataroot</code>","paths","Where images are read. We use absolute scratch paths."],
 ["<code>data.image_size</code>","256","Slices are resized to this; U-Net asserts square input of this size."],
 ["<code>model.type</code>","sg","Selects the loss and the input-channel layout."],
 ["<code>model.in_channels / out_ch</code>","2 / 1","Condition + noisy target in; predicted noise out."],
 ["<code>model.ch, ch_mult, num_res_blocks</code>","128, [1,1,2,2,4,4], 2","U-Net width and depth."],
 ["<code>model.attn_resolutions</code>","[16]","Self-attention only at 16×16 feature maps."],
 ["<code>model.ema_rate</code>","0.999","EMA smoothing μ."],
 ["<code>diffusion.beta_*, num_diffusion_timesteps</code>","1e-4, 0.02, 1000","The underlying noise curve."],
 ["<code>training.batch_size</code>","16","Matches the paper."],
 ["<code>training.snapshot_freq</code>","100000","Checkpoint interval (we use 10 for smoke tests)."],
 ["<code>sampling.ckpt_id</code>","list","Which <code>ckpt_N.pth</code> files to load; we use <code>[1]</code>."],
 ["<code>sampling_fid.batch_size</code>","128 → <b>16</b>","128 runs out of memory on an L40S (44 GB); the paper used an 80 GB A100."],
 ["<code>optim.lr</code>","2e-5","Paper says 2e-4 (see “To verify”)."]], "cfg"),
 src="configs/ldfd_linear.yml; our changes in configs/ldfd_smoke_test.yml")

# 21 sbatch
slide("Running it on the cluster: the job script", f"""
<div class="cols w">
<div>{code('''
#!/bin/bash
#SBATCH --job-name=fastddpm-ldfdct-smoke       # ⟦1⟧
#SBATCH --partition=interactive                 # ⟦2⟧
#SBATCH --gres=gpu:l40s:1                       # ⟦3⟧
#SBATCH --cpus-per-task=8                       # ⟦4⟧
#SBATCH --mem=32G
#SBATCH --time=01:00:00                         # ⟦5⟧
#SBATCH --output=fastddpm-ldfdct-smoke-%j.out   # ⟦6⟧
#SBATCH --error=fastddpm-ldfdct-smoke-%j.err

source ~/.bashrc
conda activate fastddpm                          # ⟦7⟧
cd ~/ECE_551_problems_ML/fast-ddpm               # ⟦8⟧
python fast_ddpm_main.py \\
  --config ldfd_smoke_test.yml --dataset LDFDCT \\
  --exp ~/carc-scratch/fastddpm-exp --doc ldfdct_smoke_test \\
  --sample --fid --scheduler_type uniform --timesteps 10
''', None, 755)}</div>
<div>{legend([
 "Name shown in <code>squeue</code>.",
 "Queue: <code>interactive</code> (4 h cap) for tests; <code>general</code> (2 days) for long runs.",
 "One L40S GPU.",
 "CPU cores for data loading (matches <code>num_workers: 8</code>).",
 "Walltime limit: the scheduler kills the job at this time.",
 "<code>%j</code> = job id. Python's logging and progress bars go to the <b>.err</b> file; .out stays empty.",
 "Activate the environment inside the job.",
 "Configs and data paths are relative to this folder, so the job must start here.",
 ])}</div>
</div>""", src="slurm/ldfdct_smoke_test.sh")

# 22 HPC lessons
slide("What running on HPC taught us", """
<ul class="lessons">
<li><b>GPU memory sets batch size.</b> The paper's sampling batch of 128 (80 GB A100) fails with <i>CUDA out of memory</i> on an L40S; 16 works. Settings do not transfer between GPUs.</li>
<li><b>Software must match the GPU.</b> The paper's PyTorch 1.12.1 has no L40S/H100 support, so we use 2.2.2, which in turn forced <code>numpy&lt;2</code> and <code>opencv-python&lt;4.10</code>.</li>
<li><b>Installs belong on the login node.</b> Installing inside a GPU allocation burned walltime and was killed at the limit.</li>
<li><b>Data lives on scratch, not in git.</b> <code>~/carc-scratch</code>: 7.3 GB dataset bundle, 1.7 GB checkpoint, run outputs.</li>
<li><b>Read the .err file.</b> A failed batch job shows nothing at submit time; the traceback is in <code>.err</code>.</li>
<li><b>Training can delete a folder.</b> A fresh (non-resume) training run removes an existing <code>--doc</code> directory, so training and inference need different <code>--doc</code> names.</li>
<li><b>The README was incomplete.</b> No <code>requirements.txt</code>; five imports (<code>yaml, pandas, lmdb, matplotlib, requests</code>) were undocumented; one version pairing was invalid.</li>
<li><b>Cancel jobs you no longer need.</b> GPUs are shared.</li>
</ul>""")

# 23 verify
slide("To verify", """
<ul class="lessons">
<li><b>Learning rate.</b> Paper: 2×10⁻⁴. Configs: <code>lr: 0.00002</code> (2×10⁻⁵). Which is intended?</li>
<li><b>Uniform scheduler length.</b> <code>arange(−1, 1000, 100)</code> with the first set to 0 gives 11 values, not 10.</li>
<li><b>Iteration cap.</b> <code>n_iters</code> is never checked in the training loop; the paper's 400,000 iterations are not enforced.</li>
<li><b>Validation.</b> <code>validation_freq</code> is effectively disabled and no validation loop was found; only training loss is logged to TensorBoard.</li>
<li><b>Multi-GPU.</b> Training uses <code>DataParallel</code>, not <code>DistributedDataParallel</code>.</li>
<li><b>Training behavior on our stack.</b> Inference matched the paper; whether training from scratch on PyTorch 2.2.2 / L40S matches is <i>empty</i> (not yet run).</li>
<li><b>Textbook references.</b> Chapter and equation numbers: <i>empty</i>.</li>
</ul>""")

# 24 2D -> 3D
slide("From 2D to 3D: what would change", """
<div class="cols">
<div>
<ul class="tight">
<li><b>Layers:</b> 2D convolutions, normalization and attention become 3D (<code>Conv2d → Conv3d</code>); the U-Net data path stays the same.</li>
<li><b>Data:</b> loaders return volumes or sub-volumes (crops), shape <code>B × C × D × H × W</code>, not slices.</li>
<li><b>Memory:</b> activations grow roughly with the number of voxels, about <i>D</i> times a 2D slice. At the same batch size this multiplies memory by <i>D</i>, so batch size or crop size must shrink.</li>
<li><b>Compute:</b> per-step cost also grows about <i>D</i>×; this is why the paper expects 3D diffusion to take “weeks to months.”</li>
<li><b>Paper's proposal:</b> run diffusion in a smaller 3D latent space learned by an autoencoder.</li>
<li><b>Small-scale example:</b> the MLSP 2026 video-diffusion tutorial (Pattichis) trains a 3D U-Net on downsampled 16×32×32 clips.</li>
</ul></div>
<div class="callout">Analysis, not a measured result. Measuring seconds per step and GPU memory at several batch sizes on Easley would turn this into numbers.</div>
</div>""", src="paper Sec. VI-D; reasoning from tensor shapes")

# =====================================================================
CSS = """
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
body { margin:0; font-family: Helvetica, Arial, sans-serif; color:#1b1b1b; }
.slide { width:1280px; height:720px; padding:34px 54px 40px 54px; position:relative; page-break-after:always; overflow:hidden; background:#fff; }
h1 { font-size:42px; margin:0 0 22px 0; font-weight:600; letter-spacing:-0.3px; border-bottom:3px solid #1f5fbf; padding-bottom:8px; }
.title { display:flex; align-items:center; }
.titlebox .big { font-size:62px; font-weight:700; color:#12396f; }
.titlebox .sub { font-size:30px; margin:12px 0 34px 0; color:#333; max-width:980px; line-height:1.3; }
.titlebox .meta { font-size:19px; color:#444; line-height:1.7; }
.titlebox .note { margin-top:40px; font-size:15px; color:#888; }
.cols.w { grid-template-columns: 1.95fr 1fr; }
.cols { display:grid; grid-template-columns: 1.55fr 1fr; gap:30px; align-items:start; }
pre.code { background:#f5f6f8; border:1px solid #d9dde3; border-radius:8px; padding:14px 16px; margin:0 0 12px 0; font-family: Menlo, Consolas, monospace; line-height:1.5; white-space:pre; overflow:hidden; }
.k { color:#a626a4; } .c { color:#6a737d; font-style:italic; }
b.n { display:inline-block; min-width:21px; height:21px; line-height:21px; text-align:center; border-radius:50%; background:#d9480f; color:#fff; font:700 13px Helvetica,Arial,sans-serif; margin:0 2px; font-style:normal; vertical-align:middle; }
ol.legend { list-style:none; padding:0; margin:0; font-size:21px; line-height:1.38; }
ol.legend li { display:flex; gap:11px; margin-bottom:14px; }
ol.legend li b.n { flex:none; margin-top:1px; }
.eq { font-family: 'Times New Roman', Georgia, serif; font-size:30px; background:#eef4ff; border-left:5px solid #1f5fbf; padding:10px 18px; margin:0 0 16px 0; font-style:italic; }
.lead { font-size:27px; margin:4px 0 16px 0; }
table { border-collapse:collapse; font-size:20px; width:100%; margin-bottom:10px; }
th { background:#12396f; color:#fff; text-align:left; padding:7px 10px; font-weight:600; }
td { border-bottom:1px solid #d9dde3; padding:6px 10px; vertical-align:top; }
table.small { font-size:17px; } table.small td, table.small th { padding:4px 9px; }
table.cfg { font-size:17.5px; } table.cfg td { padding:7px 10px; }
code { font-family: Menlo, Consolas, monospace; font-size:0.9em; background:#f0f1f3; padding:1px 4px; border-radius:3px; }
.callout { background:#fff6e5; border-left:5px solid #e8890c; padding:14px 18px; font-size:21px; line-height:1.4; margin-top:14px; }
.cap { font-size:16px; color:#555; margin-top:6px; line-height:1.35; }
ul.tight, ul.lessons, ul.refs { margin:6px 0; padding-left:24px; font-size:23px; line-height:1.42; }
ul.lessons li, ul.refs li { margin-bottom:11px; } ul.tight li { margin-bottom:10px; }
ul.refs { font-size:22px; }
.note2 { margin-top:18px; font-size:15px; color:#777; }
.h2 { font-size:26px; font-weight:600; color:#12396f; margin-bottom:10px; }
.flow { display:flex; align-items:center; gap:6px; }
.flow .arr { font-size:30px; color:#555; }
.box { flex:1; padding:14px 8px; text-align:center; border-radius:8px; font-size:20px; line-height:1.3; border:2px solid; min-height:96px; display:flex; align-items:center; justify-content:center; }
.box.tr { background:#e8f0fd; border-color:#1f5fbf; } .box.dp { background:#fdeee3; border-color:#d9480f; }
.shapes { display:flex; align-items:center; gap:12px; margin-top:6px; font-size:19px; justify-content:space-between; }
.shapes span { font-size:24px; color:#555; }
.sh { border:2px solid #888; border-radius:6px; padding:7px 9px; text-align:center; line-height:1.4; background:#fafafa; }
.sh.hl { border-color:#1f5fbf; background:#e8f0fd; }
.src { position:absolute; left:54px; bottom:12px; font-size:12.5px; color:#777; font-family:Menlo, Consolas, monospace; }
.pg { position:absolute; right:54px; bottom:12px; font-size:12.5px; color:#999; }
.draft { position:absolute; right:54px; top:40px; background:#c92a2a; color:#fff; font-weight:700; padding:4px 12px; border-radius:4px; font-size:14px; letter-spacing:1px; }
svg.plot { width:100%; height:auto; }
"""

parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Fast-DDPM Code Notes</title><style>', CSS, '</style></head><body>']
for n, s in enumerate(slides, 1):
    h = f"<h1>{s['title']}</h1>" if s["title"] else ""
    d = '<div class="draft">DRAFT</div>' if s["draft"] else ""
    src = f'<div class="src">{html.escape(s["src"])}</div>' if s["src"] else ""
    pg = f'<div class="pg">{n}</div>' if n > 1 else ""
    parts.append(f'<section class="slide {s["cls"]}">{d}{h}{s["body"]}{src}{pg}</section>')
parts.append("</body></html>")
SUB = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}
final = "".join(parts)
final = re.sub("[₀-₉]+", lambda m: "<sub>" + "".join(SUB[c] for c in m.group()) + "</sub>", final)
open("slides.html", "w").write(final)
print("slides:", len(slides))
for w in WARN: print("WARN", w)

# Build PDF:
#   python3 build.py
#   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --no-pdf-header-footer \
#       --print-to-pdf=../fast_ddpm_code_notes.pdf slides.html
