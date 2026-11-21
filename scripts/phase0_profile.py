#!/usr/bin/env python
"""Phase 0 profile of Fast-DDPM LDFDCT training: where does one step's time go?

It rebuilds the loop of Diffusion.sg_train (runners/diffusion.py) with the same model, loss,
optimizer, EMA, data loader and DataParallel wrapper, and changes nothing in the model or the
training code. Run it from the fast-ddpm folder, on a GPU node:

    cd ~/ECE_551_problems_ML/fast-ddpm
    python ../scripts/phase0_profile.py --batch 16 --steps 60

Five measurements, in this order:
  1. model size: parameter count in total and per top-level block, and the checkpoint size it implies
  2. data loader alone: images/s the loader can deliver with no GPU work (is it a bottleneck?)
  3. natural pass: step time exactly as the real loop runs it (no extra synchronization)
  4. phase pass: the same step with a CUDA synchronize between phases, so each phase gets its own
     time (the sum is a little longer than the natural step: the synchronizes remove overlap)
  5. torch.profiler: the CUDA and CPU operations that take the most time
plus the true peak GPU memory (torch.cuda.max_memory_allocated).
"""
import argparse
import collections
import os
import statistics
import sys
import time
import types

sys.path.insert(0, os.getcwd())          # the fork's modules (models, functions, datasets, runners)

import torch
import torch.utils.data as data
import yaml


def to_ns(obj):
    """dict -> attribute-style namespace, like the fork's dict2namespace."""
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    return obj


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ldfd_train_smoke.yml")
    ap.add_argument("--batch", type=int, default=16, help="global batch (the paper uses 16)")
    ap.add_argument("--steps", type=int, default=60, help="measured steps per pass")
    ap.add_argument("--warmup", type=int, default=15, help="steps ignored at the start of each pass")
    ap.add_argument("--timesteps", type=int, default=10)
    args = ap.parse_args()

    from models.diffusion import Model
    from models.ema import EMAHelper
    from functions import get_optimizer
    from functions.losses import loss_registry
    from datasets.LDFDCT import LDFDCT
    from runners.diffusion import get_beta_schedule

    cfg = to_ns(yaml.safe_load(open(args.config)))
    cfg.training.batch_size = args.batch
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True            # fast_ddpm_main.py sets this too
    gpu = torch.cuda.get_device_name(0)
    print(f"=== {gpu} | torch {torch.__version__} | global batch {args.batch} | {args.steps} steps (+{args.warmup} warm-up) per pass ===")

    # ---- diffusion constants, as in Diffusion.__init__ -------------------------------------------
    betas = torch.from_numpy(get_beta_schedule(
        beta_schedule=cfg.diffusion.beta_schedule, beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end, num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps,
    )).float().to(dev)
    num_timesteps = betas.shape[0]

    # ---- data, as in sg_train --------------------------------------------------------------------
    dataset = LDFDCT(cfg.data.train_dataroot, cfg.data.image_size, split="train")

    def make_loader():
        return data.DataLoader(dataset, batch_size=args.batch, shuffle=True,
                               num_workers=cfg.data.num_workers, pin_memory=True)

    # ---- 2. data loader alone --------------------------------------------------------------------
    it = iter(make_loader())
    for _ in range(5):
        next(it)
    t0 = time.perf_counter()
    nb = 40
    for _ in range(nb):
        next(it)
    dl_s = (time.perf_counter() - t0) / nb
    print(f"[loader alone] {dl_s * 1000:.1f} ms per batch of {args.batch} = {args.batch / dl_s:.0f} images/s "
          f"({cfg.data.num_workers} workers, {len(dataset)} training images)")
    del it

    # ---- 1. model size ---------------------------------------------------------------------------
    model = Model(cfg).to(dev)
    model = torch.nn.DataParallel(model)
    base = model.module
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"[model] parameters: {total:,} total, {trainable:,} trainable")
    named = 0
    for name, child in base.named_children():
        n = sum(p.numel() for p in child.parameters())
        named += n
        if n:
            print(f"[model]   {name:<12} {n:>14,}  {100 * n / total:5.1f}%")
    if total - named:
        print(f"[model]   (direct parameters) {total - named:,}")
    print(f"[model] checkpoint implied by 4 fp32 copies (weights, 2 Adam moments, EMA): {16 * total / 2**30:.3f} GiB "
          f"(the saved checkpoint files are 1.736 GiB)")

    optimizer = get_optimizer(cfg, model.parameters())
    ema = EMAHelper(mu=cfg.model.ema_rate)
    ema.register(model)

    skip = num_timesteps // args.timesteps
    t_intervals = torch.arange(-1, num_timesteps, skip)
    t_intervals[0] = 0
    clip_state = {"ran": 0, "raised": 0}

    def sample_t(n):                                   # antithetic sampling, as in sg_train
        idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
        idx = torch.cat([idx_1, len(t_intervals) - idx_1 - 1], dim=0)[:n]
        return t_intervals[idx].to(dev)

    def clip():
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            clip_state["ran"] += 1
        except Exception:                              # sg_train does exactly this: `except Exception: pass`
            clip_state["raised"] += 1

    # ---- 3. natural pass: the real loop, no extra synchronization --------------------------------
    torch.cuda.reset_peak_memory_stats()
    loader = make_loader()
    it = iter(loader)
    step_t, wait_t = [], []
    prev = None
    for i in range(args.steps + args.warmup):
        tw = time.perf_counter()
        x = next(it)
        wait = time.perf_counter() - tw
        model.train()
        x_img = x["LD"].to(dev)
        x_gt = x["FD"].to(dev)
        e = torch.randn_like(x_gt)
        t = sample_t(x_img.size(0))
        loss = loss_registry[cfg.model.type](model, x_img, x_gt, t, e, betas)
        loss.item()                                    # the real loop's logging.info(...loss.item()) syncs here
        optimizer.zero_grad()
        loss.backward()
        clip()
        optimizer.step()
        ema.update(model)
        now = time.perf_counter()
        if prev is not None and i >= args.warmup:
            step_t.append(now - prev)
            wait_t.append(wait)
        prev = now
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_res = torch.cuda.max_memory_reserved() / 2**20
    nat = med(step_t)
    print(f"[natural] median {nat:.4f} s/step = {args.batch / nat:.1f} images/s; "
          f"waiting for data: median {1000 * med(wait_t):.1f} ms/step ({100 * med(wait_t) / nat:.1f}% of the step)")
    print(f"[memory] peak allocated {peak_alloc:,.0f} MiB ({peak_alloc / 1024:.1f} GiB); "
          f"peak reserved by the caching allocator {peak_res:,.0f} MiB")
    ms = torch.cuda.memory_stats()
    print(f"[memory] allocator retries (cudaMalloc failed, cache flushed, tried again): {ms.get('num_alloc_retries', 0)}; "
          f"cudaMalloc calls: {ms.get('segment.all.allocated', 'n/a')}  (many retries mean the card is too full)")
    print(f"[clip] grad clipping ran {clip_state['ran']} times and raised (silently skipped) {clip_state['raised']} times "
          f"(config.optim has grad_clip: {hasattr(cfg.optim, 'grad_clip')})")

    # ---- 4. phase pass: synchronize between phases -----------------------------------------------
    phases = collections.OrderedDict((k, []) for k in
                                     ["data_wait", "host_to_device", "noise_and_t", "forward_and_loss",
                                      "backward", "grad_clip", "optimizer_step", "ema_update", "loss_item_and_log"])
    it = iter(make_loader())

    def sync_now():
        torch.cuda.synchronize()
        return time.perf_counter()

    for i in range(args.steps + args.warmup):
        mark = [sync_now()]
        x = next(it)
        mark.append(time.perf_counter())
        model.train()
        x_img = x["LD"].to(dev)
        x_gt = x["FD"].to(dev)
        mark.append(sync_now())
        e = torch.randn_like(x_gt)
        t = sample_t(x_img.size(0))
        mark.append(sync_now())
        loss = loss_registry[cfg.model.type](model, x_img, x_gt, t, e, betas)
        mark.append(sync_now())
        optimizer.zero_grad()
        loss.backward()
        mark.append(sync_now())
        clip()
        mark.append(sync_now())
        optimizer.step()
        mark.append(sync_now())
        ema.update(model)
        mark.append(sync_now())
        loss.item()
        mark.append(sync_now())
        if i >= args.warmup:
            for (k, lst), a, b in zip(phases.items(), mark[:-1], mark[1:]):
                lst.append(b - a)
    tot = sum(med(v) for v in phases.values())
    print(f"[phases] median per phase, with a synchronize between phases (sum {tot:.4f} s, natural step {nat:.4f} s):")
    for k, v in phases.items():
        print(f"[phases]   {k:<18} {1000 * med(v):8.1f} ms  {100 * med(v) / tot:5.1f}%")

    # ---- 5. torch.profiler -----------------------------------------------------------------------
    from torch.profiler import ProfilerActivity, profile
    it = iter(make_loader())
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            x = next(it)
            x_img = x["LD"].to(dev)
            x_gt = x["FD"].to(dev)
            e = torch.randn_like(x_gt)
            t = sample_t(x_img.size(0))
            loss = loss_registry[cfg.model.type](model, x_img, x_gt, t, e, betas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
        torch.cuda.synchronize()
    print("[profiler] top operations by CUDA time (5 steps):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=60))
    print("[profiler] top operations by self CPU time (5 steps): Python and launch overhead shows up here")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10, max_name_column_width=60))


if __name__ == "__main__":
    main()
