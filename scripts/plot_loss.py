#!/usr/bin/env python3
"""Plot training loss from a run's stdout.txt.

Usage: python plot_loss.py <path to stdout.txt> [window=100] [out.png=loss.png]

Raw loss is a sum over pixels, and a batch containing an image at time step 0
adds a large spike (the network cannot predict 1% noise there), so the raw curve
is very jagged. This plots the raw values faintly, plus a moving MEDIAN (robust to
the spikes, tracks the typical loss) and a moving MEAN. If a run was resumed, the
log repeats some step numbers; the last occurrence of each step is kept.
"""
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = os.path.expanduser(sys.argv[1])
window = int(sys.argv[2]) if len(sys.argv) > 2 else 100
out = sys.argv[3] if len(sys.argv) > 3 else "loss.png"

by_step = {}
for line in open(log):
    m = re.search(r"step: (\d+), loss: ([0-9.eE+-]+)", line)
    if m:
        by_step[int(m.group(1))] = float(m.group(2))
steps = sorted(by_step)
loss = [by_step[s] for s in steps]
print(f"{len(steps)} steps, {steps[0]} to {steps[-1]}")

xs, med, mean = [], [], []
stride = max(1, window // 4)
for i in range(window, len(loss) + 1, stride):
    w = loss[i - window:i]
    xs.append(steps[i - 1])
    med.append(statistics.median(w))
    mean.append(sum(w) / len(w))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, loss, color="#bbbbbb", lw=0.5, label="raw")
ax.plot(xs, mean, color="#d9480f", lw=1.8, label=f"moving mean ({window})")
ax.plot(xs, med, color="#1f5fbf", lw=1.8, label=f"moving median ({window})")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("loss (sum over pixels, mean over batch)")
ax.set_title(os.path.basename(os.path.dirname(os.path.abspath(log))))
ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
