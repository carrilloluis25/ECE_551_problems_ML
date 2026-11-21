#!/bin/bash
#SBATCH --job-name=fastddpm-scale
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=fastddpm-scale-%j.out
#SBATCH --error=fastddpm-scale-%j.err

# Scaling benchmark: time a FIXED number of training steps on however many GPUs this job was given.
# It runs the existing training code unchanged. That code wraps the model in torch.nn.DataParallel,
# so with 2 or 4 GPUs it already uses all of them (single process; this is the "free" multi-GPU
# baseline that DDP is later compared against).
#
# The number of GPUs comes from the job's --gres line; nothing else changes. Examples:
#   1 H100 :  sbatch --gres=gpu:h100:1 scaling_bench.sh
#   2 H100 :  sbatch scaling_bench.sh                                   (the defaults above)
#   1 L40S :  sbatch --partition=interactive --gres=gpu:l40s:1 scaling_bench.sh
#   2/3 L40S: sbatch --partition=l40s --gres=gpu:l40s:2 scaling_bench.sh   (the l40s partition)
# Per-user caps (sacctmgr, 2026-11-21): 2 H100, 3 L40S, and only 1 GPU on `interactive`, so a
# 4-GPU job cannot run. With 3 GPUs use BATCH=12 or 18 so the batch divides evenly.
# Settings go through the shell, not --export (which holds jobs on the h100 partition):
#   BATCH=16 STEPS=300 sbatch scaling_bench.sh
#   BATCH is the GLOBAL batch. With N GPUs, DataParallel gives each GPU BATCH/N images.
#   Strong scaling: keep BATCH fixed and add GPUs (same recipe, fewer seconds per step).
#   Weak scaling:   BATCH grows with N (e.g. 16 per GPU), a different recipe.
BATCH=${BATCH:-16}
STEPS=${STEPS:-300}
OUT=$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

NGPU=$(nvidia-smi -L | wc -l)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -oE 'L40S|H100|A100' | head -1)
GPU=${GPU:-GPU}
TAG=${GPU}_${NGPU}gpu_bs${BATCH}
echo "=== job $SLURM_JOB_ID | node $(hostname) | $NGPU x $GPU | global batch $BATCH ($((BATCH / NGPU)) per GPU) | $STEPS steps | tag $TAG ==="
echo "=== CPUs for this job: $(nproc) | GPU interconnect (NV# = NVLink, PIX/PHB/SYS = PCIe paths) ==="
nvidia-smi topo -m

# the largest batch the cards can hold is not what we are measuring: stop clearly if BATCH < NGPU
[ "$BATCH" -ge "$NGPU" ] || { echo "ERROR: global batch $BATCH is smaller than the $NGPU GPUs"; exit 1; }

# scratch must accept writes (the final step saves a 1.7 GB checkpoint, deleted at the end)
dd if=/dev/zero of=$HOME/carc-scratch/.quota_probe bs=1M count=100 2>/dev/null && rm -f $HOME/carc-scratch/.quota_probe || { rm -f $HOME/carc-scratch/.quota_probe; echo "ERROR: cannot write to scratch (quota?)"; exit 1; }

CFG=scale_${TAG}.yml
sed "/^training:/,/^sampling:/ s/batch_size: .*/batch_size: ${BATCH}/; /^training:/,/^sampling:/ s/snapshot_freq: .*/snapshot_freq: 100000000/" configs/ldfd_train_smoke.yml > configs/$CFG
sed -n '/^training:/,/^sampling:/p' configs/$CFG

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $OUT/scale_mem_${TAG}.csv &
SMI=$!

python fast_ddpm_main.py \
  --config $CFG --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp --doc scale_${TAG} \
  --scheduler_type uniform --timesteps 10 \
  --max_steps $STEPS --no_first_ckpt

kill $SMI

python - "$BATCH" "$OUT" "$TAG" "$NGPU" <<'PY'
import re, sys, os, statistics, datetime, collections
b, out, tag, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
log = os.path.expanduser(f"~/carc-scratch/fastddpm-exp/logs/scale_{tag}/stdout.txt")
txt = open(log).read()
rows = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) - step: (\d+), loss", txt)
t = [(int(s), datetime.datetime.strptime(ts.replace(",", "."), "%Y-%m-%d %H:%M:%S.%f")) for ts, s in rows]
dts = [(t[i][1] - t[i-1][1]).total_seconds() for i in range(1, len(t)) if t[i][0] >= 30]   # skip warm-up steps
peak = collections.defaultdict(int)
samples = collections.defaultdict(list)      # per GPU: (memory MiB, utilization %)
for line in open(f"{out}/scale_mem_{tag}.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3 and p[1].isdigit() and p[2].isdigit():
        peak[p[0]] = max(peak[p[0]], int(p[1]))
        samples[p[0]].append((int(p[1]), int(p[2])))
# utilization while training: ignore samples taken before the model was loaded or after it exited
util = {}
for k, v in samples.items():
    act = [u for m, u in v if m >= 0.5 * peak[k]]
    util[k] = statistics.median(act) if act else 0
oom = "OutOfMemoryError" in txt
failed = "Traceback" in txt
med = statistics.median(dts) if dts else None
status = "OOM" if oom else ("ERROR" if failed or med is None else "OK")
line = (f"tag={tag} gpus={n} global_batch={b} per_gpu_batch={int(b)//int(n)} status={status} steps_logged={len(t)} "
        f"median_s_per_step={'n/a' if med is None else f'{med:.3f}'} "
        f"images_per_s={'n/a' if med is None else f'{int(b)/med:.1f}'} "
        f"peak_mem_MiB_per_gpu={','.join(f'{k}:{v}' for k, v in sorted(peak.items()))} "
        f"median_util_pct_per_gpu={','.join(f'{k}:{v:g}' for k, v in sorted(util.items()))}")
open(f"{out}/scale_result_{tag}.txt", "w").write(line + "\n")
print(line)
PY

# the run saves a final checkpoint (--max_steps); it is not needed
rm -f ~/carc-scratch/fastddpm-exp/logs/scale_${TAG}/ckpt*.pth
