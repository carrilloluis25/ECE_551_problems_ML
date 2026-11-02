#!/bin/bash
#SBATCH --job-name=fastddpm-bs-limit
#SBATCH --partition=interactive
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-4%3
#SBATCH --output=fastddpm-bs-limit-%A_%a.out
#SBATCH --error=fastddpm-bs-limit-%A_%a.err

SIZES=(4 8 10 12 14)
B=${SIZES[$SLURM_ARRAY_TASK_ID]}
OUT=$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

CFG=bs_test_${B}.yml
sed "/^training:/,/^sampling:/ s/batch_size: .*/batch_size: ${B}/; /^training:/,/^sampling:/ s/snapshot_freq: .*/snapshot_freq: 100000/" configs/ldfd_train_smoke.yml > configs/$CFG
echo "=== config used for batch size $B ==="
sed -n '/^training:/,/^sampling:/p' configs/$CFG

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > $OUT/bs_mem_${B}.csv &
SMI=$!

timeout 240 python fast_ddpm_main.py \
  --config $CFG --dataset LDFDCT \
  --exp ~/carc-scratch/fastddpm-exp --doc bs_test_${B} \
  --scheduler_type uniform --timesteps 10

kill $SMI

python - "$B" "$OUT" <<'PY'
import re, sys, os, statistics, datetime
b, out = sys.argv[1], sys.argv[2]
log = os.path.expanduser(f"~/carc-scratch/fastddpm-exp/logs/bs_test_{b}/stdout.txt")
txt = open(log).read()
rows = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) - step: (\d+), loss", txt)
t = [(int(s), datetime.datetime.strptime(ts.replace(",", "."), "%Y-%m-%d %H:%M:%S.%f")) for ts, s in rows]
dts = [(t[i][1] - t[i-1][1]).total_seconds() for i in range(1, len(t)) if t[i][0] >= 20]
mem = [int(x) for x in open(f"{out}/bs_mem_{b}.csv").read().split() if x.strip().isdigit()]
oom = "OutOfMemoryError" in txt
med = statistics.median(dts) if dts else None
line = (f"batch={b} status={'OOM' if oom else 'OK'} steps_logged={len(t)} "
        f"median_s_per_step={'n/a' if med is None else f'{med:.3f}'} "
        f"images_per_s={'n/a' if med is None else f'{int(b)/med:.1f}'} "
        f"peak_gpu_mem_MiB={max(mem) if mem else 'n/a'}")
open(f"{out}/bs_result_{b}.txt", "w").write(line + "\n")
print(line)
PY

rm -f ~/carc-scratch/fastddpm-exp/logs/bs_test_${B}/ckpt*.pth
