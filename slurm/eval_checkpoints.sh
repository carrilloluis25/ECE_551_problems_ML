#!/bin/bash
#SBATCH --job-name=fastddpm-eval
#SBATCH --partition=interactive
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fastddpm-eval-%j.out
#SBATCH --error=fastddpm-eval-%j.err

# Score checkpoints from a training run on the LDFDCT test set (PSNR / SSIM).
# Run from the slurm folder:
#   sbatch --export=ALL,DOC=<training --doc name>,CKPTS="<step> <step>",STRIDE=10 eval_checkpoints.sh
#   DOC    : the --doc used for training (checkpoints are <exp>/logs/<DOC>/ckpt_<step>.pth)
#   CKPTS  : checkpoint steps to evaluate, space separated
#   STRIDE : keep every STRIDE-th slice (10 = ~350 images, fast); 1 = full 3,501-image test set
# Result: eval_<DOC>.csv in the folder you submitted from.
DOC=${DOC:?set DOC to the training run name}
CKPTS=${CKPTS:?set CKPTS to checkpoint steps, e.g. "100 200"}
STRIDE=${STRIDE:-10}
OUT=$SLURM_SUBMIT_DIR
EXP=~/carc-scratch/fastddpm-exp
DATA=~/carc-scratch/fastddpm-data/data

source ~/.bashrc
conda activate fastddpm
cd ~/ECE_551_problems_ML/fast-ddpm

if [ "$STRIDE" -gt 1 ]; then
  # Named LDSUB_*, NOT LD_FD_CT_test_*: the loader globs "<path>**/**/*ld.png", so a
  # sibling folder starting with LD_FD_CT_test would be picked up by the full test set.
  TESTDIR=$DATA/LDSUB_test_s${STRIDE}
  python - "$DATA/LD_FD_CT_test" "$TESTDIR" "$STRIDE" <<'PY'
import os, re, sys
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = 0
for root, _, files in os.walk(src):
    for f in files:
        m = re.search(r"_(\d+)_(ld|fd)\.png$", f)
        if m and int(m.group(1)) % k == 0:
            rel = os.path.relpath(root, src)
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            link = os.path.join(dst, rel, f)
            if not os.path.exists(link):
                os.symlink(os.path.join(root, f), link)
            n += 1
print("subset: linked", n, "files (ld + fd) into", dst)
PY
else
  TESTDIR=$DATA/LD_FD_CT_test
fi

CSV=$OUT/eval_${DOC}.csv
[ -f "$CSV" ] || echo "ckpt,psnr,ssim,s_per_batch,stride" > "$CSV"

for N in $CKPTS; do
  if [ ! -e "$EXP/logs/$DOC/ckpt_${N}.pth" ]; then
    echo "SKIP ${N}: $EXP/logs/$DOC/ckpt_${N}.pth does not exist"
    continue
  fi
  CFG=eval_${DOC}_${N}.yml
  sed "s#sample_dataroot: .*#sample_dataroot: \"$TESTDIR\"#; s/ckpt_id: .*/ckpt_id: [${N}]/" configs/ldfd_smoke_test.yml > configs/$CFG
  python fast_ddpm_main.py --config $CFG --dataset LDFDCT \
    --exp $EXP --doc $DOC --sample --fid \
    --scheduler_type uniform --timesteps 10 2>&1 | tee $OUT/eval_${DOC}_${N}.log
  LINE=$(grep "Average: PSNR" $OUT/eval_${DOC}_${N}.log | tail -1 | sed -E 's/.*Average: PSNR ([0-9.]+), SSIM ([0-9.]+), time ([0-9.]+).*/\1,\2,\3/')
  # a failed evaluation must leave a visibly failed row (NA), not a short row with missing columns
  [ -n "$LINE" ] || LINE="NA,NA,NA"
  echo "${N},${LINE},${STRIDE}" >> "$CSV"
  rm -rf $EXP/image_samples/$DOC      # generated images only; checkpoints are untouched
done

echo "=== $CSV ==="
cat "$CSV"
