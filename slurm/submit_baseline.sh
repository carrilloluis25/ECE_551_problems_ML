#!/bin/bash
# Queue the whole baseline: training chunks chained one after another, plus a PSNR/SSIM evaluation
# after each chunk. Run from the slurm folder (not with sbatch):
#   DRY=1 bash submit_baseline.sh      # print what would be submitted, submit nothing
#   bash submit_baseline.sh            # submit for real
# Variables (defaults): TOTAL=500000 CHUNK=125000 EVERY=25000 DOC=baseline_bs16_lr2e-5 NODELIST=easley053
#   NODELIST= (empty) removes the node pin; easley053 is the node where H100 tasks were not held.
#
# Chaining rules:
#   training chunk i+1  afterok  chunk i     a failed or timed-out chunk stops the chain and shows
#                                            DependencyNeverSatisfied; fix it and rerun that chunk by hand
#   eval of chunk i     afterok  chunk i     scores every EVERY steps on the stride-10 subset (~350 images)
#   eval after eval     afterany             evals share <DOC> output folders, so they never overlap
#   final eval                               the last checkpoint on the full 3,501-image test set
TOTAL=${TOTAL:-500000}
CHUNK=${CHUNK:-125000}
EVERY=${EVERY:-25000}
DOC=${DOC:-baseline_bs16_lr2e-5}
NODELIST=${NODELIST-easley053}
NODE_ARG=(); [ -n "$NODELIST" ] && NODE_ARG=(--nodelist="$NODELIST")
[ -n "$EXCLUDE" ] && NODE_ARG+=(--exclude="$EXCLUDE")   # e.g. EXCLUDE=easley052,easley053 to avoid specific nodes

submit() {   # prints the job id
  if [ -n "$DRY" ]; then echo "[dry] ${ENVNOTE:+($ENVNOTE) }sbatch $*" >&2; echo "J$((RANDOM % 900 + 100))"; else sbatch --parsable "$@"; fi
}

N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
PREV_TRAIN=""; PREV_EVAL=""
for ((i = 1; i <= N; i++)); do
  LO=$(( (i - 1) * CHUNK )); HI=$(( i * CHUNK )); [ "$HI" -gt "$TOTAL" ] && HI=$TOTAL
  DEP=(); [ -n "$PREV_TRAIN" ] && DEP=(--dependency=afterok:$PREV_TRAIN)
  # No --export here: on the H100 nodes, a job submitted with --export=ALL,VAR=... was held with
  # "user env retrieval failed". The variables go through the environment instead (default --export=ALL).
  TRAIN=$(ENVNOTE="env TOTAL=$TOTAL CHUNK=$CHUNK DOC=$DOC" TOTAL=$TOTAL CHUNK=$CHUNK DOC=$DOC submit "${NODE_ARG[@]}" "${DEP[@]}" baseline_train.sh)
  echo "chunk $i: steps $LO -> $HI   train job $TRAIN"

  # the chunk's last step is always saved (max_steps), even when it is not a multiple of EVERY
  CK=$( { [ $(( LO + EVERY )) -le "$HI" ] && seq $(( LO + EVERY )) "$EVERY" "$HI"; [ $(( HI % EVERY )) -ne 0 ] && echo "$HI"; } | tr '\n' ' ' | sed 's/ $//')
  EDEP=(--dependency=afterok:$TRAIN); [ -n "$PREV_EVAL" ] && EDEP=(--dependency=afterok:$TRAIN,afterany:$PREV_EVAL)
  EVAL=$(submit "${EDEP[@]}" --export="ALL,DOC=$DOC,CKPTS=$CK,STRIDE=10" eval_checkpoints.sh)
  echo "         eval job $EVAL   checkpoints: $CK"
  PREV_TRAIN=$TRAIN; PREV_EVAL=$EVAL
done

FINAL=$(submit --dependency=afterok:$PREV_TRAIN,afterany:$PREV_EVAL --export="ALL,DOC=$DOC,CKPTS=$TOTAL,STRIDE=1" eval_checkpoints.sh)
echo "final: full test set at step $TOTAL   eval job $FINAL   (results: eval_$DOC.csv, stride column 1)"
