#!/usr/bin/env bash
# Production slot 1. Run from the PROJECT ROOT:  bash src/run_slot1.sh
#
# Steps 1-4 run in the FOREGROUND and stop at a hard confirmation gate. Nothing
# long starts until you type the confirmation. After the gate, steps 6-8 run
# under nohup so an SSH drop does not kill the run.
#
# The gate exists because step 4 is the only place the time budget can be
# checked against the actual card, and that answer is worth 30 minutes at the
# start rather than a truncated run discovered at hour 19.

set -Eeuo pipefail

RUN_ID="${RUN_ID:-slot1}"
CONFIG="${CONFIG:-prod_300m}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
SLOT_HOURS="${SLOT_HOURS:-20}"
SMOKE_MINUTES="${SMOKE_MINUTES:-20}"
EPOCHS="${EPOCHS:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
CKPT_EVERY_MIN="${CKPT_EVERY_MIN:-30}"
MILESTONE_HOURS="${MILESTONE_HOURS:-2}"
KEEP_LAST="${KEEP_LAST:-2}"
RESERVE_GB="${RESERVE_GB:-5}"
EXPECTED_TRAIN_HOURS="${EXPECTED_TRAIN_HOURS:-12}"
PY="${PY:-python}"

REPORT_DIR="runs/${RUN_ID}/preflight"
mkdir -p "$REPORT_DIR" "$CKPT_DIR"

hr() { printf '=%.0s' {1..70}; echo; }
step() { hr; echo "STEP $1 -- $2"; hr; }
fail() { echo; echo "FAILED at step $1: $2"; exit 1; }

trap 'echo; echo "ABORTED (exit $?) -- nothing further will start."; exit 1' ERR

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

# ---------------------------------------------------------------- 1
step 1 "ENVIRONMENT REPORT"
$PY src/slot1.py env --out "$REPORT_DIR/env.json" \
  || fail 1 "environment report / OLM API check"

echo
echo "--- disk space vs checkpoint retention ---"
DISK_RC=0
$PY src/slot1.py disk --config "$CONFIG" --checkpoint-dir "$CKPT_DIR" \
  --train-hours "$EXPECTED_TRAIN_HOURS" --milestone-hours "$MILESTONE_HOURS" \
  --keep-last "$KEEP_LAST" --reserve-gb "$RESERVE_GB" --strict \
  --out "$REPORT_DIR/disk.json" || DISK_RC=$?

if [ "$DISK_RC" -ne 0 ]; then
  echo
  echo "  *** CHECKPOINT RETENTION DOES NOT FIT FREE DISK ***"
  echo "  A policy that would fit is printed above. Re-run with, e.g.:"
  echo "     KEEP_LAST=1 MILESTONE_HOURS=4 bash src/run_slot1.sh"
  echo "  or point CKPT_DIR at a larger volume:"
  echo "     CKPT_DIR=/mnt/big/checkpoints bash src/run_slot1.sh"
  echo
  read -r -p "Type CONTINUE to proceed anyway (anything else aborts): " D
  [ "$D" = "CONTINUE" ] || { echo "Aborted at the disk check. Nothing started."; exit 1; }
  echo "  proceeding despite insufficient disk, on your confirmation"
fi

# ---------------------------------------------------------------- 2
step 2 "TOKENIZER + SHARD VERIFICATION"
$PY src/slot1.py verify || fail 2 "tokenizer hash or shard manifest mismatch"

# ---------------------------------------------------------------- 3
step 3 "BATCH SIZE PROBE ($CONFIG)"
$PY src/slot1.py probe --config "$CONFIG" \
  --out "$REPORT_DIR/probe.json" || fail 3 "batch probe"

BATCH=$($PY -c "import json;print(json.load(open('$REPORT_DIR/probe.json'))['recommended_batch'])")
echo
echo "  -> using batch size $BATCH (recommended, one step below max for headroom)"

# ---------------------------------------------------------------- 4
step 4 "SMOKE TEST (${SMOKE_MINUTES} min) + WALL-CLOCK EXTRAPOLATION"
$PY src/slot1.py smoke --config "$CONFIG" --batch-size "$BATCH" \
  --minutes "$SMOKE_MINUTES" --epochs "$EPOCHS" --slot-hours "$SLOT_HOURS" \
  --out "$REPORT_DIR/smoke.json" || fail 4 "smoke test"

FITS=$($PY -c "import json;print(json.load(open('$REPORT_DIR/smoke.json'))['fits'])")
TPS=$($PY -c "import json;print(round(json.load(open('$REPORT_DIR/smoke.json'))['tokens_per_sec']))")
HOURS=$($PY -c "import json;print(round(json.load(open('$REPORT_DIR/smoke.json'))['pretrain_hours_extrapolated'],2))")

# steps/2h, from measured throughput -- checkpoint cadence follows the card,
# not a guess about how fast it is
CKPT_EVERY=$($PY -c "
import json
s=json.load(open('$REPORT_DIR/smoke.json'))
tok_per_step=$BATCH*s['context']*$GRAD_ACCUM
sec_per_step=tok_per_step/s['tokens_per_sec']
print(max(int(($CKPT_EVERY_MIN*60)/sec_per_step),1))
")

# ---------------------------------------------------------------- 5
step 5 "HARD STOP -- CONFIRMATION REQUIRED"
echo "  GPU throughput      : ${TPS} tokens/sec"
echo "  Pretrain extrapolated: ${HOURS} h"
echo "  Fits in slot        : ${FITS}"
echo "  Batch size          : ${BATCH} x ${GRAD_ACCUM} accum"
echo "  Checkpoint every    : ${CKPT_EVERY} steps (~${CKPT_EVERY_MIN} min)"
echo "  Checkpoint dir      : ${CKPT_DIR}"
echo
if [ "$FITS" != "True" ]; then
  echo "  *** THE RUN DOES NOT FIT. See the token count that would fit, above. ***"
  echo "  *** Re-run with fewer --epochs, or accept a shorter pretrain.        ***"
  echo
fi
read -r -p "Type CONFIRM to start pretraining (anything else aborts): " ANSWER
[ "$ANSWER" = "CONFIRM" ] || { echo "Aborted by user. Nothing started."; exit 0; }

# ---------------------------------------------------------------- 6-8
step 6 "PRETRAIN -> FINETUNE -> EVALUATE (nohup, backgrounded)"
LOG="runs/${RUN_ID}/slot1.log"
mkdir -p "runs/${RUN_ID}"

nohup bash -c "
set -Eeuo pipefail
export PYTHONPATH=src

echo '--- PRETRAIN ---'
$PY src/train.py pretrain --config $CONFIG --run-id ${RUN_ID}-pretrain \\
  --batch-size $BATCH --grad-accum $GRAD_ACCUM --epochs $EPOCHS \\
  --checkpoint-dir $CKPT_DIR --ckpt-every $CKPT_EVERY \\
  --keep-last $KEEP_LAST --milestone-hours $MILESTONE_HOURS

LATEST=\$($PY -c \"
from pathlib import Path
import sys; sys.path.insert(0,'src')
from train import find_latest_checkpoint
print(find_latest_checkpoint(Path('$CKPT_DIR/${RUN_ID}-pretrain')))
\")
echo \"--- FINETUNE (init from \$LATEST) ---\"
$PY src/train.py finetune --config $CONFIG --run-id ${RUN_ID}-finetune \\
  --init-from \"\$LATEST\" --batch-size $BATCH --grad-accum 4 --epochs 3 \\
  --checkpoint-dir $CKPT_DIR --ckpt-every 200 --keep-last $KEEP_LAST \\
  --final-model $CKPT_DIR/final_model.pt

echo '--- FINAL MODEL (weights only -- download this first) ---'
ls -lh $CKPT_DIR/final_model.pt

FT=\$($PY -c \"
from pathlib import Path
import sys; sys.path.insert(0,'src')
from train import find_latest_checkpoint
print(find_latest_checkpoint(Path('$CKPT_DIR/${RUN_ID}-finetune')))
\")
echo \"--- EVALUATE (\$FT) ---\"
$PY src/evaluate.py --ckpt \"\$FT\" --config $CONFIG --split test \\
  --out runs/${RUN_ID}/eval_test.json

echo '--- SLOT 1 COMPLETE ---'
" > "$LOG" 2>&1 &

echo "  started, pid $!"
echo "  log: $LOG"
echo "  follow with:  tail -f $LOG"
