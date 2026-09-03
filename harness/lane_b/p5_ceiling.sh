#!/usr/bin/env bash
# P5 helper (canonical copy: harness/lane_b/p5_ceiling.sh).
# usage: p5_ceiling.sh <export_onnx run> <model_step.pt> <tag>  -> label_tokens (all steps) + oracle_b (20 rollouts)
# The run folder is taken from the "^[bakeoff] run folder" line only (bakeoff also echoes plan/ORCHESTRATOR_NOTES.md, which may contain the words "run folder").
set -u
EXP="$1"; CK="$2"; TAG="$3"; cd ~/code/franka-sonic
LOGDIR=${P5_LOGDIR:-/tmp/franka-sonic/lane_b}; mkdir -p "$LOGDIR"; LOG=$LOGDIR/p5_ceiling_${TAG}.log
FULL=$LOGDIR/p5_ceiling_${TAG}.full.log
{ echo "=== label_tokens $(date -u +%H:%M:%S) onnx=$EXP ckpt=$CK"; python3 harness/bakeoff.py run lane_b label_tokens --onnx "$EXP" --checkpoint "$CK" > "$FULL" 2>&1; grep -E "run folder|CUDA_VISIBLE|VERDICT|OK rc|FAILED" "$FULL" | tail -6; } > "$LOG" 2>&1
TOK=$(grep -oE "^\[bakeoff\] run folder \S+" "$FULL" | head -1 | awk '{print $3}')
if [ -z "$TOK" ] || [ ! -f "$TOK/out/tokens/index.json" ]; then echo "$TAG: LABEL_TOKENS FAILED (see $LOG)" | tee -a "$LOG"; exit 1; fi
FULL2=$LOGDIR/p5_ceiling_${TAG}.oracle.full.log
{ echo "=== oracle_b $(date -u +%H:%M:%S) tokens=$TOK"; python3 harness/bakeoff.py run lane_b oracle_b --onnx "$EXP" --tokens "$TOK" --rollouts 20 > "$FULL2" 2>&1; grep -E "run folder|CUDA_VISIBLE|OK rc|FAILED" "$FULL2" | tail -4; echo "=== done $(date -u +%H:%M:%S)"; } >> "$LOG" 2>&1
ORC=$(grep -oE "^\[bakeoff\] run folder \S+" "$FULL2" | head -1 | awk '{print $3}')
~/Isaac-GR00T/.venv/bin/python - "$ORC/out/eval/eval_results.csv" "$TAG" "$TOK" "$ORC" <<'PY' 2>&1 | tee -a "$LOG"
import csv,sys
try:
    r=list(csv.DictReader(open(sys.argv[1])))
    succ=sum(1 for x in r if str(x.get("success","")).strip().lower()=="true")
    prog=[float(x.get("progress",0) or 0) for x in r]
    print(f"{sys.argv[2]}: B-ORACLE {succ}/{len(r)} successes, progress mean {sum(prog)/max(1,len(prog)):.3f}; tokens {sys.argv[3]} oracle {sys.argv[4]}")
    print("   columns:", list(r[0].keys())[:12] if r else None)
except Exception as e:
    print(f"{sys.argv[2]}: could not read {sys.argv[1]}: {e}")
PY
