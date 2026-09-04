#!/usr/bin/env bash
# P8 WP 8.2 ceiling test (canonical copy: harness/lane_b/p8_ceiling.sh).
#
#   p8_ceiling.sh <model_step.pt|last.pt> <tag>
#
# One checkpoint -> export_onnx -> decoder_replay (+ MuJoCo flange FK, record only) ->
# label_tokens on the FIRST 20 dataset episodes (--max-episodes 20, no `dataset` sub-step so
# the gate can never mistake a 20-episode gr00t_v2_sonic for the real one) -> oracle_b on those
# same 20 episodes. Prints ONE summary line and appends it to $P8_SERIES.
#
# Round-1 finding that this encodes: `decoder_replay` is NOT a valid selector (0.041-0.047 rad
# spanned 1/20 .. 19/20); only the 20-rollout ceiling test ranks decoders. The replay number is
# recorded, never used to choose.
set -u
CK="$1"; TAG="$2"
cd "$HOME/code/franka-sonic"
ML="${P8_ML:-$HOME/runs/franka-sonic/lane_b/2026-09-04_motion_lib/out/motions}"
DS="${P8_DS:-$HOME/runs/franka-sonic/shared/2026-09-04_dataset-2}"
DEMOS="${P8_DEMOS:-$HOME/runs/franka-sonic/shared/2026-09-04_demos}"
N_EP="${P8_N_EPISODES:-20}"
LOGDIR="${P8_LOGDIR:-/tmp/franka-sonic/lane_b/p8}"; mkdir -p "$LOGDIR"
SERIES="${P8_SERIES:-$LOGDIR/series.txt}"
LOG="$LOGDIR/ceil_${TAG}.log"
PYS=(env PYTHONUSERBASE="$HOME/env/pyuser-sonic" /isaac-sim/python.sh)
runfolder() { grep -aoE "^\[bakeoff\] run folder \S+" "$1" | head -1 | awk '{print $4}'; }

echo "=== $TAG $(date -u +%H:%M:%S) ckpt=$CK" > "$LOG"

# 1 ------------------------------------------------------------------ export
F1="$LOGDIR/ceil_${TAG}.export.log"
python3 harness/bakeoff.py run lane_b export_onnx --checkpoint "$CK" > "$F1" 2>&1
EXP=$(runfolder "$F1")
if [ -z "${EXP:-}" ] || [ ! -f "$EXP/out/model_decoder.onnx" ]; then
  echo "$TAG: EXPORT FAILED (see $F1)" | tee -a "$LOG" "$SERIES"; exit 1
fi
echo "export $EXP" >> "$LOG"

# 2 ------------------------------------------------- replay + flange FK (record only)
F2="$LOGDIR/ceil_${TAG}.replay.log"
python3 harness/bakeoff.py run lane_b decoder_replay --onnx "$EXP" --checkpoint "$CK" \
    --motions "$ML" > "$F2" 2>&1
REP=$(runfolder "$F2")
REPLAY="n/a"; FLANGE="n/a"
if [ -n "${REP:-}" ] && [ -f "$REP/out/replay.json" ]; then
  REPLAY=$("$HOME/Isaac-GR00T/.venv/bin/python" -c \
    "import json,sys;d=json.load(open(sys.argv[1]));print(f\"{d['mean_joint_error_rad']:.3f}\")" \
    "$REP/out/replay.json")
  FLANGE=$("${PYS[@]}" harness/lane_b/fk_flange_error.py "$REP" 2>/dev/null | tail -1)
fi
echo "replay $REP  mean $REPLAY rad" >> "$LOG"
echo "flange $FLANGE" >> "$LOG"

# 3 ------------------------------------------------------ token labels, first N episodes
F3="$LOGDIR/ceil_${TAG}.label.log"
python3 harness/bakeoff.py run lane_b label_tokens --onnx "$EXP" --checkpoint "$CK" \
    --motions "$ML" --dataset "$DS" --max-episodes "$N_EP" \
    --steps validate,obs,encode,check > "$F3" 2>&1
TOK=$(runfolder "$F3")
if [ -z "${TOK:-}" ] || [ ! -f "$TOK/out/tokens/index.json" ]; then
  echo "$TAG: LABEL_TOKENS FAILED (see $F3)" | tee -a "$LOG" "$SERIES"; exit 1
fi
echo "tokens $TOK" >> "$LOG"
# `check` is an EXPORT verification, not a token check (P8 2026-09-04 21:00): it is the only
# clause that compares the exported ONNX against the env policy's own actions, and export_onnx
# is not deterministic — two exports of one checkpoint have produced different g1-encoder
# weights. A ceiling number measured on an unverified export is not the checkpoint's number.
if ! grep -aq "VERDICT: OK" "$F3"; then
  echo "$TAG: CHECK MISMATCH — export not verified against the env policy, ceiling number not reported (see $F3)" \
      | tee -a "$LOG" "$SERIES"
  exit 1
fi

# 4 ------------------------------------------------------------------ B-oracle
F4="$LOGDIR/ceil_${TAG}.oracle.log"
python3 harness/bakeoff.py run lane_b oracle_b --onnx "$EXP" --tokens "$TOK" \
    --demos "$DEMOS" --rollouts "$N_EP" > "$F4" 2>&1
ORC=$(runfolder "$F4")
echo "oracle $ORC" >> "$LOG"

"$HOME/Isaac-GR00T/.venv/bin/python" - "$ORC" "$TAG" "$CK" "$REPLAY" "$FLANGE" "$EXP" "$TOK" <<'PY' | tee -a "$LOG" "$SERIES"
import csv, os, sys
orc, tag, ck, replay, flange, exp, tok = sys.argv[1:8]
csv_path = os.path.join(orc, "out", "eval", "eval_results.csv") if orc else ""
try:
    rows = list(csv.DictReader(open(csv_path)))
    succ = sum(1 for r in rows if str(r.get("success", "")).strip().lower() == "true")
    prog = [float(r.get("progress", 0) or 0) for r in rows]
    res = f"{succ}/{len(rows)} progress {sum(prog)/max(1,len(prog)):.3f}"
except Exception as exc:
    res = f"ORACLE UNREADABLE ({exc})"
print(f"{tag}: B-ORACLE {res} | replay {replay} rad | {flange} | ckpt {ck} | export {exp} | tokens {tok} | oracle {orc}")
PY
