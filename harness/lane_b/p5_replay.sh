#!/usr/bin/env bash
# P5 helper (canonical copy: harness/lane_b/p5_replay.sh; runs from any cwd).
# usage: p5_replay.sh <model_step.pt> <tag>   -> export_onnx + decoder_replay --checkpoint (allocator picks a free device)
# Always passes --checkpoint to decoder_replay (without it bakeoff falls back to the newest run WITH a train_summary).
set -u
CK="$1"; TAG="$2"; cd ~/code/franka-sonic
LOGDIR=${P5_LOGDIR:-/tmp/franka-sonic/lane_b}; mkdir -p "$LOGDIR"; LOG=$LOGDIR/p5_${TAG}.log
{ echo "=== export $(date -u +%H:%M:%S) $CK"; python3 harness/bakeoff.py run lane_b export_onnx --checkpoint "$CK" 2>&1 | grep -E "CUDA_VISIBLE|run folder|OK rc|FAILED" | tail -3; } > "$LOG" 2>&1
EXP=$(grep -oE "^\[bakeoff\] run folder \S+" "$LOG" | tail -1 | awk '{print $4}')
if [ -z "$EXP" ] || ! grep -q "$CK" "$EXP/out/export_summary.json" 2>/dev/null; then echo "$TAG: EXPORT FAILED (see $LOG)"; exit 1; fi
{ echo "=== replay $(date -u +%H:%M:%S) onnx=$EXP"; python3 harness/bakeoff.py run lane_b decoder_replay --onnx "$EXP" --checkpoint "$CK" 2>&1 | grep -E "CUDA_VISIBLE|run folder|replay:|OK rc|FAILED" | tail -4; echo "=== done $(date -u +%H:%M:%S)"; } >> "$LOG" 2>&1
REP=$(grep -oE "^\[bakeoff\] run folder \S+" "$LOG" | tail -1 | awk '{print $4}')
~/Isaac-GR00T/.venv/bin/python - "$REP/out/replay.json" "$TAG" "$EXP" "$REP" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); names=['L1','R1','L2','R2','L3','R3','L4','R4','L5','R5','L6','R6','L7','R7']
pj=d['per_joint_target_error_rad']; worst=sorted(zip(pj,names),reverse=True)[:3]
print(f"{sys.argv[2]}: mean {d['mean_joint_error_rad']:.3f} (measured {d['mean_measured_joint_error_rad']:.3f}) body_pos {d['env_error_body_pos_m_mean']:.3f} m worst {[(n,round(e,2)) for e,n in worst]} export {sys.argv[3]} replay {sys.argv[4]}")
print("   per joint", {n:round(e,2) for n,e in zip(names,pj)})
PY
