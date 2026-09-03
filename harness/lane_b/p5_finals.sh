#!/usr/bin/env bash
# P5 finals copy (HARD RULE 3: small finals only, never model_step_*.pt trees; copy, never move).
# usage: p5_finals.sh <name> <export_onnx run> <decoder_replay run> <label_tokens run> <oracle_b run> "<description>"
# -> ~/runs/franka-sonic/lane_b/final/p5/<name>/{onnx,replay,tokens,oracle_eval,README.md}
set -u
NAME="$1"; EXP="$2"; REP="$3"; TOK="$4"; ORC="$5"; DESC="$6"
FREE_GB=$(df -BG ~ | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "$FREE_GB" -lt 100 ]; then echo "NEEDS-COPY: home has ${FREE_GB} GB free (< 100 GB); finals for $NAME not copied: $EXP $REP $TOK $ORC"; exit 2; fi
F=~/runs/franka-sonic/lane_b/final/p5/$NAME
mkdir -p "$F/onnx" "$F/replay" "$F/tokens" "$F/oracle_eval"
cp -n "$EXP"/out/*.onnx "$EXP"/out/export_summary.json "$EXP"/out/model_config.yaml "$F/onnx/" 2>/dev/null
cp -n "$REP"/out/replay*.json "$REP"/out/replay_trajectories.npz "$F/replay/" 2>/dev/null
cp -n "$TOK"/out/tokens/index.json "$TOK"/out/tokens/hold_token.json "$TOK"/out/tokens/label_summary.json "$F/tokens/" 2>/dev/null
cp -rn "$ORC"/out/eval "$F/oracle_eval/" 2>/dev/null
cp -n "$ORC"/README.md "$F/oracle_eval/oracle_README.md" 2>/dev/null
{ echo "P5 finals — $NAME. $DESC"; echo "- onnx/ from $EXP; replay/ from $REP; tokens/ index + hold token + summary from $TOK (the token npz + gr00t_v2_sonic dataset stay in that run folder); oracle_eval/ from $ORC."; echo "- copied $(date -u +%Y-%m-%dT%H:%M:%SZ) by harness/lane_b/p5_finals.sh"; } > "$F/README.md"
echo "finals -> $F ($(du -sh "$F" | cut -f1))"
