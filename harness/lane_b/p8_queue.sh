#!/usr/bin/env bash
# P8 WP 8.2 ceiling-test queue (canonical copy: harness/lane_b/p8_queue.sh).
#
#   p8_queue.sh <lane-name> <tag>:<sonic_rl run folder>:<iteration> [...]
#
# Runs harness/lane_b/p8_ceiling.sh serially over a list of (variant, iteration) pairs,
# waiting (bounded) for each checkpoint to be written first. Two of these run in parallel on
# the two devices deliberately left outside the six training claims; each p8_ceiling.sh step
# acquires its own device through harness/gpus.py, so a lane never holds more than one.
#
# `last` as the iteration means the run's last.pt.
set -u
LANE="$1"; shift
cd "$HOME/code/franka-sonic"
LOGDIR="${P8_LOGDIR:-/tmp/franka-sonic/lane_b/p8}"; mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue_${LANE}.log"
WAIT_MIN="${P8_WAIT_MIN:-90}"

for spec in "$@"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; rf="${rest%%:*}"; iter="${rest##*:}"
  if [ "$iter" = "last" ]; then
    pat="$rf/out/TRL_DualFR3_Track/*/last.pt"
  else
    pat="$rf/out/TRL_DualFR3_Track/*/model_step_$(printf '%06d' "$iter").pt"
  fi
  ck=""
  for i in $(seq 1 $((WAIT_MIN * 4))); do
    ck=$(ls -1 $pat 2>/dev/null | head -1)
    [ -n "$ck" ] && break
    sleep 15
  done
  if [ -z "$ck" ]; then
    echo "$(date -u +%H:%M:%S) $LANE $tag: checkpoint $pat never appeared in ${WAIT_MIN} min" >> "$QLOG"
    continue
  fi
  # last.pt is rewritten every 50 iterations: give the writer a moment to finish.
  sleep 20
  echo "$(date -u +%H:%M:%S) $LANE $tag: testing $ck" >> "$QLOG"
  bash harness/lane_b/p8_ceiling.sh "$ck" "$tag" >> "$QLOG" 2>&1
  echo "$(date -u +%H:%M:%S) $LANE $tag: done" >> "$QLOG"
done
echo "$(date -u +%H:%M:%S) $LANE: QUEUE EMPTY" >> "$QLOG"
