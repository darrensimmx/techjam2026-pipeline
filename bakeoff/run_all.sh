#!/usr/bin/env bash
# Runs the parts that need the bi-encoder embeddings, in order, once dense.py
# has finished. Each step is allowed to fail without blocking the next, so one
# bug does not cost the whole queue -- check the log for "FAILED".
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG_DIR="${1:?usage: run_all.sh <log-dir>}"

# Single-instance lock. Two copies of this script racing is not hypothetical:
# it happened once, when a background job was stopped while parked in the wait
# loop below and its detached subshell outlived the kill, so both it and the
# replacement fired the moment the embeddings landed. They then wrote the same
# .npy and .json cache files concurrently. Nothing crashed and both reported
# OK -- which is the dangerous part, because a torn cache file would have been
# read back as data.
LOCK="bakeoff/cache/.run_all.lock"
mkdir -p bakeoff/cache
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING: $LOCK exists -- another run_all.sh is live (or died dirty)."
  echo "Check with: wmic process where \"name='python.exe'\" get CommandLine"
  echo "If nothing is running: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

echo "waiting for dense.py to finish..."
until [ -f bakeoff/cache/queries-bge.npy ]; do sleep 30; done
echo "embeddings ready: $(date)"

step () {
  local name="$1"; shift
  echo ">>> $name  $(date +%H:%M:%S)"
  if "$PY" -u "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "<<< $name OK"
  else
    echo "<<< $name FAILED (see $LOG_DIR/$name.log)"
  fi
}

step part2 bakeoff/part2_dense.py
step part3 bakeoff/part3_fusion.py
step part5 bakeoff/part5_realqueries.py
step part4 bakeoff/part4_rerank.py
echo "ALL STEPS DONE $(date)"
