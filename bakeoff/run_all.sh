#!/usr/bin/env bash
# Runs the parts that need the bi-encoder embeddings, in order, once dense.py
# has finished. Each step is allowed to fail without blocking the next, so one
# bug does not cost the whole queue -- check the log for "FAILED".
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG_DIR="${1:?usage: run_all.sh <log-dir>}"

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
