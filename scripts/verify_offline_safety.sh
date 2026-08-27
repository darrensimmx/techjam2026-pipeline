#!/usr/bin/env bash
# P1 offline-safety verification, end to end.
#
#   1. control: prove the network block is REAL (unsandboxed network works,
#      sandboxed network does not) -- otherwise "passes offline" proves nothing
#   2. criterion 4, literally: `python3 -m evaluator.local_evaluator` over the
#      full 200-session public set with networking revoked
#   3. criteria 1-3 and 5: the same run, instrumented per call
#
# Usage: scripts/verify_offline_safety.sh
#
# macOS only: sandbox-exec is the kernel-level block. CI runs on Linux and so
# runs the tests (criteria 1-3) but NOT this script -- the Linux equivalent is
# `unshare -rn`, unavailable on GitHub-hosted runners without privileges. The
# criterion-4 evidence is therefore a local run, not a CI artifact.
#
# The criteria 4/5 numbers quoted in PR #2 (recommended_technical_score =
# 0.722818) came from @JamisonTeng's local macOS run of this script -- not
# from CI, and results_offline*.json is gitignored, so there's no committed
# artifact to check them against (see issue #5). Ping him if those numbers
# ever need to be reproduced or re-verified.
set -u
cd "$(dirname "$0")/.."
PROFILE="scripts/no-network.sb"
status=0

echo "===================================================================="
echo "STEP 1a  control -- network UNSANDBOXED (expected: reachable)"
echo "===================================================================="
if python3 scripts/netprobe.py; then
  echo ">> UNEXPECTED: no network available even unsandboxed."
  echo ">> The block below cannot be shown to be doing anything. Aborting."
  exit 1
fi
echo ">> Confirmed: the probe detects a working network, so it is not vacuous."

echo
echo "===================================================================="
echo "STEP 1b  control -- network SANDBOXED (expected: every op denied)"
echo "===================================================================="
if ! sandbox-exec -f "$PROFILE" python3 scripts/netprobe.py; then
  echo ">> FAIL: the sandbox did not deny every network operation."
  exit 1
fi
echo ">> Confirmed: kernel-level denial. The guard is real, not a Python stub."

echo
echo "===================================================================="
echo "STEP 2  criterion 4 -- the graded command, networking revoked"
echo "===================================================================="
if sandbox-exec -f "$PROFILE" python3 -m evaluator.local_evaluator --output results_offline_plain.json; then
  echo ">> PASS: python3 -m evaluator.local_evaluator completed with no network."
else
  echo ">> FAIL: the evaluator crashed under the network block."
  status=1
fi

echo
echo "===================================================================="
echo "STEP 3  criteria 1-3, 5 -- instrumented run, networking revoked"
echo "===================================================================="
if ! sandbox-exec -f "$PROFILE" python3 scripts/verify_offline_safety.py; then
  status=1
fi

echo
if [ "$status" -eq 0 ]; then
  echo "ALL P1 ACCEPTANCE CRITERIA PASS (networking revoked, block verified real)"
else
  echo "P1 VERIFICATION FAILED -- see above"
fi
exit "$status"
