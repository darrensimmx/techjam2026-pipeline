# Benchmark tracking plan

**Goal:** answer "did this change help or hurt?" every time a feature lands
(Phase 2 retrieval changes, Phase 3 ask-yield, any tuning pass), without
re-deriving it from memory or eyeballing a `results.json` that gets
overwritten on every run.

## What gets recorded

Everything `evaluator/local_evaluator.py`'s `evaluate()` already computes —
nothing new to calculate, just something to stop discarding:

- `hit_rate_at_10`, `mrr`, `mttc`, `efficiency`, `recommended_technical_score`
- `scenario_metrics` broken out by Buying / Browsing / Intent Override /
  Boundary — tracked *separately*, not just the overall number, because
  they're weighted completely differently (Boundary is 5% of sessions,
  Intent Override's detection ceiling is ≤0.0078 per the planning repo) and a
  change can easily help one and quietly cost another
- `reported_token_usage` — a feasibility signal, currently always zero since
  Phase 1 uses no model, but the field exists so a future LLM-based feature
  doesn't need new plumbing to be tracked

Plus metadata to make one row meaningful on its own: timestamp, git commit
SHA, branch, and a short human label (`phase1-baseline`,
`phase2-hybrid-retrieval`, `phase3-ask-yield-v1`, ...).

## Where it lives

`benchmarks/history.jsonl` in this repo — one JSON object per line,
append-only, git-tracked. Not a database, not a dashboard service:

- **Diffable in PRs.** A reviewer sees `"hit_rate_at_10": 0.23 -> 0.31`
  directly in the diff, next to the code change that produced it.
- **No infrastructure.** It's a file. It needs nothing running to exist.
- **Survives exactly as long as the repo does** — same durability as the code.

This is deliberately separate from `results.json` (the evaluator's default
output), which stays exactly as it is today: a full per-session dump,
overwritten every run, useful for debugging one run in detail.
`history.jsonl` is the append-only ledger of just the summary metrics,
sitting next to it.

## How a row gets added

**1. Locally, whenever you tune something:**

```bash
python3 scripts/benchmark.py --label "phase3-ask-yield-v1"
```

Runs the real evaluator against the real `data/catalog.jsonl` and the full
200-session `public_set.jsonl`, prints the summary table, and appends one row
to `benchmarks/history.jsonl` (commit SHA and timestamp filled in
automatically). Commit that file alongside the code change in the same PR —
the score and the change that produced it land together in git history,
instead of the score living only in someone's terminal scrollback.

**2. Automatically in CI, on every merge to `main`:**

Extend the CI pipeline (`.github/workflows/ci.yml`) with a third job,
triggered on `push` to `main` only — not every PR or WIP branch, so the
history reflects landed decisions, not every experiment:

- downloads the real `catalog.jsonl.gz` from the competition repo's GitHub
  Release (Actions runners have full internet access, unlike the sandbox
  this was built in — this is *also* how Phase 1's open gap, "no CI run
  against the real catalog," gets closed)
- runs the full evaluator against all 200 public sessions
- appends a row labeled with the commit SHA and PR title
- commits `benchmarks/history.jsonl` back to `main` (or opens a small
  single-file auto-PR, if pushing directly to `main` from CI isn't wanted)

## Comparing runs

```bash
python3 scripts/compare_benchmarks.py --against phase1-baseline
# or
python3 scripts/compare_benchmarks.py --last
```

Prints a delta table — overall metrics and the per-scenario breakdown —
between the current run and any past labeled row. This is the actual
"are we heading the right direction" check.

On a PR, the same delta could post as a comment against the last row on
`main`. Deliberately **informative, not a gate** — no hard CI failure on
regression. Early iteration legitimately trades one scenario for another
(ask-yield might cost Boundary sessions a turn while fixing Browsing), and
the planning repo's own standing findings already say TechnicalScore isn't
the whole rubric. A hard gate would fight that judgment call instead of
informing it.

## What this buys, concretely

- **After Phase 3 (ask-yield):** does HitRate@10 move toward the ~0.68–0.75
  range the planning repo's review claims, or not? One row answers it.
- **After Phase 2 (retrieval bake-off):** does hybrid retrieval or rerank
  actually beat BM25-only, *per scenario type*, or was the prior assumption
  right all along? The scenario breakdown answers this without re-running
  anything by hand.
- **Before the Phase 5 report:** the whole trend line already exists in git
  history — pull `benchmarks/history.jsonl` straight into the report's
  results section instead of reconstructing it from memory or old terminal
  output.

## Not doing (deliberately, for now)

- No dashboard or visualization service — a JSONL file plus a compare script
  is enough at this scale (a handful of labeled runs, not thousands)
- No hard regression gate in CI — informative delta only, human decides
- No tracking on every push or PR — only merges to `main`, so the recorded
  history reflects landed changes, not every experiment branch
