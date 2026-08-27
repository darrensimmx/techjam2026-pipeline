# Benchmarks

This directory is the "did that change help or hurt" ledger. See
[`../docs/benchmark-tracking-plan.md`](../docs/benchmark-tracking-plan.md)
for the full rationale; this file is the practical how-to.

## Files

- **`history.jsonl`** — one JSON object per line, append-only, one row per
  benchmark run. Never edit past rows by hand; only append. Fields:
  `timestamp`, `label`, `commit`, `branch`, `sample_count`,
  `hit_rate_at_10`, `mrr`, `mttc`, `efficiency`,
  `recommended_technical_score`, `scenario_metrics` (broken out by the
  competition's four scenario types -- buying/browsing/intent_override/
  boundary, each with its own hit_rate/mrr/mttc), `reported_token_usage`,
  `elapsed_seconds`.
- **`findings.md`** — the qualitative parallel to `history.jsonl`. History
  tells you *what* the numbers were; this tells you *why it mattered* — a
  dated entry each time a run reveals something worth remembering (a
  surprising number, a hypothesis confirmed or reversed, a tuning lever
  that mattered or didn't). Append-only, same convention as
  `decisions/standing-findings.md` in the planning repo — never delete an
  entry, mark it superseded if a later run contradicts it.

## Running a benchmark

```bash
python3 scripts/benchmark.py --label "phase2-ask-yield-v1"
```

Runs the real vendored evaluator against the real `data/catalog.jsonl`
(decompressed automatically from `catalog.jsonl.gz` if missing) and the
full 200-session `data/public_set.jsonl`, prints a summary, and appends one
row to `history.jsonl`.

Pick a label that says what changed, not just a run number —
`phase2-ask-yield-v1` beats `run3`. Commit the updated `history.jsonl` (and
a `findings.md` entry, if the result says something worth keeping) in the
same PR as the code change that produced it, so the score and the change
land together in git history.

## Reading the trend

Right now: open `history.jsonl` and read it top to bottom — it's one row.
A `scripts/compare_benchmarks.py` delta tool (diff two labeled runs) isn't
built yet; add it once there are enough rows that eyeballing stops being
enough.

## Current baseline

```
phase1-baseline: HitRate@10 0.85, MRR 0.525, MTTC 3.99, TechnicalScore 0.7228
```

See `findings.md` for what this number means and why it's higher than the
planning repo projected.
