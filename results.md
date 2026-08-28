# Benchmark results

Auto-updated by `.claude/skills/run-sol/bench.py` on every `eval` and `check`.
Newest run first. `score` is `recommended_technical_score` over the 200-session
public set: `0.50*hit@10 + 0.30*mrr + 0.20*efficiency`.

Reference points — organizer weak-BM25 baseline `0.106710`; `phase1-baseline`
@ `ecacc52` `0.722818`. See `docs/ledger-freeze-regression.md`.

A `*` after the commit means the worktree had uncommitted changes, so that row
does not correspond to the commit alone.

| when (UTC) | commit | branch | score | delta | hit@10 | mrr | mttc | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-28 11:13 | `29455e3` | `skill/run-sol-benchmark-harness` | **0.692586** | — | 0.8000 | 0.525619 | 4.255 | harness landed; establishes the reference row |
