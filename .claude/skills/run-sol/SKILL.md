---
name: run-sol
description: Build, run, test, score, and regression-check the TechJam conversational-search agent. Use when asked to run or start the agent, run the tests, evaluate or benchmark it, get a technical score or hit rate, check for a score regression, restore the catalog, or bisect a metric change across commits.
---

# Run the TechJam conversational-search agent

There is no server and no UI. The "app" is `Agent` in `starter/agent.py`, driven
in-process by the organizer's evaluator over 200 simulated shopping sessions.
Running it means **scoring it**.

Everything goes through one driver: `.claude/skills/run-sol/bench.py`.
All paths below are relative to the repo root; run from there.

## Prerequisites

Python 3.10+ and git. Nothing else — `requirements.txt` is deliberately empty
and the driver is standard-library only. Verified on Python 3.14.2 / SQLite
3.50.4 (FTS5 required, and present in the stock macOS build).

## Setup — do this first, always

`data/catalog.jsonl` (50,000 rows) is gitignored and **absent from `main`**.
`data/README.md` tells you to download it from a GitHub Release. Ignore that —
the gzipped blob is already in the local object store:

```bash
python3 .claude/skills/run-sol/bench.py setup
```

Restores the catalog from `origin/benchmark-tracking`, verifies its SHA-256,
and confirms the BM25 index actually builds. Takes ~2s. Idempotent.

If it reports the blob is unreadable: `git fetch origin benchmark-tracking`.

## Run

```bash
python3 .claude/skills/run-sol/bench.py eval
```

Scores all 200 public sessions (~15s) and prints the headline metrics, the
four per-scenario breakdowns, and deltas against every reference score in
`baselines.json`. Writes `results.json` (gitignored).

```
hit@10 0.8000  mrr 0.525619  mttc 4.2550  score 0.692586

scenario            n    hit@10        mrr     mttc
boundary           10    0.6000   0.533333   6.2000
browsing           80    0.7875   0.494072   4.3625
buying             80    0.8250   0.506835   3.6750
intent_override    30    0.8333   0.657262   4.8667

  vs organizer weak-BM25 baseline       0.106710  +0.585876
  vs phase1-baseline @ ecacc52          0.722818  -0.030232
  vs main @ 2ba2747 (current)           0.692586  +0.000000
```

`recommended_technical_score` is the number that matters:
`0.50*hit@10 + 0.30*mrr + 0.20*efficiency`.

## Test

```bash
python3 .claude/skills/run-sol/bench.py test
```

Runs the project suite (22 tests, no catalog needed — uses
`tests/fixtures/catalog.jsonl`) **and** the organizer's own
`tests/test_evaluator.py` against our vendored evaluator, plus a `diff` drift
check that `evaluator/` is still byte-identical to the organizer's copy.

The organizer repo is expected at `../techjam-conversational-search`; override
with `--official-repo PATH`. If it is absent, that half is skipped, not failed.

Note the organizer's test drives a fake `EchoTargetAgent` against its own 2-row
catalog — it tests the *evaluator*, not our agent. It passes trivially and
proves only that we have not modified the vendored harness. **The test of our
agent is `eval`/`check`.**

## Guard against regressions

```bash
python3 .claude/skills/run-sol/bench.py check          # exits 1 if score drops
python3 .claude/skills/run-sol/bench.py check --min 0.75
```

Exits non-zero below `regression_guard` in `baselines.json` (currently `0.69`).
Raise that number whenever a real improvement lands.

**Run this before every merge.** The unit tests pass in both the pre- and
post-regression states of the ledger filter — they structurally cannot see a
score change. `check` is the only thing that can.

## Find which commit moved the score

```bash
python3 .claude/skills/run-sol/bench.py bisect ecacc52 94e8916 3bc061f main
```

Scores each revision in an isolated `git archive` tree (~15s each) and flags
deltas. Never touches the worktree or the index. This is how the `3bc061f`
regression in `docs/ledger-freeze-regression.md` was found.

## Gotchas

- **A missing catalog scores 0.0 silently.** `Agent.__init__` swallows the load
  failure and sets `self._index = None`; every turn then returns an empty but
  schema-valid response. No exception, no warning — it reads as a catastrophic
  solution regression when it is a data problem. `setup` exists to make this
  loud, so never skip it.
- **Never `git checkout origin/benchmark-tracking -- data/catalog.jsonl.gz`.**
  `.gitignore` covers `data/catalog.jsonl` but *not* the `.gz`, so that stages a
  19 MB blob into your index. Use `git show` (what the driver does).
- **The evaluator turns crashes into zeros.** `respond()` is wrapped in a bare
  `except`, and a response that is not a dict with a `str` message is silently
  replaced by an empty one (`local_evaluator.py:239-244`). A broken agent scores
  low rather than erroring. But `Agent()` and `reset()` are **not** wrapped —
  a raise there kills the entire run.
- **There is no timeout anywhere in the evaluator.** A hang blocks forever.
- **`tests/__init__.py` must exist.** Without it `unittest discover` reports
  "Ran 0 tests ... OK" — a green false negative.
- **The suite is `unittest`, not `pytest`,** and there is no `pyproject.toml`.
- **Results are bit-identical across runs** even though the evaluator mints a
  random `uuid4` session id. Any delta you see is real signal, never noise.
- **`docs/windows-dev-setup.md:124` claims 21 tests.** It is 22. Stale doc.
- **`ResourceWarning: unclosed database` spam** from `test_p1_offline_safety` is
  noise from the test itself, not a failure.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not read the catalog blob` | `git fetch origin benchmark-tracking` |
| Every metric `0.0`, no error | Catalog missing or truncated — run `setup` |
| `Ran 0 tests ... OK` | `tests/__init__.py` was deleted |
| `DRIFT vendored evaluator/ differs` | Someone edited `evaluator/`. It is a verbatim vendored copy — revert it |
