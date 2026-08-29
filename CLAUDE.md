# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ships

The entire submission is one class: `starter/agent.py::Agent`. The organizer's
evaluator imports it and calls `reset()` / `respond()` **in-process** — no network,
no CLI on the graded path. Everything else here (`cli/`, `tests/`, `scripts/`) is dev
tooling that never reaches the organizer.

Planning and reasoning live in a **separate repo**,
[`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs).
That repo is the *why*; this one is the *how*. When a change needs a rationale — why
BM25 and not dense, why the "clean six" attributes and not `other` — the answer is
in `project/standing-findings.md` there, not here.

## Commands

This is Windows-first development. **The interpreter is `python`, not `python3`** —
every docstring and README line in this repo uses the POSIX spelling and will fail
as written. `docs/windows-dev-setup.md` is the authoritative, verified setup guide;
read it before debugging any environment problem.

Always run from the repo root — `starter`, `evaluator`, and `cli` are imported as
top-level packages, and tests resolve paths relative to the root.

```powershell
# Full suite (22 tests)
python -m unittest tests.test_agent_contract tests.test_cli_integration tests.test_evaluator_smoke tests.test_ledger_scheduler tests.test_offline tests.test_p1_offline_safety -v

# One class / one test
python -m unittest tests.test_p1_offline_safety.TestRespondSafety -v
python -m unittest tests.test_agent_contract.TestAgentContract.test_never_raises_on_malformed_input -v

# Mirror the two CI jobs exactly
python -m unittest tests.test_agent_contract tests.test_ledger_scheduler tests.test_offline tests.test_p1_offline_safety -v
python -m unittest tests.test_cli_integration tests.test_evaluator_smoke -v

# Real scoring run (needs data/catalog.jsonl)
python -m evaluator.local_evaluator --catalog data\catalog.jsonl --dataset data\public_set.jsonl --output results.json

# Instrumented run: the five P1 acceptance criteria, per call
python scripts\verify_offline_safety.py

# Manual turn-by-turn chat (client spawns its own server subprocess)
python -m cli.client --catalog data\catalog.jsonl
python -m cli.client --catalog tests\fixtures\catalog.jsonl   # no catalog needed
```

There are no third-party dependencies. `requirements.txt` is comments only, and it
stays that way deliberately — fewer dependencies is less that can fail under the
organizer's offline final-scoring conditions.

## Architecture

`respond()` is a thin, paranoid wrapper around three collaborators:

- **`starter/ledger.py` — `SessionState`.** Accumulates every customer reply
  verbatim into one `disclosed_constraints` string. This is *the* lever the planning
  repo identified (0.16 → 0.75); no structured slot parsing. The single documented
  exception is a content-free reply (`_CONTENT_FREE_PATTERNS`), anchored at `^`
  because a message that merely *contains* a decline phrase after real content must
  still be appended.
- **`starter/retrieval.py` — `Bm25Index`.** SQLite FTS5 over an in-memory index,
  built once at construction. Queries are OR-joined over ≤40 unique stopword-filtered
  terms, ranked with a weighted `bm25()`.
- **`starter/scheduler.py` — `next_attribute(state)`.** Fixed six-attribute order.
  Takes the whole `state`, not just the asked list, so Phase 3 (ask-yield) can swap
  its body in behind this exact signature — it will need `state.retired` and
  `state.yield_seen` added to `SessionState`. **Do not widen this signature again.**

### The rule that governs every change here

The evaluator swallows exceptions into a silent zero. A schema-invalid dict is zeroed
just as silently. So:

- `respond()` must never raise — it returns `_empty_response()` on any exception.
- `__init__` and `reset()` are **not** wrapped by the evaluator
  (`local_evaluator.py:306` and `:228`). A raise in either kills the *entire run*,
  not one session. Both are guarded here; keep them guarded.
- Every outgoing payload passes through `_validated()`, which coerces each field to
  its schema-valid empty form rather than letting an invalid value through.

Adding any optional layer (rerank, LLM, classifier) means: its own try/except, a
local fallback, and never on the critical path.

`evaluator/` is **vendored from the competition kit and never edited.**
`scripts/leak_controlled_benchmark.py` needs to change `intent_card()` and honors
this by monkeypatching the imported module object at runtime and restoring it before
exit. Do the same if you ever need to vary evaluator behavior.

## Two things that will mislead you

**Local scores are inflated by a leak in the vendored simulator.** `public_set.jsonl`
carries no real `intent_card`, so the evaluator falls back to building the simulated
customer's "hidden" preferences out of the *target product's own listing* and reciting
them back turn by turn — 94% of disclosed constraint strings are exact substrings of
the target's indexed text. Phase 1's local hit@10 of 0.80 is an upper bound, not a
score. `scripts/leak_controlled_benchmark.py` brackets it. Never quote a local number
without saying which bracket it came from.

**A green test run proves less than it looks like.** `tests/fixtures/catalog.jsonl`
has 6 products against `top_k=10`, so any query matching one term returns the whole
fixture — `test_evaluator_smoke` passes even with a query-blind ranker.
`test_offline` is an AST check for banned import *names* in `starter/*.py` only; it
executes nothing and covers neither `evaluator/` nor `cli/`. And `python -m unittest
discover` reports `Ran 0 tests ... OK` if `tests/__init__.py` is ever deleted —
check the count before believing green. `docs/windows-dev-setup.md` §7 has the full
list.

## Silent failure to check first

If the agent returns `recommendations: []` on every turn with no error,
`data/catalog.jsonl` is missing. `Agent.__init__` swallows the load failure by design
and sets a null index; nothing warns. The file is gitignored (~50k rows, distributed
as a release asset) — see `docs/windows-dev-setup.md` §1. Confirm with:

```powershell
python -c "from starter.retrieval import Bm25Index; print(Bm25Index('data/catalog.jsonl').search('waterproof leather boots', 5))"
```

## `evaluation-data/` is test-only — do not read it while building

**Rule: during design and development, do not open, sample, quote, or tune
against anything under `evaluation-data/`. Read it only when running an
evaluation, and only through a scoring script.** This binds you as an assistant
exactly as it binds a human developer. If the task is to build or change the
reply parser, the retrieval config, the ledger, or the scheduler, that data is
out of scope for the work — say so and proceed without it, rather than opening
it "just to check".

The rule covers the generated artifacts too, not only the committed ones:
`evaluation-data/esci/esci_public_set.jsonl` is committed;
`bakeoff/cache/esci_catalog.jsonl` and any future `evaluation-data/paraphrase/`
corpus are generated and gitignored, and are equally out of bounds.

Why this is stated rather than enforced: **nothing enforces it.** These are plain
files in the working tree; no script can check who read what, and a README
saying "don't look" is a marker of intent, not a control.
`evaluation-data/README.md` says so outright, and records the guard that does
hold — *provenance*, not access. ESCI's queries were written by Amazon
customers, so reading them can cost you a config overfitted to 600 queries but
cannot make the data circular. A self-authored paraphrase holdout has no such
protection: reading it **is** the circularity, and it needs a real control
(kept out of the working tree, or decrypted only at scoring time) chosen before
it is generated, not after.

Tuning any component until a number on this data improves is the failure mode
being prevented. If you find yourself iterating against it, stop — that is
development, not testing, whatever it is labelled.

## Phase numbering

**This repo's `Phase 0`–`Phase 5` (see README) is the single canonical execution
plan.** Cite it as "Phase N" in issues, PRs, branch names, and benchmark labels.

Two other schemes exist in the planning repo and are **not** interchangeable — the
same token means different things. The README has the mapping table. Two traps:

- Planning repo `G1–G5` are *architecture gates*, not a delivery schedule (relabelled
  from `P1–P5` on 28 Aug 2026 to end a collision with this repo's phase numbers).
- **Phases 2 and 3 were swapped on 28 Aug 2026.** Retrieval was Phase 3, ask-yield was
  Phase 2. Pre-28-Aug references use the old numbering; where a number and a work item
  disagree, **the work item named alongside it is authoritative.**

Phase 3 (ask-yield) is on hold: planning repo issue #4 challenges its premise, since
only 40/200 public sessions reach turn 7 at all. Do not start it before that reports.

## Conventions

- Each phase gets its own branch, opened off `main` once the prior phase merged.
- Offline verification (`scripts/verify_offline_safety.sh`) is **macOS-only** — it
  needs `sandbox-exec` with `scripts/no-network.sb`. On Windows, the closest real
  equivalent is `docker run --rm --network none ...`; say explicitly that you supplied
  the block yourself when reporting such a result. Running
  `scripts/verify_offline_safety.py` unsandboxed covers criteria 1–3 only, never 4–5.
- `results*.json` are all gitignored and there is no committed baseline artifact, so
  record any number you measure somewhere durable — see
  `docs/benchmark-tracking-plan.md`.
