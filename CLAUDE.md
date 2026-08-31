# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ships

The submission is `agent.py` at the repo root, which exports `Agent` from `src/`.
The organizer's evaluator imports it and calls `reset()` / `respond()`
**in-process** — no network, no CLI on the graded path. The layout is the one
`docs/submission_rules.md` in the organizer's kit prescribes (`agent.py`,
`requirements.txt`, `README.md`, `src/`); that file is **not** vendored here, so
citations to it point at `../techjam-conversational-search/docs/`.

```text
agent.py    entry point — a sys.path self-heal, then `from src.agent import Agent`
src/        the system: 18 modules, standard library only
```

**`starter/` is the superseded first-generation system and is NOT part of the
submission.** It is retained unmodified as the historical record and as the
baseline the rebuild is measured against — do not edit it, and do not add
features to it. Everything else (`cli/`, `demo/`, `tests/`, `scripts/`,
`bakeoff/`) is dev tooling that never reaches the organizer.

`demo/` is the two-terminal demo — a chat frontend and a pipeline explainer,
driven by the evaluator's own simulated customer over `data/public_set.jsonl`.
It observes `src/` by **monkeypatching the private stage functions of
`src/pipeline.py` at runtime and restoring them in a `finally`**, so the
submission stays byte-identical and the graded path pays nothing. Two rules bind
anyone editing it, both enforced by `tests/test_demo_tracer_targets.py`: the
original stage function is called *before* any recording and every recorder sits
in its own `try/except` (a raise there would be swallowed by `run_turn` into
`_degraded_plan()`, and the demo would silently show a *different* agent), and
`demo/` never sets a seam flag. See `demo/README.md`.

The trap this creates: `evaluator/local_evaluator.py:12` hardcodes
`from starter.agent import Agent`, so running the vendored evaluator directly
scores the **superseded** system and reports a plausible number for the wrong
agent. Score `src/` only through `scripts/evaluate_src.py`.

Planning and reasoning live in a **separate repo**,
[`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs).
That repo is the *why*; this one is the *how*. When a change needs a rationale — why
BM25 and not dense, why the seven-slot schedule and never `other` — the answer is
in `project/standing-findings.md` there, or in `docs/artifacts`, not here.

## Commands

Development happens on both Windows and macOS — neither is the primary platform,
so keep tooling cross-platform and don't dismiss hygiene that only bites one side.
**On Windows the interpreter is `python`, not `python3`** — every docstring and
README line in this repo uses the POSIX spelling and will fail as written there.
`docs/windows-dev-setup.md` is the authoritative, verified setup guide; read it
before debugging any environment problem.

Always run from the repo root — `src`, `starter`, `evaluator`, and `cli` are
imported as top-level packages, and tests resolve paths relative to the root.

```powershell
# Full suite. Check the count before believing green: 492 at cfd6841 (main).
# ANCHOR THE NUMBER TO A COMMIT when you update it -- a bare count is stale the
# next time anyone adds a test, which is how 438 and 390 survived here so long.
python -m unittest discover -s tests -p "test_*.py" -t .

# One class / one test
python -m unittest tests.test_src_askpolicy -v
python -m unittest tests.test_agent_contract.TestAgentContract.test_never_raises_on_malformed_input -v

# Mirror the two CI jobs exactly. Modules are named EXPLICITLY in ci.yml, not
# discovered, so a NEW TEST FILE DOES NOT RUN UNTIL IT IS ADDED THERE.
python -m unittest tests.test_agent_contract tests.test_ledger_scheduler tests.test_offline tests.test_p1_offline_safety tests.test_src_agent tests.test_src_askpolicy tests.test_src_contract tests.test_src_frames tests.test_src_layering tests.test_src_layers tests.test_src_ledger tests.test_src_no_network tests.test_src_overlap tests.test_src_pipeline tests.test_src_rerank tests.test_src_retrieval tests.test_src_semantic_rung3 tests.test_src_shown tests.test_src_slots tests.test_bakeoff_llmrr tests.test_demo_tracer_targets -v
python -m unittest tests.test_cli_integration tests.test_evaluator_smoke tests.test_src_end_to_end tests.test_demo_trace -v

# Real scoring run — THE SUBMISSION (needs data/catalog.jsonl).
# --bracket both reports the leaky/scrubbed spread; quote both, never one.
python scripts\evaluate_src.py --catalog data\catalog.jsonl --dataset data\public_set.jsonl --bracket both

# Scores the SUPERSEDED starter/ agent, not the submission — see "What ships".
# Useful only as the historical control.
python -m evaluator.local_evaluator --catalog data\catalog.jsonl --dataset data\public_set.jsonl --output results.json

# Instrumented run: the five P1 acceptance criteria, per call (starter/ only)
python scripts\verify_offline_safety.py

# Manual turn-by-turn chat (client spawns its own server subprocess; starter/)
python -m cli.client --catalog data\catalog.jsonl
python -m cli.client --catalog tests\fixtures\catalog.jsonl   # no catalog needed
```

There are no third-party dependencies. `requirements.txt` is comments only, and it
stays that way deliberately — fewer dependencies is less that can fail under the
organizer's offline final-scoring conditions.

## Architecture

The design of record is `docs/artifacts` — "Statement 4 Architecture v5" and
"The Seven-Slot Ask Policy". Where this file and those disagree, they win.

`src/agent.py::Agent.respond()` is a never-raise wrapper around one pass in
`src/pipeline.py::run_turn`:

> decode → contradiction check → ledger append → query → BM25 → rerank seam →
> overlap gate → never-repeat selection → ask policy → schema coercion

- **`src/frames.py` — Tier 1 intent decode.** Anchored regex against the eight
  f-string templates the simulator emits, so it is a *decode*, not an estimate.
  It splits the two declines on the single token `additional`: "I don't have **a**
  preference" leaves the bucket live (re-ask later); "I don't have an
  **additional** preference" proves it empty (retire permanently). `TIER_15_HEDGE`
  is the one unanchored pattern and the one tuning knob.
- **`src/ledger.py` — `ConstraintLedger`.** Every disclosed reply appended
  verbatim; the concatenation of those raw strings **is** the query. Append-only,
  enforced by the absence of any deletion method — do not add one. Never erased,
  not even on intent override (`docs/hard-rules.md` rule 6).
- **`src/retrieval.py` — `Bm25Index`.** SQLite FTS5 over an in-memory index,
  built once at construction. The sole retrieval route: dense fusion was measured
  twice and rejected (−0.206, −0.065). Terms are quoted as phrases and OR-joined
  over ≤40 unique stopword-filtered terms so a stray FTS5 operator in a customer
  reply cannot break the query.
- **`src/askpolicy.py` — `next_attribute(state)`.** Seven-slot fixed order for
  turns 1–7, then a fallthrough ladder re-evaluated on each free turn. Never
  `null`, never `other`. Takes the whole `state`, so `src/askyield.py` can swap in
  behind this exact signature. **Do not widen this signature.**
- **`src/shown.py` — never-repeat.** `partition()`, never `filter()` — the top-10
  is always full. Carries the override guard: in `intent_override` sessions the
  evaluator's hit check is off early, so everything shown before the override goes
  back in play.
- **`src/slots.py` — scheduling only.** The typed slot view never touches
  retrieval, so a parsing bug can corrupt *which question we ask* but never *what
  we search*. Asserted structurally in `tests/test_src_layering.py`.

### The rule that governs every change here

The evaluator swallows exceptions into a silent zero. A schema-invalid dict is zeroed
just as silently. So:

- `respond()` must never raise — it returns `src/contract.py::empty_response()`
  on any exception.
- `__init__` and `reset()` are **not** wrapped by the evaluator
  (`local_evaluator.py:306` and `:228`). A raise in either kills the *entire run*,
  not one session. Both are guarded here; keep them guarded.
- Every outgoing payload passes through `src/contract.py::validated()`, which
  coerces each field to its schema-valid empty form rather than letting an invalid
  value through.
- The guards catch `Exception`, so they do not cover a raise at **import** time.
  `src/agent.py` imports its siblings at module scope unguarded, and `agent.py`
  catches only `ImportError` — a module-level `re.compile` that throws would kill
  the run before `Agent` exists. Keep module scope free of anything that can fail.

Four optional layers exist as typed seams behind master switches:
`src/rerank.py` (cross-encoder), `src/semantic.py` (Tier 2 intent fallback),
`src/llm_rerank.py` (LLM ranking escalation), plus `src/askyield.py` (adaptive
ask ordering). **Three of the four now default to enabled** — `cfd6841` flipped
them. Note the switches are not uniform in kind: two are module constants, one
is a parameter default with no named constant at all.

| module | switch | default | selected | needs |
|---|---|---|---|---|
| `src/rerank.py` | `load_reranker(enabled=True)` — **param default, no constant** | on | `ms-marco-MiniLM-L-6-v2` | `sentence-transformers` + `data/models/` checkpoint |
| `src/semantic.py` | `TIER2_ENABLED` | `True` | `rung3_centroid` | `model2vec` + `data/models/potion-base-8m` |
| `src/llm_rerank.py` | `LLM_RERANK_ENABLED` | `True` | `gemini-3.5-flash` | `google-genai` + `GEMINI_API_KEY` |
| `src/askyield.py` | `ADAPTIVE_ENABLED` | `False` | off | — |

**A flipped flag is necessary but not sufficient**: each loader checks its flag,
then its dependency, then that the built object is usable, and falls back to its
null implementation if any of those fails. So on a bare stdlib checkout — no
optional deps, no checkpoints, no key — all three still load `NullReranker` /
`NullSemanticDecoder` / `NullLlmReranker` and the graded path is byte-for-byte
the stdlib BM25 pipeline. **This means the flags alone do not tell you what ran.
Check the loader return, not the constant** (`Agent._deps`, or
`scripts/evaluate_src.py`'s printed `degraded`).

The consequence to keep in mind: installing `requirements-optional.txt` now
*does* change behaviour, and for `llm_rerank` it puts a hosted API call on the
graded path. Verified 1 Sep 2026 that this degrades safely — with the layer
live and the network down, all escalations fail and the 200-session score is
bit-identical to the stdlib baseline (`0.497383` scrubbed). The same contract
still binds every layer: its own try/except, a local fallback, never load-bearing
on the critical path. `src/` itself is standard library only; keep it that way.

`evaluator/` is **vendored from the competition kit and never edited.**
`scripts/leak_controlled_benchmark.py` needs to change `intent_card()` and honors
this by monkeypatching the imported module object at runtime and restoring it before
exit. Do the same if you ever need to vary evaluator behavior.

## Two things that will mislead you

**Local scores are inflated by a leak in the vendored simulator.** `public_set.jsonl`
carries no real `intent_card`, so the evaluator falls back to building the simulated
customer's "hidden" preferences out of the *target product's own listing* and reciting
them back turn by turn — 94.5% of disclosed constraint strings are exact substrings
of the target's indexed text. **Never quote a local number without saying which
bracket it came from.** `scripts/evaluate_src.py --bracket both` reports the spread;
`scripts/leak_controlled_benchmark.py` is the older single-arm tool.

Measured 31 Aug 2026 over the 200 public sessions, both arms, for reference:

| system | leaky (upper) | scrubbed (lower) |
|---|---|---|
| `src/` (the submission) | 0.872057 | 0.497383 |
| `starter/` (superseded) | 0.692586 | 0.198439 |

The rebuild's gain is *larger* with the leak removed (+0.299 scrubbed vs +0.179
leaky), which is the opposite of a measurement artifact. A leaky hit@10 of 0.995 is
still an upper bound, not a score.

**A green test run proves less than it looks like.** `tests/fixtures/catalog.jsonl`
has 6 products against `top_k=10`, so any query matching one term returns the whole
fixture — `test_evaluator_smoke` passes even with a query-blind ranker.
`test_offline` is an AST check for banned import *names* in `starter/*.py` only; it
executes nothing and covers neither `evaluator/` nor `cli/`. `tests/test_src_no_network.py`
is the equivalent for `src/` and is the one that matters now. CI names its test
modules explicitly rather than discovering them, so **a new test file is silently
not run until it is added to `.github/workflows/ci.yml`.** And `python -m unittest
discover` reports `Ran 0 tests ... OK` if `tests/__init__.py` is ever deleted —
check the count before believing green (492 at `cfd6841`; `layer3-true-installable-sourced`
carries more). `docs/windows-dev-setup.md` §7 has the full list.

**A passing count is not a covered count, and this file has been wrong about it
twice.** Two live examples, both found 1 Sep 2026: `test_load_llm_reranker_defaults_to_null`
asserted an *unconditional* null, so it stayed green in CI while asserting
nothing about the flag it was named for (fixed in `85f9f53`); and the whole of
`CentroidSemanticDecoder` is dead code on any checkout without `model2vec` and
the potion-8m weights, so a full green suite there exercises **none** of it. If
a module's dependency is absent, its tests are not failing — they are not
running. Say which configuration a green run was green *in*.

## Silent failure to check first

If the agent returns `recommendations: []` on every turn with no error,
`data/catalog.jsonl` is missing. `Agent.__init__` swallows the load failure by design
and sets a null index; **there is no logging anywhere in `src/`**, so nothing warns.
The file is gitignored (~50k rows, distributed as a release asset) — see
`docs/windows-dev-setup.md` §1. Confirm with:

```powershell
python -c "from src.retrieval import Bm25Index; print(Bm25Index('data/catalog.jsonl').search('waterproof leather boots', 5))"
python -c "from agent import Agent; print('degraded:', Agent('data/catalog.jsonl').degraded)"
```

`Agent.degraded` is the programmatic form of the same check, and
`scripts/evaluate_src.py` prints it on every run. A `TechnicalScore` of exactly
`0.00000` is almost always this, not a solution regression.

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
- `results*.json` are all gitignored — they are regenerable per-session dumps. The
  durable record is the two **tracked** run logs: `results_src.md` (the submission,
  appended by `scripts/evaluate_src.py`) and `results.md` (the `starter/` control,
  appended by `.claude/skills/run-sol/bench.py`). An unrecorded number is a lost
  number — commit the row alongside whatever change produced it. See
  `docs/benchmark-tracking-plan.md`.
