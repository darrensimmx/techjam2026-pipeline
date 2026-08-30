# Findings — approaches ruled out or deprioritized

**Status:** ✅ current. **Advisory for `starter/`** — `docs/hard-rules.md` is what binds.

Each entry records a decision *and the evidence for it*, so it can be reopened by
new measurement rather than by re-argument. **`Basis: UNMEASURED` means a priority
call, not a finding** — an entry says so plainly rather than implying evidence
that does not exist.

Established 30 Aug 2026 against `evaluator/local_evaluator.py`, the 200-sample
public set, and `bakeoff/results-part1.json`. Planning and rationale live in the
separate repo `darrensimmx/techjam2026-docs`; this file records what *this* repo
has decided not to build, and why.

Open questions go in `docs/TBD.md`, not here. An entry belongs here only once it
is closed.

---

## SLM / LLM per-turn intent classification

**Verdict:** RULED OUT · **Basis:** STRUCTURAL · **Decided:** 30 Aug 2026

The proposal: classify each turn's intent with a small model, then branch —
retrieve for a buying turn, have an LLM write clarifying questions for a browsing
turn.

It cannot work here, for reasons that are properties of the harness rather than
of model quality:

- **The agent's `message` is never read.** `customer_reply()`
  (`evaluator/local_evaluator.py:166`) takes
  `(sample, ask_attribute, disclosed, boundary_used)`. The response's `message`
  string is not a parameter and reaches nothing. An LLM-written question is
  written into a field with no reader; only the 10-value `ask_attribute` enum
  reaches the simulated customer.
- **There is no natural language to classify.** Every reply is one of four
  templates — `local_evaluator.py:169`, `:171`, `:183`, `:185` — plus three
  openers from `initial_message()` (`:154`). Reading them is format decoding,
  closer to `json.loads` than to NLU, and a substring check is already at 100%.
- **Intent is not the agent's to infer.** `scenario_type` is fixed in the dataset
  before the session starts (80 buying, 80 browsing, 30 intent_override,
  10 boundary across the 200 public samples). Nothing in the evaluation loop
  branches on what the agent believes it to be.
- **The branches are not alternatives.** Recommendations are checked every turn
  (`local_evaluator.py:252`) and the ask is consumed on the same turn (`:266`).
  Doing one *instead of* the other forfeits half of every turn. This is what
  `docs/hard-rules.md` rules 1 and 2 encode.
- **It would put a model dependency on the critical path.** `respond()` failures
  become silent zeros, and final scoring may run with networking disabled.

*Binds:* `docs/hard-rules.md:29` (rule 3, no model on the parse path) and
`:81` (rule 7, no scenario classifier).

**Reopen when:** the organizer exercises the paraphrasing clause in
`competition_specification.md`. Reply *frames* would then stop being templates,
and the scoped response is the Tier-2 semantic fallback described in the docs
repo's glossary — a classifier over four known frames behind a regex fast path —
not a general LLM in the loop.

---

## Weighted attributes

**Verdict:** DEPRIORITIZED · **Basis:** **UNMEASURED** · **Decided:** 30 Aug 2026

The proposal covers two related ideas: sweeping the FTS5 column weights, and
weighting ledger terms by attribute or recency when building the query.

**Nothing here has ever been measured.** This is a sequencing decision, not a
negative result:

- The column weights at `starter/retrieval.py:76` —
  `0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0` — are inherited **verbatim** from the
  organizer's weak baseline at
  `techjam-conversational-search/starter/agent.py:93`. So are `_TOKEN_RE`, the
  stopword list, the `len(token) > 1` filter, the 40-term cap, and the OR-join.
  Nobody on this team chose them.
- No bake-off arm has swept them. The only weight sweep in `bakeoff/` is R4's
  BM25-vs-dense **fusion** weight (`bakeoff/part3_fusion.py:5`), which is a
  different quantity.
- `bakeoff/cache/` retains per-turn candidate pools, so a sweep is cheap to run
  when it is scheduled (`bakeoff/capture.py:16`).

Two levers with real measurements sit ahead of it:

| Lever | Effect | State |
|---|---|---|
| Cross-encoder rerank | **+0.047**, CI excludes zero | Blocked on the organizer's unpublished timeout (`README.md:111`, planning repo issue #9) |
| Ledger-freeze regression | **−0.030**, unrecovered | Diagnosed in `docs/ledger-freeze-regression.md`; fix belongs in ask-yield |

**Do not cite `attribute band ≈+0.004` as evidence against weighting.** That
figure (`README.md:56-57`, drawn in Plate 5 of `docs/pipeline-drawings.html`) is
the attribute-**selection** band — the spread available from choosing *which
attribute to ask*, i.e. the ask-yield lever. It measures nothing about attribute
weighting, and was misread that way once already.

**Reopen when:** both rows above are closed — the rerank decision is made and the
−0.030 is recovered. At that point the sweep is a contained experiment: re-rank
the retained pools under candidate weight vectors and score with
`bench.py eval`. Note that a pure reweighting of the existing top-10 is bounded
above by oracle@10 = **+0.0825** (`bakeoff/results-part1.json`,
`current.oracle.10.delta_ci.mean_delta`), so size the effort against that ceiling
rather than against oracle@100.

---

## How to add an entry

Copy the shape above: verdict, basis, decision date, evidence bullets that each
carry a `file.py:line` pin, and a concrete **Reopen when**. If you cannot write
the reopen condition, the decision is not closed — put it in `docs/TBD.md`
instead.

Pins go stale. `python3 .claude/skills/update-drawings/drawings.py check` verifies
the ones this repo's drawings depend on; the rest are on you to re-check when you
move code.
