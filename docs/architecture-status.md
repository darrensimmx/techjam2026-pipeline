# Architecture status — what `starter/` actually is, against the current design

**Status:** ✅ current as of 30 Aug 2026. **Derived, not normative.**

**Authoritative source: `techjam2026-docs/project/architecture-diagram.md`** — the mermaid,
and only the mermaid. This file is the *implementation* view of it: which boxes exist in
`starter/`, which do not, and what each missing one binds. It carries no design reasoning,
so it cannot drift from the evidence; if it disagrees with the docs repo, the docs repo
wins and this file is what needs rewriting.

Synced from the v5 rendering of that diagram (30 Aug 2026, supersedes v4 of 28 Aug).
`docs/hard-rules.md` is the normative companion — that file says what `starter/` *must*
do; this one says what it currently *does*.

## The three layers, by blast radius

Layer membership encodes the **runtime safety contract** — whether a component runs every
turn and is forbidden to raise. It does **not** encode build order. Several Layer 1
components below are designed and unbuilt.

| Layer | Members | If it fails |
|---|---|---|
| **1 — offline core** | BM25 index, intent classifier Tier 1, constraint ledger, slot state, cross-encoder rerank, response builder + overlap gate | Swallowed into an empty response — that turn scores 0. Raise on *every* turn and the whole run is exactly `0.00000`. |
| **2 — adaptive orchestration** | yield meter, retirement register, yield trace | Falls back to the fixed schedule. Whole band is worth ≈0.004. |
| **3 — optional, deletable** | Tier 2 semantic fallback, LLM escalation over the CE | Caught by its own try/except; Layer 1's ordering stands. Score unchanged. |

The per-turn `try/except` is at `evaluator/local_evaluator.py:240-243`, **inside** the turn
loop. One throw costs one turn, not the run — the run only zeroes if the failure is
systematic (missing model file, bad import, a ranking call reaching the network on a rig
where it is disabled). Those are the failures worth designing around, because they throw
on every turn of every session and produce no traceback, just a plausible-looking bad
number.

`__init__` and `reset()` are the exception: the evaluator does **not** wrap them
(`local_evaluator.py:306` and `:228`). A raise there ends the run before session 1.

## Layer 1 — component by component

| Component | State here | Where | Gap |
|---|---|---|---|
| Product index · BM25 | **Shipped** | `starter/retrieval.py` | None for Layer 1. Does not expose IDF, which Layer 2's yield meter needs. |
| Intent classifier · Tier 1 | **Partial** | `starter/ledger.py` → `_CONTENT_FREE_PATTERNS` | Two anchored patterns covering the three content-free frames. It is *not* the full frame decode: no decline split on the token `additional` (hard rule 4), no burned-ask signal, no override detection, no scenario-type readout. |
| Constraint ledger | **Shipped** | `starter/ledger.py` → `SessionState` | None. Append-only, verbatim, and it *is* the query — matches the design exactly. |
| Slot state | **Absent** | — | No typed slot dict exists. Blocks the G5 contradiction check. Scheduling-only by design: it must never reach retrieval. |
| Cross-encoder rerank | **Absent from `starter/`** | measured only in `bakeoff/part4_rerank.py` | Model load, its own try/except, and the "keep BM25's order" bypass edge are all unbuilt. Also a packaging change: `requirements.txt` is comments-only today and a bundled local checkpoint ends that. |
| Response builder + overlap gate | **Partial** | `starter/agent.py` → `_validated`, `_limit` | Builder and schema coercion ship. The verbatim-overlap gate is absent. "Never repeat a shown product" is absent — no `shown` set exists. The override guard that puts pre-override products back in play is absent. |

### The ask channel (gate G3), inside the same turn loop

| Item | State here | Gap |
|---|---|---|
| Seven-slot schedule | **Six of seven** | `starter/scheduler.py::FIXED_SCHEDULE` is the clean six. `budget` is the decided seventh slot (+0.0277 scrubbed, +0.0000 leaky) and is not in it. |
| Free turns 8–10 | **Absent** | `next_attribute()` returns `None` once the six are exhausted, so every session emits a null ask from turn 7 on. Not a literal breach of hard rule 1 — no askable attribute *remains* under a six-slot schedule — but the seventh slot, the burned-ask re-ask and this fallthrough exist precisely to make one always remain. None of the three are built. Measured cost of the status quo: 160 null turns, 0 constraints gained, 0 hits. |
| Burned-ask re-ask (hard rule 5) | **Absent** | `SessionState` has no burned slot. Re-asking recovers constraints in 25 of the 40 sessions that burn an ask. |
| Retirement register | **Absent** | `state.retired` does not exist. Its trigger is the customer's words, not a yield number. |

Turns 8–10 are **not** a fixed order. Each free turn independently runs the same
fallthrough test against the state at that moment. Writing them as "turn 8 does this,
turn 9 does that" is the most common misreading of the design.

## What v5 changed that binds this repo

Seven gaps between v4 and the mermaid it was drawn from. Most of them touch `starter/`:

1. **The cross-encoder rerank moved from optional into the offline core** (30 Aug). It was
   drawn optional on the assumption that reranking might need a network call; the measured
   arm is a bundled local model with no network at any point. Its try/except and its
   BM25-order bypass edge both stay exactly where they were — the move is about which
   contract it is held to, not about dropping its fallback.
2. **The rerank's open axes are three, not one.** The ≈1.2 s/turn is a **timeout risk, not
   a score cost** — Efficiency is turn-based and wall-clock never enters TechnicalScore.
   Still open alongside it: which checkpoint, and the rubric reading.
3. **`budget` is the seventh askable attribute** and was never being asked. Decided, unbuilt.
4. **"Never repeat a shown product"** is now the return rule. A running session has already
   failed on everything it returned, so excluding those cannot drop the target — *except*
   across an intent override, where the evaluator's hit check was suppressed and those
   products were never tested. The override guard is not optional; skipping it discards
   roughly 5–6 sessions permanently. Designed, unmeasured, and the biggest untested idea
   on the page.
5. **The verbatim-overlap gate ships regardless** — a substring check over text already in
   memory, no model, no measurable cost. Build it first: it is also how the 94.5% claim
   gets measured per session.
6. **Dense fusion is rejected, not gated.** BM25 is the sole retrieval route. Do not
   re-propose without new evidence.
7. **`ITC` / intent trajectory is deleted from the diagram**; the two-tier intent
   classifier is what ships. Already reflected in `docs/hard-rules.md` → Naming.

Tier 2 of the intent classifier remains **not approved to build**, and its implementation
is an untested XOR — rung 3 (embedding nearest-centroid) or rung 4 (fine-tuned encoder
head), one slot, one winner, decided on held-out paraphrase numbers. Rung 4 needs a
training run, which this project's sandbox cannot do (no network to fetch weights).

## Numbers — read the bracket, and check the file

Local scores are inflated: the vendored simulator builds the "hidden" customer preferences
out of the target product's own listing, and `bakeoff/overlap.py` measures **94.5%** of
disclosed constraint strings as verbatim substrings of it. Leaky and scrubbed are the two
ends and which end is being quoted has to be said out loud. The verified offline
TechnicalScore is **0.722818** (legacy ledger, `bakeoff/results-part4.json` → `legacy.k0`)
— not the v2 brief's 0.75040, which is a different agent on a different rig.

**Open discrepancy, raised 30 Aug 2026 — do not propagate the v5 figures for the rerank or
for dense fusion into this repo until the docs repo reconciles them.** The v5 page quotes
the cross-encoder at **+0.047 over BM25's top-50** and dense fusion at **−0.206 top-100 /
−0.065 top-50**. Neither pair appears in this repo's committed bake-off artifacts, and for
the rerank the *depth* is wrong in a way that matters — top-50 is the weakest arm measured,
and it is the one whose CI includes zero:

| Arm | Ledger | Depth | TechnicalScore delta | CI excludes zero? |
|---|---|---|---|---|
| CE rerank | current | N=10 | +0.029102 | yes |
| CE rerank | current | **N=20** | **+0.041746** | **yes** |
| CE rerank | current | N=50 | +0.017252 | **no** |
| CE rerank | legacy | N=10 | +0.023950 | yes |
| CE rerank | legacy | N=20 | +0.021835 | no |
| CE rerank | legacy | N=50 | +0.010744 | no |

Source: `bakeoff/results-part4.json`. On the ESCI rig (600 human-authored queries, depth
20, `bakeoff/results-followup-ce-esci.json`) the same cross-encoder moves MRR@10 from
0.6686 to 0.7173 — **+0.0487**, which is the closest thing in this repo to the quoted
"+0.047", but it is a different metric on a different rig at a different depth.

Dense RRF fusion, `bakeoff/results-part3.json`: −0.111047 (minilm/current, top-50) and
−0.155405 (top-100) on TechnicalScore; −0.199823 and −0.215349 on mean paired RR delta.
The *conclusion* — fusion loses, on every arm and both rigs — is unaffected. Only the two
quoted figures are unaccounted for.

## Distance to the offline core

Two of six Layer 1 components are complete, two are partial, two are absent; four items in
the ask channel are unbuilt. Ordered by the design's own sequencing (free things first,
then the thing everything else is measured against):

1. **Verbatim-overlap gate** — free, deterministic, ships regardless.
2. **`budget` as slot 7** — one tuple entry plus an ask template; decided already.
3. **Free-turn fallthrough + burned-ask re-ask** — ends the 160 null turns. Needs a burned
   slot on `SessionState`.
4. **Never-repeat-a-shown-product + the override guard** — needs a `shown` set and override
   detection; measure the rank readout across the 40 sessions that miss before building it.
5. **Full Tier 1 frame decode** — including the decline split on `additional`, which the
   current single pattern collapses.
6. **Slot state + the G5 contradiction check** — ceiling ≤0.0078; built for coverage, not score.
7. **Cross-encoder rerank** — the only one that changes the dependency story, and the only
   one blocked on an unresolved question (checkpoint, depth, timeout).
