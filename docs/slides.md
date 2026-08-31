# Slide content — TikTok TechJam 2026, Track 4

**Shopping Copilot: AI Conversational Search and Recommendations**

This file is the *content* of the deck, not the deck. Each slide below is a spec:
what it says, what goes on it, what to draw, what the presenter says, and where
every number came from.

---

## Rules for whoever renders this

1. **Every figure on a slide is copied verbatim from its `Source:` line.**
   Invent nothing. Round nothing. If a number is not in this file, it does not
   go on a slide.
2. **Never quote a score without its bracket.** Local scores in this project are
   inflated by a leak in the organizer's simulator. `0.872057` alone is a wrong
   answer; `0.872057 leaky / 0.497383 scrubbed` is the right one. This rule is
   `CLAUDE.md`'s and it is not negotiable, including in speaker notes.
3. **`+0.047` must not appear on any slide** except inside Appendix A4's
   do-not-quote note. `docs/todo.md` item 4 records it as unreconciled.
4. Slides 1–14 are the deck. A1–A9 are appendix / Q&A ammunition — render them,
   but they sit after the close.
5. Keep the visual language consistent: one accent colour for *shipped*, one for
   *measured and rejected*, one for *held open*. Three states, used everywhere.

---

# Main deck

---

## Slide 1 — Title

**Says:** This is a conversational shopping agent that finds a hidden product in
ten turns without a language model.

**On the slide:**
- Shopping Copilot — Track 4
- *Ten turns. Fifty thousand products. No model.*
- Team name, members
- Repo link

**Visual:** Nothing busy. The three-clause thesis line is the visual.

**Notes:** "Our agent talks to a shopper for up to ten turns, asks one useful
question each turn, and has to have the exact product they had in mind inside
its top ten. We do that with no LLM, no embeddings, and no network — and I'll
show you the numbers and then show you why we made that choice on purpose."

**Source:** `README.md` §What ships; catalog size from `docs/windows-dev-setup.md` §1.

---

## Slide 2 — The problem

**Says:** The task, and the exact formula we are scored on.

**On the slide:**
- A simulated customer has one specific product in mind and never names it.
- Each turn: we return up to 10 products **and** ask one clarifying question.
- The session ends the moment our list contains their target. Hard stop at 10
  turns — exceed it and the session scores zero.
- `TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency`
- `Efficiency = clip((11 − MTTC) / 10, 0, 1)` — turn-based. **Wall-clock never
  enters the score.**
- 50,000-product frozen Amazon catalog; 200 public sessions; 800 held out.

**Visual:** A ten-cell turn strip, cells 1–10, with a "hit" marker landing at
turn 3. Underneath, the score formula with each of the three terms tinted.

**Notes:** "Three things to hold onto. One: we're graded on whether the right
product is in the top ten, how high it ranks, and how few turns it took. Two:
efficiency is measured in *turns*, not seconds — latency is a timeout risk, not
a score cost. Three: eight hundred of the thousand sessions are held out, so
anything that only works on the public set is worth nothing."

**Source:** `docs/todo.md` framing point 2; `docs/agent_api_contract.json`
(`turn` max 10, `top_k` const 10); Lark §4.3 Limits; Lark §4.4 Competition Data.

---

## Slide 3 — Results

**Says:** We are far above the provided baseline, and we report both ends of our
uncertainty rather than the flattering one.

**On the slide:**

| | leaky (upper bound) | scrubbed (lower bound) |
|---|---|---|
| **Ours (`src/`)** | **0.872057** | **0.497383** |
| Our first-generation system | 0.692586 | 0.198439 |
| Organizer weak-BM25 baseline | 0.106710 | — |

Leaky detail: hit@10 **0.9950**, MRR **0.705855**, MTTC **2.86**
Scrubbed detail: hit@10 **0.6600**, MRR **0.251944**, MTTC **6.41**

**Visual:** A horizontal range bar per system — the bar spans scrubbed→leaky, so
the *width* is the uncertainty. The organizer baseline is a single tick far to
the left. This one picture makes slide 4 almost unnecessary.

**Notes:** "Two numbers, not one, and I'll explain why on the next slide. Against
the organizer's own starter agent at 0.107 we're between 4.7× and 8.2× better
depending on which end you read. Against our own first attempt we gained
0.179 at the top end — and 0.299 at the bottom end."

**Source:** `results_src.md` rows 2026-08-31 02:21 and 02:11 (`6e4c32b`);
`README.md` §Results; baseline `0.106710` from `scripts/evaluate_src.py:92`.

---

## Slide 4 — Why there are two numbers

**Says:** The local evaluator leaks the answer, so any single local score is an
upper bound. We measured the leak instead of ignoring it.

**On the slide:**
- `public_set.jsonl` ships **no real `intent_card`**, so the evaluator builds the
  "hidden" customer preferences out of **the target product's own listing** and
  recites them back turn by turn.
- **94.5%** of disclosed constraint strings are exact substrings of the target's
  indexed text.
- So we score twice: **leaky** (as shipped) and **scrubbed** (that channel
  removed). The organizer's held-out set should land between them.
- The tell that this is real and not a measurement artifact:

| | leaky | scrubbed |
|---|---|---|
| our gain over our own baseline | +0.179471 | **+0.298944** |

**Visual:** Two side-by-side session transcripts, identical layout — left one
with the leaked substrings highlighted in the customer's replies *and* in the
product listing; right one with them stripped.

**Notes:** "This is the slide I'd want to see from anyone claiming 0.87. The
simulator hands us the answer in the customer's own words. When we take that
channel away, our score drops a lot — but our *lead* over our previous system
gets bigger, not smaller. A measurement artifact would do the opposite. That's
what convinced us the architecture is doing real work."

**Source:** `CLAUDE.md` §Two things that will mislead you; `bakeoff/overlap.py`
(the 94.5% instrument); `README.md` §Results.

---

## Slide 5 — What it costs to run

**Says:** Zero models, zero dollars, zero tokens, twenty milliseconds a turn,
and it runs with the network physically revoked.

**On the slide:**

| Disclosure | Value |
|---|---|
| Models used | **None.** No LLM, SLM, or neural model on the graded path |
| Network access | **None required, none attempted** — enforced by an AST test over every `src/` module |
| API credentials | None. The system cannot fail for want of a key |
| Token usage reported | `0` prompt, `0` completion, every turn — truthfully |
| Estimated model cost | **$0.00** |
| Latency | index build **1.16 s** once; **~19 ms per turn**; 200 sessions / 571 turns in **9.7 s** |
| Dependencies | **Zero.** `requirements.txt` is comments only |
| Offline verified | Full 200-session run under `sandbox-exec` with networking revoked — **scores identical** |

**Visual:** A cost/latency comparison bar against a notional LLM-per-turn
pipeline, with our bar essentially at the axis. Keep the comparison honest and
label it as illustrative.

**Notes:** "This is the Feasibility slide. There is nothing to provision, nothing
to pay for, no key to rotate, and no rate limit to hit. We verified the offline
claim by running the whole thing inside a sandbox with networking revoked and
confirming the scores were byte-identical — and we ran a control probe in the
same sandbox to prove the block was real and not vacuous."

**Source:** `README.md` §Model choice, cost, tokens, latency, network;
`tests/test_src_no_network.py`; `scripts/no-network.sb`.

---

## Slide 6 — Architecture: one pass, nineteen stages

**Says:** Every turn is one deterministic pass. No branches on model output, no
retries, no agent loop.

**On the slide:**

```
decode → contradiction check → ledger append → query → BM25
      → rerank seam → overlap gate → never-repeat → ask policy → schema coercion
```

- 18 modules, **3,056 lines**, standard library only
- Every stage is individually `try`/`except`-wrapped, and the whole turn once more
- Three stages are **order-only** and permutation-checked: a stage that silently
  drops a candidate has performed retrieval, and is rejected

**Visual:** The pipeline left-to-right. Shade the two stages that are inert
seams (rerank, semantic fallback) differently from the eight live ones — this
sets up A4/A5 without a word.

**Notes:** "One pass, ten stages on the slide, nineteen in the code. Nothing here
loops, nothing retries, nothing calls out. The important property is the one you
can't see: three of these stages are allowed to *reorder* the candidate list and
forbidden to *change* it, and we check that at runtime by comparing the multiset
of product IDs before and after."

**Source:** `src/pipeline.py:78-149`; module count and line count from `src/`.

---

## Slide 7 — Retrieval

**Says:** BM25 over an in-memory SQLite FTS5 index. Boring, fast, and the only
route — because we measured the alternatives.

**On the slide:**
- In-memory SQLite FTS5, built once at construction over 50,000 products
- Six indexed fields, weighted: `title` **6.0**, `categories` **4.0**,
  `features` 2.5, `details` 2.5, `store` 1.5, `description` 1.0
  — the organizer's own weight vector, carried over unchanged
- Query = up to **40** unique stopword-filtered terms, each **quoted as a
  phrase** and OR-joined, so a stray FTS5 operator in a customer reply cannot
  break the query
- Two-phase: `search()` pulls 300 IDs cheaply; `hydrate()` pulls text only for
  the 50 that survive, by rowid
- Construction **never raises** — a malformed catalog line is skipped, not fatal

**Visual:** The two-phase funnel: 50,000 → 300 (ids only) → 50 (hydrated text)
→ 10 (returned). Annotate the cost of each arrow.

**Notes:** "Category is a first-class weighted retrieval field here, at 4.0 —
second only to the title. The phrase-quoting matters more than it looks: a
customer saying the word 'or' or 'near' would otherwise be parsed as an FTS5
operator and take the query down. And retrieval is two-phase because pulling
concatenated product text for three hundred candidates every turn is what would
actually make this slow."

**Source:** `src/retrieval.py:29-34` (fields and weights), `:305-326`
(`_match_expression`), `:246-301` (`hydrate`); `src/types.py:69-71`.

---

## Slide 8 — The idea that carries the score

**Says:** The concatenation of everything the customer literally said *is* the
search query. Not an input to building one — the thing itself.

**On the slide:**

| | After turn 1 | After turn 2 |
|---|---|---|
| Customer said | "A key requirement is: black leather." | "…what matters is: formal." |
| Slot state | `{material: leather, color: black}` | `{…, style: formal}` |
| **Ledger** | `["A key requirement is: black leather."]` | `[…, "For that, what matters is: formal."]` |
| **Query** | `"black leather"` | `"black leather formal"` |

- Append-only. **No deletion method exists** — enforced by an AST test that
  scans the module for any function named like `clear`/`remove`/`pop`/`reset`.
- The typed slot view exists to catch contradictions and drive the override —
  and is **architecturally barred from touching retrieval**, asserted by test.

**Visual:** The table above, with a second panel: a red "parser bug" injected
into the slot row, and the Query row visibly unchanged.

**Notes:** "Here's the safety property. Suppose our parser mis-tags 'black
leather' as colour-only. The slot state is now wrong — but the query is
unaffected, because the word 'leather' is sitting in the raw string whether or
not the parser ever tagged it. A parsing bug can corrupt *which question we
ask*. It cannot corrupt *what we search*. That's the concrete reason structured
slot parsing measured plus zero point zero zero zero for us: it was never wired
into search in the first place."

**Source:** `src/ledger.py:35-95`; `src/slots.py:3-12`;
`tests/test_src_layering.py::test_ledger_defines_no_erasing_api`; the worked
example is from the design of record ("Statement 4 Architecture v5").

**Presenter note, do not put on the slide:** today `SlotState` is consumed only
by the override path (`slots.apply_override:148-153` reads it to detect the
contradiction). Ask selection reads the ask registers, not the slots. So say
"slots catch contradictions", not "slots decide what we ask" — the second is the
design intent, not the current wiring.

---

## Slide 9 — Understanding the customer

**Says:** The customer speaks in a closed set of sentence shapes, so we *decode*
intent exactly rather than estimating it.

**On the slide:**
- Eight anchored regexes against the eight templates the simulator emits — a
  decode, not a classifier. **A classifier cannot beat a substring check that is
  already at 100%.**
- The load-bearing detail — two replies that differ by **one token** and mean
  opposite things:

| Reply | Token | Means | We do |
|---|---|---|---|
| "I don't have **a** preference for X" | — | bucket never opened | **re-ask X later** |
| "I don't have an **additional** preference for X" | `additional` | bucket provably empty | **retire X permanently** |

- Two separate patterns, never one alternation. Collapsing them is a real bug we
  found in our own first-generation system.

**Visual:** The two sentences stacked, near-identical, with `additional`
highlighted, and two divergent arrows: one looping back into the question queue,
one into a "retired" bin.

**Notes:** "This is the single highest-leverage line of code in the project. In
the evaluator, the first reply returns *before* the constraint filter runs — the
customer just declined to answer. The second returns *after* the filter found
nothing — that attribute is genuinely empty. Same eight words apart from one.
Our first system collapsed them into one regex and permanently retired
attributes the customer would happily have answered later."

**Source:** `src/frames.py:19-24`, `:82`, `:88`; `docs/hard-rules.md` rule 4;
`src/askpolicy.py:53`, `:133-134`.

---

## Slide 10 — Deciding what to ask

**Says:** Seven fixed questions, then an adaptive ladder — and never a null
question, ever.

**On the slide:**
- Turns 1–7, fixed order: `material → feature → color → style → size →
  use_case → budget`. Seven because those are the only labels the evaluator's
  own classifier can return.
- Turns 8–10: **not** a fixed order. Each free turn re-runs a fallthrough ladder
  against the state at that moment:
  1. re-ask a **burned** question (one that was asked but never answered)
  2. re-mine an attribute whose reply may have been truncated
  3. the `brand` / `category` hedge
  4. first attribute not yet retired
- **Never `null`.** Measured: the 160 null turns our old system sent across turns
  7–10 gained **0 constraints** and produced **0 hits**.
- **Never `other`** — it is absent by construction, not by a filter.

**Visual:** Ten turn-boxes. Boxes 1–7 are a straight fixed track; 8–10 fan into
the four-rung ladder. Mark the two "burn" events with a small flame.

**Notes:** "Two questions get burned in this harness. In a boundary session the
customer refuses; in an override session the evaluator never even asks the
customer, so our question goes into the void. Re-asking those recovers a real
constraint in 25 of the 40 sessions where it happens — eight of ten boundary,
seventeen of thirty override. And the cost of a re-ask that finds nothing is
exactly zero, because we were going to spend that turn anyway."

**Source:** `src/types.py:40-42` (`FIXED_SCHEDULE`); `src/askpolicy.py:211-268`
(the ladder); `docs/hard-rules.md` rules 1 and 5.

---

## Slide 11 — Never show the same product twice

**Says:** In a running session, everything still on screen is confirmed wrong —
so we push it down. But we never remove it.

**On the slide:**
- The session ends the instant our list contains the target. So if the session is
  still running, everything we showed is **proven not to be the answer**.
- `partition()`, never `filter()` — the shown set is an **ordering preference**,
  and the top-10 is always full. Backfill from already-shown if the pool drains.
- Effect: a losing session walks up to **100 distinct products** instead of
  re-issuing the same failed ten.
- The override guard: in `intent_override` sessions the evaluator's hit check is
  off for the first turns, so those products were never really tested —
  everything shown before the override goes **back in play**.

**Visual:** Two filmstrips of ten product tiles across turns 1→4. Top: the same
ten every turn. Bottom: forty distinct tiles. Then a third strip showing the
override moment resetting the greyed-out tiles back to live.

**Notes:** "The guard is the subtle half. In an override session the evaluator
doesn't start checking for a hit until the customer changes their mind at turn
three or four. So products we showed at turn one were never actually tested
against the target — treating them as eliminated would throw away good
candidates. We detect that session type at turn one from the opening sentence
shape and simply switch recording off until the override lands."

**Source:** `src/shown.py:3-10`, `:18-32`, `:125-145`;
`src/pipeline.py:232-263`.

---

## Slide 12 — Engineering for the silent zero

**Says:** The evaluator turns a crash and a malformed response into the same
silent zero. That single fact shaped the whole codebase.

**On the slide:**
- An exception → zero, silently. A schema-invalid dict → zero, **just as
  silently**. There is no error path worth taking.
- So: `respond()` **never raises** — it returns a schema-valid empty response.
- Worse: `__init__` and `reset()` are **not** wrapped by the evaluator. A raise
  in either kills **all 200 sessions**, not one turn. Both are guarded.
- Every outgoing payload is **coerced**, field by field, to its schema-valid
  form — a bad `ask_attribute` never costs you the recommendations.
- **390 tests** over 21 files — **5,821 lines of tests against 3,056 lines of
  source, 1.9×.** Two of them are architecture assertions: the layering test
  proves retrieval can't import the slot layer; the no-network test AST-scans
  every module for networking imports.

**Visual:** A blast-radius diagram: concentric rings labelled *one stage → one
turn → one session → the entire run*, with the guard that stops each.

**Notes:** "This is a defensive-engineering slide, and it's the one I'd defend
hardest. Most failure modes in this harness are invisible — you don't get a
traceback, you get a plausible-looking lower number. So we made the boundary
coerce rather than validate, we guarded the two constructors the evaluator
doesn't wrap, and we wrote tests that assert the *architecture*, not just the
behaviour."

**Source:** `CLAUDE.md` §The rule that governs every change here;
`src/contract.py:3-7`; `src/agent.py:3-11`; `tests/` (390 methods, 21 files).

---

## Slide 13 — Method: the score is not the objective

**Says:** We declined three changes that would have raised our score, and kept
one that lowered it.

**On the slide:**

| Change | Measured | Decision |
|---|---|---|
| Emit `other` to bypass the constraint filter | **+0.004** | **Declined** — harness-gaming |
| Return top-1 then top-10 to game MTTC | **+0.018602** | **Declined** — harness-gaming |
| Ledger content-free filter | **−0.030232** | **Kept** — it is correct |

- `TechnicalScore` is one input to **35 of 90 points**. Innovation & Problem
  Insight (20), Impact & Relevance (20), and Feasibility & Practicality (15)
  **never read it at all.**
- And "Technical Execution 35%" is a human judgement of engineering
  fundamentals — *not* the automated score.
- So a decision justified only by a score delta is not finished.

**Visual:** A 90-point stacked bar, with the slice `TechnicalScore` actually
touches shaded and the other 55 points left plain.

**Notes:** "We wrote this down as rule zero before we needed it. The `other`
attribute bypasses the evaluator's constraint filter and hands back two
undisclosed constraints every single time — it was the highest-scoring option on
the board and we turned it down, because it's an exploit of the simulator, not a
shopping agent. Same with returning one product then ten to shave the
mean-turns-to-conversion. If we'd optimised the number we'd have shipped both."

**Source:** `docs/hard-rules.md` rule 0 and rule 2; `docs/todo.md` framing
point 2; Lark §4.6 (the five weights).

---

## Slide 14 — Limitations, and what's next

**Says:** Here is what we know is weak, stated before anyone asks.

**On the slide:**
- **Both numbers are this simulator.** The bracket *direction* is our signal; the
  absolute values are not portable to the held-out set.
- **Intent decode is exact against the current phrasing.** Paraphrase it and
  frames degrade to "content-bearing unknown" — we still search, we just stop
  knowing which attribute was answered. A semantic fallback is built as a typed
  seam and deliberately left off.
- **We never read the long-term user profile.** The harness hands us one per
  session, but session IDs are random UUIDs with no identity behind them, so
  cross-session memory would be mixing up different shoppers.
- **One measurement we still owe:** the never-repeat rule shipped ahead of the
  rank readout that was supposed to justify it. Its cost is provably zero, so we
  shipped it early — but the number is a debt, and we've said so in writing.
- **Next:** paraphrase-robustness harness → the semantic fallback → then, and
  only then, the cross-encoder.

**Visual:** Keep it text. This slide's credibility comes from being plain.

**Notes:** "The one I'd most want to fix is paraphrase robustness. Our intent
decode is exact against the eight sentence shapes this simulator emits, and the
organizer has explicitly reserved the right to reword them. Our fallback for
that is built and switched off, because we couldn't measure it honestly without
writing the test data ourselves — which would have made the result circular."

**Source:** `README.md` §Limitations; `docs/todo.md` items 1 and 7;
`src/session.py:1-16`; `evaluation-data/README.md` (the circularity argument).

---

# Appendix

---

## Slide A1 — The four pillars, clause by clause

**Says:** Here is our work mapped onto your problem statement, including the two
things we chose not to build.

**On the slide:**

| Pillar | Clause | Us |
|---|---|---|
| I | Instantly detect intent | **Built** — decoded exactly at turn 1 from the opener's sentence shape |
| I | Dual retrieval track (Buying vs Browsing) | **Refused, with an argument** — see A6 |
| I | Keyword + category retrieval | **Built** — `categories` weighted 4.0, second only to title |
| I | Vector similarity | **Measured, rejected** — see A2 |
| I | Semantic ranking stage | **Built, non-LLM** — two live reordering stages after BM25: the shown-set partition and the verbatim-overlap gate |
| II | Information accumulation | **Built** — append-only ledger + incremental slots |
| II | Intent override / slot erasure | **Built** — classify → diff → clear the slot |
| II | Structured proactive clarification | **Built** — a templated question on every single turn |
| II | Guide convergence | **Built** — stops mining once the card is provably full |
| II | Over-generality pool cutoff | **Not built** — our truncation is unconditional |
| III | Short-term session state from history | **Built** — ledger, slots, two ask registers, shown set |
| III | Long-term user profile | **Not read** — see slide 14 |
| III | Adaptive orchestration | **Built, deterministic** — turns 8–10 re-plan against live state |
| IV | Hit@K / MRR / MTTC | **Built and exceeded** — plus the leak bracket you didn't ask for |

**Visual:** The table, with the three states colour-coded to match slide 6.

**Notes:** "We'd rather show you this than have you wonder. Two clauses we did
not build. Both were built, measured, and removed — and the next few slides are
the receipts."

**If pressed, three honest refinements:** (a) the overlap gate is live but
*data-dependent* — a browsing opener carries no constraint segments, so on that
turn the gate passes the list through unchanged; (b) the difference between a
buying and a browsing session is emergent from what the customer said, not a
scenario branch in the code — `session.scenario` is written and never read;
(c) pool truncation is fixed at 300/50/40, never conditional on result breadth.

**Source:** verified against `src/` this session; file:line references in
`docs/` and the plan file.

---

## Slide A2 — Rejected: dense retrieval

**Says:** We built the vector arm the statement asks for. It made us worse,
twice.

**On the slide:**
- BM25 + MiniLM RRF fusion: **−0.206** at top-100, **−0.065** at top-50 — two
  independent rigs, same direction.
- Why: the target is **BM25's rank 1 in 87 of 176 hit sessions**, but sits around
  **dense rank 72**. Blending dilutes a strong list with a weak one.
- BM25 is therefore the sole retrieval route — a measured decision, not a
  default.

**Visual:** Two rank distributions overlaid — BM25's mass piled at rank 1,
dense's smeared across the mid-hundreds — with the fused result landing between.

**Notes:** "The intuition that dense retrieval helps is a good one and it's wrong
here, for a specific reason: the customer is quoting the product listing almost
verbatim. When the query and the document share literal strings, lexical search
is already near-optimal and semantic similarity only adds noise."

**Source:** `src/retrieval.py:4-6`; `README.md` §Model choice.

---

## Slide A3 — Rejected: phrase retrieval, and how we caught it

**Says:** A change that looked like a large win was a measurement artifact, and
we have the instrument that proved it.

**On the slide:**

| Measurement | Arm | Result |
|---|---|---|
| Local public set (leaky) | `phrase_plus` | **+0.0588** ΔScore, CI 0.0316–0.0879, excludes zero |
| ESCI, 600 **human-written** queries | unigrams (shipped) | recall@10 **0.8233** |
| ESCI, 600 human-written queries | phrases + unigrams | recall@10 **0.8250** |
| ESCI, 600 human-written queries | phrases only | recall@10 **0.2400** |

> A large gain on a leaky local set and **+0.0017** on real human queries is the
> signature of a measurement artifact, not an improvement.

**Decision: do not ship. Never cite +0.0588 without the ESCI row.**

**Visual:** Two bars side by side — the leaky gain tall, the ESCI gain
essentially invisible. Let the asymmetry do the talking.

**Notes:** "This is the one I'm proudest of, and it's a negative result. Phrase
matching wins big when the customer is quoting the listing — which is exactly
the leak. Against six hundred queries written by actual Amazon customers it
bought us seventeen ten-thousandths of a point. We'd have shipped it on the
local number alone."

**Source:** `docs/todo.md` item 8; `bakeoff/README.md`; ESCI is arXiv:2206.06588,
Apache-2.0.

---

## Slide A4 — Held open: the cross-encoder

**Says:** It works, and we still haven't shipped it — because we can't price the
timeout risk.

**On the slide:**

| Rerank window | mean Δ TechnicalScore | 95% CI | excludes zero | s / reranked turn |
|---|---|---|---|---|
| top-10 | +0.0291 | 0.0147 – 0.0443 | yes | 0.22 |
| top-20 | +0.0417 | 0.0115 – 0.0736 | yes | 0.43 |
| top-50 | +0.0173 | −0.0299 – 0.0631 | **no** | 1.25 |

- Corroborated on real human queries: ESCI recall@10 **0.8233 → 0.845**, MRR@10
  **0.6686 → 0.7173**.
- Blocked on three open axes: which checkpoint (never compared), the per-turn
  timeout the organizer has not published, and the rubric reading.
- **Reconciliation note for the presenter only:** an older `+0.047` figure
  circulates in our README. It pairs the *best* delta with the *slowest* arm's
  cost. **Do not put it on a slide.** Quote the table.

**Visual:** The three CI bars against a zero line, with the top-50 bar visibly
crossing it. Cost annotated on each.

**Notes:** "It's a real gain and it survives on human-written queries, which is
more than phrase matching managed. What stops us is that the organizer hasn't
published a per-turn timeout, and the arm that gains most costs over a second a
turn. We'd rather ship at nineteen milliseconds and be certain than gain three
points and risk a zero."

**Source:** `docs/todo.md` item 4; `bakeoff/results-part4.json`.

---

## Slide A5 — Held open: an LLM in the ranking stage

**Says:** The blocker isn't cost or network. It's that a generative model can't
abstain.

**On the slide:**
- If a language model ever enters this system it goes in **ranking**, never in
  intent parsing. Today it is not merely flagged off — `src/llm_rerank.py` is
  **never imported by any shipped module**. Only the tests reference it.
- Why it's still off: a generative model **emits tokens and has no calibrated
  score to threshold** — so it cannot say "I don't know" and hand back to BM25.
  That is the disqualifier, and no measurement changes it.
- Blast radius is already bounded: our rerank guard **discards any result that
  isn't a permutation of its input**, so a broken or hallucinating model costs us
  BM25's ordering and nothing else.
- Honest expectation: against this simulator it would **barely ever fire** — the
  non-quoting case is about **5.5%** of the local set.

**Visual:** The rerank stage with a guard gate on its output, and a bypass edge
routing straight back to BM25's order.

**Notes:** "We're not avoiding LLMs out of principle — we designed the seam for
one and specified exactly what it would have to disclose. We're avoiding it
because the layer we'd put it in needs to be able to decline, and generative
models are bad at declining."

**Source:** `docs/todo.md` items 2 and 3; `src/rerank.py:66-104`
(`safe_rerank`); `src/llm_rerank.py:42`.

---

## Slide A6 — Refused: the Buying/Browsing trajectory labeller

**Says:** The dual-track router the statement asks for is right for a real
product and untestable in this benchmark — so we argued it instead of stubbing it.

**On the slide:**
- We **do** detect Buying vs Browsing — exactly, at turn 1, from the opening
  sentence shape. That part is free.
- What we didn't build is a **second retrieval track** keyed on it. Two reasons:
  1. The frame decode already gives the per-turn label exactly. There is nothing
     left to classify on top of it.
  2. **There is no drift to detect.** The scenario type is fixed before turn 1
     and the harness changes its mind exactly once, on a timer, at turn 3 or 4.
     A trajectory check has nothing to contradict.
- A real shopper changes direction repeatedly and without warning. **This
  simulator can do it once, on schedule.**
- So the component is right for the product and unexerciseable in the benchmark
  — and naming that is the honest move. **A documented extension, not a stub.**

**Visual:** Two timelines — "real shopper" with several unannounced direction
changes, "this simulator" with exactly one, marked with a clock icon.

**Notes:** "This is the clause we most obviously don't satisfy literally, and I
want to be straight about why. We could have shipped a dormant routing branch
that never fires and pointed at it. Instead we're telling you it's the right
idea for a production shopping agent, that this benchmark cannot exercise it,
and that shipping a stub to look thorough is exactly what a feasibility score
should punish."

**Source:** design of record, "Statement 4 Architecture v5", *Intent trajectory
— demoted, and deliberately so*; `docs/hard-rules.md` rule 7.

---

## Slide A7 — How we decided, before we knew the answers

**Says:** We pre-registered our ship criteria and committed them before the
experiments ran.

**On the slide:**
A change ships only if **all four** hold:
1. Paired-bootstrap 95% CI on Δ **excludes zero**
2. Point estimate exceeds **+0.020**
3. At most **5% of sessions (10 of 200)** regress — else the gain must exceed
   **3 sd (+0.070)**
4. Runs offline and in **≤1.0 s per turn**

Plus two facts that make the above meaningful:
- **There is no seed to vary.** Three runs under `PYTHONHASHSEED` 0 / 1 / 12345
  give identical metrics — seed variance is exactly **0.000000**. So results are
  bit-identical and any delta is real signal, never noise.
- The substitute for seed variance is a 2000-draw bootstrap over sessions:
  baseline sd **≈ 0.023–0.026**.
- Written commitment made in advance: **"a negative result is the expected
  outcome."**

**Visual:** The four gates as a chain, with A3's phrase arm visibly failing gate
1's real-query check and A4's top-50 arm failing gate 1 outright.

**Notes:** "This file is committed and timestamped before the results it judges.
That's the difference between an experiment and a justification."

**Source:** `bakeoff/part0-decision-rule.md`.

---

## Slide A8 — Per-scenario results, and what green tests don't prove

**Says:** Where we're weakest, and why we don't trust our own test suite as
evidence of quality.

**On the slide:**

**Per scenario — scrubbed arm (lower bound), 200 public sessions:**

| Scenario | n | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.7250 | 0.273924 | 5.575 |
| browsing | 80 | 0.6750 | 0.230109 | 6.513 |
| boundary | 10 | 0.6000 | 0.185397 | 7.200 |
| **intent_override** | 30 | **0.4667** | 0.273743 | **8.100** |

Intent override is our weakest scenario — expected, since those sessions spend
their early turns unscored.

**What a green test run does *not* prove:**
- The fixture catalog has **6 products** against `top_k=10`, so the smoke test
  passes even with a **query-blind ranker**.
- CI names its test modules **explicitly**, so a new test file silently doesn't
  run until it's added.
- `unittest discover` reports `Ran 0 tests … OK` if one `__init__.py` goes
  missing. **We check the count (390) before believing green.**

**Visual:** The scenario table as a small-multiples bar chart; the caveats as
plain text beneath.

**Notes:** "Two honest notes. Override sessions are our weakest because the
scorer isn't watching for the first few turns, which is also why we built the
suppression guard. And we don't cite 'all tests pass' as evidence — we
documented the three ways our own suite can be green and meaningless."

**Source:** `results_src.json` `scenario_metrics` (scrubbed arm — verify:
0.50·0.66 + 0.30·0.251944 + 0.20·0.459 = 0.497383); `CLAUDE.md` §A green test
run proves less than it looks like.

---

## Slide A9 — The demo

**Says:** You can watch a real scored session run, one turn at a time, with the
whole pipeline visible.

**On the slide:**
- Two terminals side by side:
  - **left** — the conversation, one turn per keypress
  - **right** — all nineteen pipeline stages for that turn, as they land
- The customer's replies are **not typed by us**. It is scripted replay: the
  evaluator's own simulated customer drives it from the public set, so what you
  see is a session the scorer would actually have produced.
- Pick any scenario and bracket: `--scenario intent_override --bracket scrubbed
  --seed 7` reproduces the same hard session every time.

**Visual:** A screenshot or short screen capture of the two-terminal layout,
mid-session, with the ledger growing on the right.

**Notes:** "Worth demoing the override scenario specifically — you can watch the
shown-set get released the moment the customer changes their mind, and see the
query keep both the old and new constraints because the ledger never erases."

**Source:** `demo/README.md` on the `worktree-demo-clis` branch. **Dev tooling —
not part of the submission and never on the graded path.**

---

# Presenter's crib sheet

Numbers you will be asked for, in the form you should say them:

| Asked | Say |
|---|---|
| "What's your score?" | "0.872057 leaky, 0.497383 scrubbed — and I'll explain why there are two." |
| "How does that compare?" | "The organizer's baseline is 0.106710. Our own first-generation system was 0.692586 leaky." |
| "What model?" | "None. Zero tokens, zero dollars, no network." |
| "How fast?" | "1.16 seconds to build the index once, about 19 milliseconds a turn." |
| "How big is the code?" | "18 modules, 3,056 lines, standard library only. 390 tests." |
| "Why no LLM?" | "In ranking it can't abstain. In intent parsing a regex is already exact." |
| "Why no embeddings?" | "Measured twice: −0.206 and −0.065. The target is BM25's rank 1 in 87 of 176 hit sessions." |

**Three things never to say:** a score without its bracket; `+0.047`; or "all our
tests pass" as a quality claim.
