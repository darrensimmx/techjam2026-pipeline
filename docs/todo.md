# Pending decisions

**What this file is.** A register of decisions that are deliberately *not* being
made in this build, and of one measurement we owe. It is written for a teammate
who was not in the room: each item says what the decision is, what the options
are, what evidence we already have, what would settle it, and where the code
seam is — so whoever picks it up knows where it plugs in without reading the
whole tree first.

Nothing here is a task list. Several items resolve to "do nothing, on purpose",
and those are recorded exactly as carefully as the ones that resolve to work.

---

## Status of this build

**Shipped and live:** the retrieval spine (BM25 over the accumulated ledger),
the verbatim constraint ledger, the seven-slot fixed ask schedule with its two
registers (retirement and burn), the shown-set / never-repeat rule with the
override guard, the slot-value contradiction diff, the wire-contract coercion
layer, and the overlap instrument (observation only — it never removes a
candidate).

**Present but INERT — a seam with a null implementation behind it:**

| Layer | Module | Flag | Ships as |
|---|---|---|---|
| Tier 2 semantic fallback | `src/semantic.py` | `TIER2_ENABLED = False` | `NullSemanticDecoder` (abstains always) |
| Cross-encoder rerank | `src/rerank.py` | `load_reranker(enabled=False)` | `NullReranker` (identity) |
| LLM ranking escalation | `src/llm_rerank.py` | `LLM_RERANK_ENABLED = False` | `NullLlmReranker` (identity, `usage() == (0, 0)`) |
| Ask-yield adaptive ordering | `src/askyield.py` | `ADAPTIVE_ENABLED = False` | the fixed schedule |

Inert means *inert*: each loader checks its flag **before** it checks for a
dependency, so `pip install -r requirements-optional.txt` changes nothing at
all. Every entry in that file is commented out, because no checkpoint and no
model has been chosen and pinning a version would imply a decision nobody made.
`requirements.txt` stays comments-only, deliberately — the graded path is
standard library only.

**No language model runs anywhere in the shipped system today.** The one place a
language model is even proposed is `src/llm_rerank.py`, which sits in *ranking*.
Intent is never model-backed: Tier 1 is regex, and the Tier 2 fallback behind it
is an **encoder**, not a generative model (item 2 explains why that distinction
is load-bearing rather than stylistic).

---

## Two framing points — every number below depends on them

### 1. Every local score is a bracket, so say which end you are quoting

The vendored simulator has no real `intent_card` for `public_set.jsonl`, so it
builds the "hidden" customer preferences out of **the target product's own
listing** and recites them back turn by turn. `bakeoff/overlap.py` measures
**94.5%** of the simulator's disclosed constraint strings as verbatim substrings
of the target's indexed text.

So a BM25 win on the public set is consistent both with "BM25 is the right
retriever" and with "the simulator hands BM25 the answer key", and the public
set cannot separate those two. Two brackets exist:

- **leaky** — as-shipped, the number the vendored evaluator prints;
- **scrubbed** — `scripts/leak_controlled_benchmark.py`, which patches the leak.

**Never quote a local number without naming its bracket.** A third and much
better instrument exists for retrieval questions: `bakeoff/part5_realqueries.py`
runs the same arms over 600 human-authored ESCI queries, which nobody here wrote
and which are not generated from the target document. Where a local number and
an ESCI number disagree, that disagreement is itself the finding (see item 8).

### 2. TechnicalScore is not the objective

`TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency` is **35 of the
90 points** assessed at the online round, and it is only one *input* to that 35.
The other three rows — Innovation & Problem Insight (20), Impact & Relevance
(20), Feasibility & Practicality (15) — never read it at all.

Consequences, both directions:

- A change that measures **~0 can still be correct.** Thread safety, degraded-mode
  signalling and offline verification are Feasibility evidence; they are scored,
  just not by the formula.
- A change that measures **positive can still be wrong.** Two precedents are
  already on the record: the `other` short-circuit was declined at **+0.004**,
  and clock-gated withholding (top-1 then top-10) was declined at **+0.0186** —
  both on judging risk, not on their numbers. A third runs the other way: the
  ledger's content-free filter was **kept at −0.030232**, because the score it
  gave up was bought with noise tokens that will not transfer to a private set.

Ranking a change by score delta alone is the failure mode this note exists to
prevent. State the architectural and feasibility read alongside any number.

Also worth carrying into every item below: **wall-clock never enters
`TechnicalScore`.** `Efficiency` is turn-based — `clip((11 − MTTC)/10, 0, 1)` —
so latency is a *timeout risk* and a Feasibility disclosure, never a score cost.

---

## 1. Tier 2 implementation — which "rung" fills the one slot

**DECIDED 1 Sep 2026 — rung 3, potion-8m.** Held-out numbers, 168-item
paraphrase set:

| setup | recovered | wrong | abstained | combined |
|---|---|---|---|---|
| potion-8m / stripped @ 0.52 | 39 | **0** | 109 | 0.3333 |
| mpnet / multi @ 0.31 | 105 | 29 | 14 | 0.7262 |

mpnet recovers more but is wrong 29 times; potion-8m is wrong zero times. Per
this file's own asymmetry rule below, a wrong refusal/exhaustion read
permanently loses a constraint bucket while an abstention costs nothing — so
zero-wrong is the property worth buying even at much lower combined recovery.
`TIER2_ENABLED = True`, `SELECTED_RUNG = "rung3_centroid"`, `src/semantic.py`.
Rung 4 is not built — it needed a training run this sandbox cannot do, and
rung 3 already won on the numbers before that mattered.

**The decision (superseded framing below, kept for context).** Tier 2 is the
semantic fallback that fires only on a Tier 1 `unknown` frame. It exists for
the **private set**, where the organizers have reserved the right to add
natural-language paraphrasing. There is exactly one slot, and exactly one rung
ships in it.

**The options.**

- **Rung 3 — embedding nearest-centroid.** Turn each of the eight known reply
  shapes into a vector once, up front. When an unrecognised reply arrives,
  vectorise it and pick the closest shape. No training, no labelled data, and
  adding a ninth shape means editing a string.
- **Rung 4 — fine-tuned encoder head.** The same encoder with a small trained
  classifier layer on top. Needs paraphrases we have labelled ourselves, and a
  training run.

**What is NOT at issue.** This is **not a speed or size decision**, and framing
it as one wastes the discussion. Both rungs run exactly one encoder pass — the
overwhelming majority of the cost — and what happens after is 8 dot products
versus one linear layer. Neither is measurable. Rung 4 also runs on rung 3's
*frozen* encoder, so it is a small delta on the same infrastructure, not a
second component. Whoever builds rung 3 has built most of rung 4.

**The evidence we already have.**

- Against today's simulator this layer **never fires at all**: every customer
  utterance is one of eight f-strings in `local_evaluator.py`, and Tier 1
  decodes all eight. So there is no local number, and there cannot be one. Any
  measurement has to come from held-out paraphrase data.
- The real question is whether a *learned decision boundary* helps here, or
  whether it just overfits to paraphrases we invented. Nobody knows what the
  organizers' rewording looks like. Rung 4 would therefore be trained on **our
  guesses about it**, while rung 3's anchors are **the evaluator's own literal
  strings**. That argues for rung 3.
- **That is an argument, not a result.** Record it as such.
- Practical constraint: rung 4 needs a training run, and **training is not
  available in this project's sandbox** — there is no network to fetch weights.

**What would settle it.** Held-out numbers on a paraphrase set, **not a
meeting**. Build rung 3 first because it is reachable today; escalate to rung 4
only on evidence that the decision boundary is the thing failing.

Note the sharp constraint on generating that holdout: ESCI's queries were
written by Amazon customers, so reading them can cost you a config overfitted to
600 queries but cannot make the data circular. **A self-authored paraphrase
holdout has no such protection** — reading it *is* the circularity. If one is
generated, it needs a real control (kept out of the working tree, or decrypted
only at scoring time) chosen **before** it is generated, not after.

**Rungs 1–2 are not part of this decision.** A cue lexicon and a fuzzy string
match are cheap, deterministic, and we build them anyway as the floor beneath
whichever of rung 3 / rung 4 wins.

**Seam.** `src/semantic.py`. A rung registers itself in `RUNG_BUILDERS` and
declares its imports in `RUNG_DEPENDENCIES`; `load_semantic_decoder()` gates on
the flag, then the chosen rung, then `try_import`, then a callable `decode`, and
falls back to `NullSemanticDecoder` at every one. `safe_decode()` is the
critical-path wrapper and returns `None` on any failure, which is exactly what
the null decoder returns — so a Tier 2 failure falls back to Tier 1's miss
handling **unchanged**.

---

## 2. Tier 2 versus an LLM for that slot

**The decision.** Whether the Tier 2 slot could instead be filled by a
generative model. The architecture document's position is settled; it has been
**raised as still open**, so it is recorded here and **needs explicit team
sign-off** rather than being treated as closed by default.

**The architecture document's position.** Rung 4.5 (a local generative model)
and rung 5 (an LLM API call) are **both ruled out**.

**The two arguments, and why they must be kept separate.**

- **Rung 5 loses on the network alone.** `submission_rules.md` says official
  scoring policy may disable network access, so an API-backed Tier 2 could be
  dead exactly when it counts. This is a decisive argument against rung 5 —
  and it says **nothing whatsoever** about rung 4.5, which is local. Collapsing
  the two arguments is how a reviewer concludes the local option was never
  really considered.
- **The reason that kills both is the same, and it is not the network.** A
  generative model emits tokens and has **no calibrated score to threshold**, so
  **it cannot abstain.** Abstaining toward refusal is precisely what makes a
  mediocre fallback safe here: Tier 2 fires only on a Tier 1 `unknown`, and the
  correct behaviour when it is also unsure is to hand back nothing and let Tier
  1's existing miss handling run. A model that always produces *an* answer
  cannot do that. It would also make the one subsystem we can prove
  deterministic stop being deterministic.

**What would settle it.** A team decision, recorded. There is no measurement
that decides this one — it is an architectural constraint about abstention, and
the evidence for it is structural rather than numeric. If the team overrides it,
the thing to demand is a *calibrated* confidence signal that can be thresholded,
because without one the safety property is simply gone.

**Seam.** `src/semantic.py` (this is the same one slot as item 1). Nothing to
build if the position stands.

---

## 3. Which LLM, for the ranking-side escalation

**The decision.** Which model — if any — backs the LLM re-ranking layer.
**Undecided.** This is the only place a language model is proposed anywhere in
the system, and it sits in ranking, never in intent.

**Scope, which bounds the risk.** It **re-orders a shortlist**. It can never
pull in a product BM25 never found, and `src/rerank.py::safe_rerank` discards
any result that is not a permutation of its input — so a broken or hallucinating
model costs BM25's order and nothing else.

**What the rules require** (`submission_rules.md`, `competition_specification.md`
"Model and API Policy") — all of it applies the moment anything lands here:

- disclose **model choice, approximate cost, token usage, latency, and any
  fallback behaviour**;
- **API keys pass through environment variables and are never committed**;
- the organizer **reserves the right to run the submission under CPU, memory,
  timeout and network restrictions**. So anything requiring live credentials
  must be **declared explicitly** and must have an **offline fallback**. Today
  that fallback is `NullLlmReranker`, and it is the shipped state.

**The honest expectation, stated up front.** Against today's simulator this
layer would **barely ever fire**. It exists for the case where the customer is
*not* quoting the listing — and 94.5% of disclosed constraint strings are
verbatim substrings of the target's own listing, so that case is ~5.5% of the
local set by construction. Expect a local delta near zero and do not read that
as a verdict on the private set; see framing point 1.

> **Corrected 1 Sep 2026 — the ~5.5% above is not the firing rate.** The
> prediction "barely ever fires" held; the number attached to it did not. 5.5%
> is the per-*string* complement of the 94.5% verbatim-overlap figure, and the
> gate is not per-string: `_llm_escalate` fires only when
> `overlap.measure(...).rate == 0.0` — **not one** disclosed segment appearing
> anywhere in the top-50 window — so one overlapping segment among several keeps
> it shut. Measured over the 200 public sessions with the layer live against a
> stubbed client: **0/571 turns leaky (0%)**, **24/1214 scrubbed (1.98%)**.
> Do not re-derive a firing rate from the overlap percentage; measure it.
> README.md carries the same correction under the cost table and "Limitations".

**What would settle it.** A measurement on **ESCI queries**, not on the public
set — human-authored queries are the only local proxy for the condition this
layer targets. Pair it with the disclosure numbers the rules demand (tokens,
latency, cost per session) and a decision about whether an API-backed layer is
declared at all, given that final scoring may be offline.

**Seam.** `src/llm_rerank.py`. A model registers in `MODEL_BUILDERS` and
declares its imports in `MODEL_DEPENDENCIES`. `usage()` returns `(0, 0)` today
and becomes the `prompt_tokens` / `completion_tokens` reported on the wire when
a real model lands — the accessor exists now so the disclosure is not
retrofitted later. Note that `load_llm_reranker` deliberately does **not** probe
network reachability at construction: a connection attempt at load time is
exactly what hangs on a network-disabled rig.

---

## 4. Cross-encoder checkpoint

**LIVE 1 Sep 2026 — `cross-encoder/ms-marco-MiniLM-L-6-v2`.** `load_reranker`
now defaults `enabled=True`; `src/rerank.py::_load_checkpoint` loads the
vendored checkpoint at `data/models/ms-marco-MiniLM-L-6-v2/`. It is the only
checkpoint this project ever measured (below), so shipping it is not a blind
choice — the "never compared" axis (below) is still genuinely open, just not
a reason to withhold a validated default. Degrades to `NullReranker` if the
checkpoint isn't vendored on a given machine or `sentence_transformers` isn't
installed, exactly as before.

**The decision (superseded framing below, kept for context).** Whether to ship
a local cross-encoder rerank, and which checkpoint. It is bundled and offline
— no network at any point — so this is not a network-policy question.

**The headline evidence.** Recorded in the architecture document and `README.md`
as **+0.047 TechnicalScore with the confidence interval excluding zero** at
**≈1.2 s/turn**. It is the largest single confirmed effect the bake-off
produced — an order of magnitude above the whole attribute-selection band
(≈0.004).

**⚠ A reconciliation debt, found while writing this register.** The `+0.047`
figure appears in `README.md` but in **no results artifact**. What
`bakeoff/results-part4.json` actually records for the current ledger is:

| rerank window | mean Δ TechnicalScore | 95% CI | excludes zero | s / reranked turn |
|---|---|---|---|---|
| top-10 | +0.0291 | 0.0147 – 0.0443 | yes | 0.22 |
| top-20 | +0.0417 | 0.0115 – 0.0736 | yes | 0.43 |
| top-50 | +0.0173 | −0.0299 – 0.0631 | **no** | 1.25 |

The headline pairs the **best delta** with the **slowest arm's cost**, and those
come from different rows: the ≈1.2 s/turn arm is top-50, which measures +0.017
with a CI that **includes zero**. **A human should reconcile this before the
figure is quoted again** — it materially changes the cost/benefit. Nothing here
reorders or overrides the approved position; it is recorded so the next person
does not re-derive it. (ESCI corroborates the *direction* independently: recall@10
0.8233 → 0.845 and MRR@10 0.6686 → 0.7173 on 600 human queries.)

**The three open axes.**

1. **Which checkpoint — never compared.** The bake-off measured *a*
   cross-encoder, not a field of them. No checkpoint comparison exists.
2. **Cost against a per-turn timeout the organizers have never published.**
   This is the load-bearing unknown. We are weighing a known cost against an
   unknown limit, which is why it stays open rather than resolving on the
   number.
3. **The rubric reading.** Per framing point 2, wall-clock never enters
   `TechnicalScore` — `Efficiency` is turn-based. So the latency is a **timeout
   risk**, not a score cost, and it also carries a Feasibility disclosure
   obligation. Do not model it as a score trade-off; it is an availability
   trade-off.

**What would settle it.** Axis 1: a checkpoint comparison on ESCI (cheap — the
bake-off caches candidates, so each arm is a re-ranking, not an evaluator
re-run). Axis 2: either the organizers publish a timeout, or we pick a window
whose worst-case per-turn cost is defensible without one and say so in the
report. Axis 3: nothing to measure — it is a reading, and it is recorded above.

**Seam.** `src/rerank.py`, `load_reranker`. `_load_checkpoint()` is the single
function that changes: it loads the bundled model, wraps it behind the
`Reranker` protocol, and returns it — or returns `None`, and we ship BM25's
order, which is what we ship today. Weights would ship as a **local asset**, not
as a download at load time.

---

## 5. Ask-yield, Layer 2

**The decision.** Whether to replace the fixed seven-slot schedule with an
ordering that adapts to what each attribute has actually taught so far. **On
hold.**

**The evidence we already have.**

- The entire question-ordering band is worth about **0.004** — the same
  magnitude as the `other` short-circuit that was declined outright. There is
  very little score here to win.
- Only about **40 of 200** public sessions reach turn 7 at all, which undercuts
  the original premise ("the fixed schedule runs dry after turn 6"): for 160
  sessions the schedule never runs out, so an adaptive ordering has nothing to
  do. (Do not confuse this 40 with the 40 *missed* sessions in item 7. Different
  forty.)

**How to judge it.** **On the design, not the delta.** Adaptive clarification
and question-value estimation are named Innovation directions in the
specification, and Innovation is 20 points that never read `TechnicalScore`. A
measurement of ~0 therefore does not condemn it — but neither does it justify
it, and the premise is under challenge. **If it ever regresses, fall straight
back to the fixed schedule**, without argument.

**What would settle it.** The premise investigation reporting first, and then a
paired measurement on the ~40 sessions that actually reach turn 7 — measuring on
all 200 dilutes the effect by 5× and will report noise either way.

**Seam.** `src/askyield.py`. It is a **swap-in, not a parallel system**: it
already sits behind the same `next_attribute(state)` signature the fixed
schedule uses, the body goes in `_adaptive()` and nowhere else, and the guard is
already written — a raise, a `None`, or a value outside
`ALLOWED_ATTRIBUTES − FORBIDDEN_ASK` all fall through to `_fixed()`. That last
rejection is the dangerous one: any value outside the enum is silently rewritten
to `other` by the evaluator, which would switch the permanently-declined exploit
on from a typo.

---

## 6. Schedule order — hand-tuned versus frequency-ordered

**The decision.** Whether the seven-slot order stays hand-tuned or is reordered
by how often each attribute actually gets classified. **Still open, and it needs
a human call.** Appending `budget` to the schedule did **not** settle it — that
fixed a missing slot, not the ordering question.

**The specific thing worth flagging.** The fixed schedule puts `budget`
**seventh**:

```
material, feature, color, style, size, use_case, budget
```

while the evaluator's `classify_constraint()` tests `budget` **first**. So
budget-classified constraints are common, and we ask for them at **turn 7** — by
which point many sessions have already ended (only ~40 of 200 reach turn 7 at
all, per item 5). That is a real structural mismatch between our ordering and
the evaluator's classifier, and it is the single most interesting thing about
the current order.

**How to judge it.** **This is the approved order, and it should not be
reordered on a score delta alone** (framing point 2). But it *is* the first
thing worth measuring once the core lands — the mismatch above is a hypothesis
with a clear test, not a hunch.

**What would settle it.** A paired measurement of the frequency-ordered
schedule against the hand-tuned one on both brackets, plus a human decision that
weighs the result against the design rationale for the hand-tuned order. Both
halves are required; the number alone does not decide it.

**Seam.** `src/types.py::FIXED_SCHEDULE` is the order itself (frozen — a change
comes back to assembly, not a local edit); `src/askpolicy.py::next_attribute`
walks it.

---

## 7. The never-repeat rank readout — a debt, not a conclusion

**What is owed.** The design asks for a specific measurement **before** the
never-repeat arm is built, and **we have shipped the rule ahead of that
readout.** Record it as a debt.

**The question, precisely.** Ten turns showing ten unseen products each means a
session examines **100 distinct items**. So the never-repeat rule can only
convert a miss into a hit when the target sits **inside the top 100 of some
turn's ranking**. The measurement:

> **What is the target's rank across the forty sessions that currently miss?**

(HitRate@10 is 0.80 on 200 sessions, so 40 sessions miss. Not the same forty as
item 5's "reach turn 7".)

**How to read the answer.**

- If those ranks cluster **inside the top 100**, paging through them is exactly
  the right idea and the rule is earning its place.
- If they cluster **past 500**, paging never reaches them, and **the idea closes
  with no build** — the rule is then not converting misses, it is only churning
  the list.

**Why it still shipped ahead of the readout.** The rule's *cost* is provably
zero and does not depend on this answer: the shown set is an **ordering
preference, not a removal** (`partition()`, then emit `fresh + seen` truncated
to k), so the top-10 is always full, and any product still on screen in a
running session is confirmed not to be the answer by the evaluator's own stop
condition. A rule that cannot cost anything is safe to ship early. That is an
argument for shipping it, **not** evidence that it works — which is exactly why
this stays on the register.

**What would settle it.** Instrument the 40 missing sessions from a captured
trajectory run (`bakeoff/capture.py` already writes the per-turn BM25 top-100)
and report the distribution of target rank. This is a read of existing
artifacts, not a new experiment.

**Seam.** `src/shown.py` (the registry and the override guard) and the
`partition` step in `src/pipeline.py::run_turn`.

---

## 8. The `phrase_plus` retrieval arm — flagged, not approved

**The decision.** Whether to add phrase matching alongside the unigram OR query.
**Not approved.** It does not appear in the architecture document, and it is
recorded here so that its local number does not get quoted as a shipped result.

**The evidence, both halves.**

| measurement | arm | result |
|---|---|---|
| local public set (leaky) | `phrase_plus` | **+0.0588** ΔTechnicalScore, CI 0.0316 – 0.0879, excludes zero (0.6926 → 0.7513) |
| ESCI, 600 human queries | unigrams (shipped) | recall@10 **0.8233** |
| ESCI, 600 human queries | phrases + unigrams | recall@10 **0.8250** |
| ESCI, 600 human queries | phrases only | recall@10 **0.2400** |

**The asymmetry, stated plainly.** A **large** gain on a leaky local set and
**roughly nothing** (+0.0017 recall@10) on real human queries is the signature
of a **measurement artifact, not an improvement.** The mechanism is not
mysterious: 94.5% of the simulator's constraint strings are verbatim phrases
copied out of the target's listing, so a phrase matcher is being handed the
answer key in a form only this simulator produces. The `phrases only` collapse
to 0.2400 is the same finding from the other side — strip the unigrams and a
phrase matcher has almost nothing to work with on language a human actually
wrote.

**What would settle it.** Nothing that is currently cheap. It would need
evidence that the private set's utterances *also* copy phrases from listings —
which is precisely what the organizers' reserved right to reword makes unsafe to
assume. Absent that, the default is: **do not ship it**, and do not cite the
+0.0588 without the ESCI row next to it.

**Seam.** `src/retrieval.py`, query construction (`MAX_QUERY_TERMS` unique
stopword-filtered terms, OR-joined). `bakeoff/followup_phrase.py` and
`bakeoff/followup_phrase_esci.py` hold the two measurements above.

## 9. Dense fusion — deferred for architecture, not refuted by evidence

**The decision.** Whether BM25 stays the sole retrieval route. **Deferred, and
the reason is the dependency budget, not the measurement.** `docs/architecture-status.md`
item 6 currently reads "rejected, not gated"; that wording is stronger than the
evidence below supports and should be revised to match this section.

**The evidence, both halves.**

| measurement | arm | result |
|---|---|---|
| local public set (leaky), top-50 | dense alone (R2) | hit@10 **0.330** vs BM25 0.800 |
| local public set (leaky), top-50 | RRF (R3) | hit@10 **0.720**, ΔTS −0.1110 |
| local public set (leaky), top-100 | RRF (R3) | hit@10 **0.660**, ΔTS −0.1554 |
| local public set (leaky), top-50 | weighted, w=0.2 (R4) | **+0.0228** ΔTechnicalScore (0.6926 → 0.7154), **no CI computed** |
| ESCI, 600 human queries | BM25 (shipped) | recall@10 **0.8233**, mrr@10 0.6686 |
| ESCI, 600 human queries | RRF | recall@10 **0.8733**, mrr@10 0.7089 |
| ESCI, 600 human queries | weighted w=0.5 | recall@10 **0.8833**, mrr@10 0.7256 |

**The asymmetry, stated plainly — and it points the opposite way from §8.** A
**large loss** on the leaky local set and a **clear gain** (+0.060 recall@10)
on real human queries is the signature of a **rig artifact suppressing a real
improvement**, which is the mirror image of the `phrase_plus` case. Same
mechanism, opposite sign: 94.5% of the simulator's constraint strings are
verbatim substrings of the target listing, so the local rig hands lexical
retrieval the answer key and gives semantic matching nothing left to recover.
The target sits at BM25 rank 1 in 87 of 176 hit sessions but around dense rank
72 — true on *this* rig, and the reason blending dilutes here.

**Two caveats on the local numbers, both load-bearing.** R2 and R3 are the arms
that lose; **R4, the weighted sweep, does not.** Its optimum is at w=0.1–0.2 on
every configuration measured (minilm and bge, top-50 and top-100, both ledgers),
not at w=0 as `bakeoff/part3_fusion.py`'s own docstring anticipated. It is also
the one arm for which **no bootstrap CI was ever computed**, so +0.0228 is an
unqualified point estimate and must not be quoted as a result. The blanket claim
that "fusion loses on every arm" is true of R2 and R3 only.

**Why it is not shipped now — the architectural reasons, which are the real ones.**

1. **`src/` is standard library only and `requirements.txt` is comments-only by
   design.** Dense retrieval means numpy, a sentence-transformer runtime, and a
   bundled checkpoint — the first third-party dependency on the graded path.
2. **Final scoring runs offline.** Each added dependency is another thing that
   can fail to import or fail to locate its weights, and the evaluator converts
   any such failure into a silent zero rather than an error.
3. **There is no live encoder path.** Every number above is replayed from
   precomputed `.npy` caches in `bakeoff/cache/`. A shippable implementation is
   new work, not a port of existing code.

**What would settle it.** Two cheap steps, in order, neither needing a model:
bootstrap a CI on the existing R4 sweep so its sign is known rather than
assumed; then decide the dependency question on its own terms, because that is
what is actually blocking — not the score.

**Seam.** `src/retrieval.py` is the route it would occupy. `bakeoff/dense.py`
(encoders), `bakeoff/part2_dense.py::dense_topk`, and
`bakeoff/part3_fusion.py::dense_ranker` / `rrf_ranker` / `weighted_ranker` hold
the three fusion implementations measured above.

---

## Where to look next

- `docs/hard-rules.md` — the seven implementation rules plus rule 0
  (TechnicalScore is not the objective), with what each one binds.
- `bakeoff/README.md` — how the cached-trajectory replay works, why there are two
  ledgers, and the ESCI provenance note.
- `evaluation-data/README.md` — the access rule and, more importantly, why
  *provenance* rather than access is the guard that actually holds.
- `requirements-optional.txt` — the commented-out candidates for every layer in
  this file, and the flag each one is gated behind.
