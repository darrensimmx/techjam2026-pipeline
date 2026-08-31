# LLM escalation (`LLMRR`) — window sizing, output contract, model selection

Consolidated findings, 1 Sep 2026. Scope: the `LLMRR` node from the docs repo's
architecture v5 — the layer above the cross-encoder that fires only on the
low-overlap branch. Companion probe: `bakeoff/followup_llmrr_esci.py`.

**Status of the numbers.** §1–§3 are measured offline on 600 real ESCI queries
(Amazon customers' own wording, human Exact-relevance labels), with no API key
and no model call. §6b is measured on the real `src/` pipeline over 30 stratified
public sessions, also with no model call. **§4–§6 are sourced from public
benchmarks and vendor pricing, and are a prior, not a measurement.** Nothing here
is a TechnicalScore, and ESCI is single-turn — §1–§3 measure turn-1 retrieval,
not the conversational pipeline.

> **Revision note, 1 Sep 2026.** Building the measurement harness turned up four
> errors in the first cut of this report. All four are corrected in place and
> flagged where they occur; none of the *decisions* changed, but two of the
> numbers behind them did.
>
> | § | was | is | why |
> |---|---|---|---|
> | 2 | 9.13 pts headroom @ top-50 | **5.48 pts** | measured against a CE that had only scored the top 20 |
> | 3 | ceilings at n=272 | **n=274**, regenerated | old table unreproducible from committed source |
> | 6 | caching saves ≤8.7% | **0%, structurally** | Haiku 4.5's cacheable prefix floor is 4096 tok; ours is ~190 |
> | 6 | full sweep ≈ $3.44 | **≈ $11.60** | §5's own $/turn column already implied $11.70; the two contradicted each other |
>
> **No live model has ever been called.** Every number in §1–§3 and §6b comes
> from offline runs; §4–§6 remain a prior. The three-arm probe is built,
> verified against offline `echo`/`reverse` controls, and blocked only on
> credentials.

---

## 1. The gate's premise holds, and it is graded

Segmenting the 600 queries into terciles by **phrase-level** verbatim overlap
between the shopper's wording and the retrieved listings — the property the
94.5% figure actually rests on:

| band | n | CE@20 R@10 | CE@20 MRR@10 | CE@50 R@10 | CE@50 MRR@10 |
|---|---|---|---|---|---|
| **vague** | 274 | 0.7518 | 0.6209 | **0.7883** | **0.6377** |
| mid | 151 | 0.8742 | 0.7424 | 0.9272 | 0.7615 |
| literal | 175 | 0.9657 | 0.8466 | 0.9771 | 0.8518 |

*The depth column was missing from the first cut of this table, which is how the
top-20 baseline came to be quoted at top-50 in §2. Both are shown now; §2's
headroom is computed against CE@50.*

BM25 + cross-encoder is markedly worse exactly where the wording is not copied,
at either depth — a 19-point R@10 gap between vague and literal at CE@50. That
is the deficit the layer is argued for, now measured on real queries rather than
asserted.

> **Method note.** An earlier cut of this used *token-level* coverage — what
> fraction of query tokens appear anywhere in the listing. That is the measure
> `bakeoff/overlap.py` reports second and `part5_realqueries.py` retracted: it is
> confounded by query length, saturates at 1.0 for short queries, and put every
> query on one side of the split. The banding above is phrase-level.

## 2. Why top-50, not less — the headroom

Oracle headroom on the **vague** band: what a *perfect* reranker could gain over
the CE baseline, i.e. the reachable ceiling minus what the CE already achieves.

> **⚠ REVISED 1 Sep 2026, and the headline number moved.** The first cut of this
> table measured headroom against a cross-encoder that had only ever scored
> BM25's **top 20** (`bakeoff/followup_ce_esci.py` shipped with `DEPTH = 20`).
> Past rank 20 the shortlist stayed in raw BM25 order, so the "CE baseline at
> top-50" was really *CE@20 + 30 unranked candidates* — and, tellingly, it came
> out **bit-identical** to the top-20 baseline, because only the CE-sorted first
> 20 can reach a top-10 at all. The LLM was therefore being credited with
> recovering targets sitting at BM25 ranks 21–50: **work a real CE@50 does by
> itself.** The cross-encoder has now been re-run at depth 50
> (`bakeoff/cache/ce-esci-top50.json`, ~12 min CPU, free) and the table below is
> against that. Numbers from `bakeoff/results-llmrr-baseline-check.json`.

| window | ceiling (vague) | CE@50 baseline | headroom over CE |
|---|---|---|---|
| top-10 | 0.7153 | 0.7153 | **0.00 pts** |
| top-20 | 0.7701 | 0.7518 | 1.83 pts |
| **top-50** | **0.8431** | **0.7883** | **5.48 pts** *(was 9.13)* |

**At top-10 the headroom is zero by construction** — you rank ten items and
return all ten, so reordering cannot change HitRate@10 at all. It moves only MRR
(11.2 pts available there, weighted 0.30 against HitRate's 0.50).

At top-20 the available gain is 1.83 pts. That is close enough to the "≈0" that
`llm-escalation-proposal.md` names as its own kill condition — *"if the remaining
gap is ≈0, this layer provably cannot act, and shipping it anyway is the worst
available Feasibility row."*

**Top-50 is still the first window at which the layer can act on the metric that
carries the most weight** — 5.48 pts against 1.83 and 0.00 — so the *decision*
survives. But **the prize is 40% smaller than this report first claimed**, and
the 3.65 points that vanished were never the LLM's to win.

Headroom concentrates in the gated band — and, against the real CE, far more
sharply than the first cut showed:

| band | headroom @ top-50 | first cut (CE@20) |
|---|---|---|
| **vague** | **5.48 pts** | 9.13 |
| mid | 1.32 pts | 6.62 |
| literal | 0.58 pts | 1.72 |

**9.4× concentration in the band the gate routes, up from 5.3×.** This is the one
place the correction argues *for* the design rather than against it: a real CE@50
mops up the mid band almost entirely (6.62 → 1.32) while leaving most of the
vague band's gap intact, so what remains is precisely the lexical blindness the
layer is argued for. The layer is aimed correctly, at a smaller target.

> **This supersedes a line in the docs repo.** `llm-escalation-proposal.md` says
> *"Escalate over the top-10 or top-20, not the top-50 — one listwise call, short
> prompt."* That was asserted without measurement, and it would have capped the
> layer at ~0 HitRate headroom. It should be revised.

## 3. Why top-50, not more — the CE cost

A cross-encoder scores each (query, document) pair **independently** — pointwise,
no cross-candidate attention. Doubling the pool does not make any individual
score worse; there is no per-pair degradation. Two other things do change:

- **Cost scales linearly.** The CE is ~1.2 s/turn at top-50 (measured,
  single rig). At top-100 that is ~2.4 s. Against a budget already flagged as
  timeout/DQ risk, that **+1.2 s alone exceeds the entire LLM layer's cost**
  with index output (~0.42 s, §4).
- **More distractors.** Candidates at BM25 ranks 51–100 are on average worse but
  some will still outscore the target. Ordinary precision/recall, not
  degradation — but it means ceiling gains do not convert 1:1 into realised gain.

Ceiling gain above 50, vague band (**n=274**, regenerated 1 Sep 2026):

| top-20 | top-50 | top-100 | top-200 |
|---|---|---|---|
| 0.7701 | 0.8431 | 0.8686 | 0.8686 |

> **The previous row read `n=272` and `0.7684 / 0.8382 / 0.8676 / 0.8676`, and
> could not be reproduced from committed source.** Band sizes are computed off
> the top-10 window and are therefore depth-independent, so `272 ≠ 274` cannot
> be a depth artifact — it means that run used a different banding or shortlist
> from a code state no longer in the repo. It is superseded rather than
> reconciled. The regenerated row is reproducible:
> `python bakeoff/followup_llmrr_esci.py --arms "" --limit 0 --depths 10,20,50,100,200`,
> which needs no API key and writes `bakeoff/results-llmrr-baseline-check.json`.
> `DEPTHS` was a module constant when the old table was made, which is how the
> numbers became unreproducible; it is a `--depths` flag now.

**+2.55 pts of ceiling for +1.2 s of CE time. Bad trade, and the conclusion is
unchanged.** (top-200 matching top-100 means the cached BM25 list stops at 100 —
beyond that is unmeasured, not flat.)

**The component that genuinely degrades with pool size is the LLM, not the CE.**
Listwise attention is O(n²) over the candidate block; the literature puts the
useful range at top-5 to top-10, and RankZephyr covers a top-100 with a **window
of 20 and stride of 10** rather than one wide call. A single listwise call over
100 candidates is well outside anything demonstrated.

**Asymmetry worth knowing:** the CE window and the LLM window need not match.
BM25 retrieval to 100 is nearly free (an FTS5 query), so the CE could hold at 50
while the LLM sees ranks 51–100 in raw BM25 order — input tokens only, no CE
time. That is already how the probe builds its depth-50 list. It buys recall by
walking into the listwise degradation problem, which is unmeasured.

**Decision: hold the CE at 50.** Revisit only if the probe finds a model whose
ranking quality holds at wide windows.

## 4. The output contract is the latency bottleneck, not the model

Generation time is driven by **output** length. Prefill is parallel, which is why
TTFT stays ~0.2–0.3 s even on long prompts — so a wide *input* window is cheap
and a long *output* is not.

Input fixed at the top-50 window (~4,640 tok) in every row:

| output encoding | out tok | Gemini 3.5 Flash | GPT-5.6 Luna | Haiku 4.5 |
|---|---|---|---|---|
| 50 ASINs + rationale | 445 | 2.09 s | 2.34 s | 4.05 s |
| 10 ASINs + rationale | 125 | 0.59 s | 0.66 s | 1.14 s |
| 10 ASINs, no rationale | 90 | 0.42 s | 0.47 s | 0.82 s |
| **10 indices, no rationale** | **30** | **0.14 s** | **0.16 s** | **0.27 s** |

**15× reduction, keeping the full 9.13-point recall headroom.** Three separate
levers: emit 10 not 50 (−72%), indices not ASINs (−67%, because a 10-char
alphanumic ASIN costs ~8 tokens in a JSON array where a small integer costs ~2),
drop the rationale (−28%).

The competition's indifference to the agent's prose licenses only the last and
smallest of those. Verified at `local_evaluator.py:280` —
`0.50*hit_rate@10 + 0.30*mrr + 0.20*efficiency`, no component reads message text;
`:243` type-checks that `message` is a `str` and zeroes the turn otherwise, so
`message: ""` is free but the field must exist.

**Consequence: latency stops discriminating between models.** With index output
all three land at ~0.4 s per turn, ~1.6 s with the CE. Latency drove the entire
earlier analysis; fixing the output contract dissolves it, and the decision
reverts to ranking quality on vague queries at a wide window — which no public
leaderboard measures.

*Token counts in this table are estimates. Confirm with `count_tokens` before
banking them; the whole argument rests on the ASIN-vs-integer ratio.*

## 5. Model selection

**Reasoning mode is disqualifying.** Published leaderboard latency is quoted at
high effort: Gemini 3.5 Flash at `high` is 15.28 s to first token, GPT-5.6 Luna
at `xhigh` is 60.67 s. Every arm must be pinned to minimum-reasoning mode. This
also rules Rank-K's test-time-reasoning approach off the request path despite its
+23% over RankZephyr.

| # | Model | Shopping-domain evidence | Gen (read 50 / emit 10 idx) | $/turn @50 |
|---|---|---|---|---|
| 1 | **Gemini 3.5 Flash** | τ² 95.3 | 0.14 s | $0.0111 |
| 2 | **GPT-5.6 Luna** | agentic 84.1, Terminal-Bench 84.7 | 0.16 s | $0.0015 |
| 3 | **Claude Haiku 4.5** | none published | 0.27 s | $0.0069 |

**Generational caveat on the shopping boards.** τ-bench Retail (frozen at a
2024–25 model set) shows the fast tier collapsing — GPT-4.1 nano 0.226 against
GPT-4.1's 0.680, Claude 3.5 Haiku 0.510 against Claude Sonnet 4.5's 0.862. τ²
shows the opposite on current models — GPT-5.4 nano 92.5 against GPT-5.4's 98.9,
Gemini 3.5 Flash 95.3 against Gemini 3.1 Pro's 95.6. **"Small models can't do
shopping reasoning" is a generational artifact, not a standing law.** Reading
either board alone misleads in opposite directions.

Two fast-tier models outscore all three candidates on τ² — GLM-4.7-Flash 98.8
and Step 3.7 Flash 98.5 — but throughput and per-turn pricing were not sourced,
and provider selection for a competition submission carries non-technical
considerations.

Caveats: the τ² ledger warns it "mixes datasets across domains rather than
offering controlled cross-provider ranking"; τ² retail tests a customer-service
agent, not listwise product reranking.

## 6. Limits and open items

- **Recall caps the layer.** Even a perfect top-50 reranker reaches 0.8431 on the
  vague band. The remaining ~15.7% is retrieval failure — the target is not in
  the window at any size — and nothing in this layer touches it. This is the
  scope limit `llm-escalation-proposal.md` already writes down, now measured, and
  it belongs in that file's "What would kill this".
- **Prompt caching buys exactly nothing here — it cannot fire at all.** Corrected
  1 Sep 2026: this bullet previously read "max saving at the 90% cache discount is
  8.7% of input cost", reasoning from the 190-token instruction being 9.6% of
  input. The payload-shape argument was right and the number was still wrong,
  because **Haiku 4.5's minimum cacheable prefix is 4096 tokens** (the Opus 4.6 /
  Opus 4.5 / Haiku 4.5 tier; the Sonnet tier is 1024). A 190-token prefix is 21×
  under the floor, so nothing caches, silently, with no error — `cache_read`
  reads 0 forever. The saving is **0%, structurally**, not 8.7%.

  Two consequences. Caching is not a cost lever on this layer under *any* payload
  shape, since the only stable content is the instruction. And it is a real, if
  small, argument against Haiku specifically: a Sonnet-tier model would clear the
  1024 floor. `bakeoff/followup_llmrr_esci.py` keeps the `cache_control` marker
  so the zero is attributable rather than mysterious.
- **The safety contract changes shape.** Emitting 10 of 50 makes "dropping" a
  `parent_asin` inherent, so the guard becomes: every index in range, unique,
  exactly 10. Arguably stronger — an out-of-range integer is trivially invalid,
  where a hallucinated ASIN could coincidentally be a real product. The
  proposal's wording needs updating.
- **Index indirection is unmeasured.** Emitting positions rather than ids may
  cost ranking accuracy, especially at 50 candidates. Run both encodings as arms.
- **Nothing is measured with a live model.** The probe is built and verified
  offline against `echo` and `reverse` control arms — `echo` reproduces the CE
  baseline to the last digit at every depth and encoding, `reverse` is measurably
  worse, so the model's output demonstrably reaches the metric. It is blocked on
  API credentials, and the Gemini and OpenAI model IDs and minimum-reasoning
  parameter spellings still need confirming against provider docs before a paid
  run. `google.genai` (which the probe imports) is not installed either; only the
  legacy `google.generativeai` is.

- **Cost, corrected.** This section previously said *"Full three-arm sweep over
  600 queries: ~$3.44"*, which contradicted §5's own table: that `$/turn @50`
  column sums to $0.0195, i.e. **$11.70** over 600 queries. Neither figure
  accounted for the depth sweep being three windows rather than one. Measured
  token shapes are still owed here — `count_tokens` is billed at $0 and would
  settle it — but on §4's estimates a **Haiku-only, both-encodings, three-depth**
  sweep is ≈**$11.60**, of which the `permutation` arm's output alone is $2.36
  against the `indices` arm's $0.27. Cheapest useful cuts: `--depths 50` only
  (≈$6, and top-50 is the only window with non-zero HitRate headroom), or
  `--limit 200` (≈$4). A `--limit 40` smoke is ≈$0.48.

- **The ASIN-vs-integer ratio is still unbanked, and §4 rests entirely on it.**
  §4 claims a 10-char ASIN costs ~8 tokens in a JSON array where a small integer
  costs ~2. If it is really 4:2, the "15× reduction" headline shrinks and the
  model-selection conclusion moves with it. `messages.count_tokens` needs
  credentials but costs nothing; it should be the first call made.

## 6b. The gate overrides the reranker on ~half of all turns

**Measured 1 Sep 2026, on the real `src/` pipeline, 30 stratified public
sessions. This was not previously known and it is not in the diagram below.**

`src/overlap.py::gate` runs **after** `_rerank` (`src/pipeline.py:140`) as a
stable sort keyed `(-overlap, incoming_index)`. It never filters — the window
comes out the same length — but it freely reorders, so it can displace whatever
the reranker put first.

Running the shipped pipeline with an instrumented reranker in the `deps.reranker`
seam and comparing what the reranker handed back against what actually reached
the wire:

| bracket | turns reranked | gate changed the top pick |
|---|---|---|
| leaky | 96 | **38 (40%)** |
| scrubbed | 213 | **112 (53%)** |

Both arms used a pass-through reranker, so this is the gate acting on BM25's own
order — the *floor*, not a reaction to anything a model did. An LLM that reorders
more aggressively would be overridden at least this often.

**Why this matters more than a score delta.** The layer is argued for on its
ability to put the right product first on vague queries. Under the current
component order, that decision is advisory on roughly half of turns: the gate
re-sorts by verbatim overlap immediately afterwards, and verbatim overlap is
exactly the signal the vague band lacks. So on the queries the layer exists to
serve, the gate is most likely to be indifferent between candidates — and where
it is not indifferent, it is ranking on the property the layer was introduced
*because* it fails.

This is an ordering question, and it is cheap to answer before any model is
chosen: run the gate first as a filter/annotation and the reranker last, or make
the gate a tie-breaker among equal-overlap candidates only. Neither needs an API
key. **Settle this before spending anything on model selection**, because a
model measured through a gate that discards half its decisions is not being
measured at all.

Reproduce: `python bakeoff/followup_llmrr_publicset.py --arms none --bracket both`.

## 7. Design of record

```
BM25 (FTS5, cheap to 100)
  → CE rerank top-50            ~1.2 s, pointwise, bundled local model
  → overlap gate                deterministic, local, free
      ├─ high overlap → return  lexical is near-optimal; nothing to gain
      └─ vague        → LLM     emits 10 indices, no rationale, ~0.4 s
  → return 10
```

The gate is not optional. Running the LLM on every session is the blanket
version the proposal rejects: 94.5% of sessions have nothing to infer, so it is
full latency for zero gain, and it is the dilution failure that sank dense fusion
(−0.206 / −0.065).
