# techjam2026-pipeline

**TikTok TechJam 2026 — Statement 4: Shopping Copilot (AI Conversational Search
and Recommendations).** A multi-turn shopping agent over a frozen 50,000-product
Amazon apparel catalog. Each turn it decodes the customer's reply, folds it into
a constraint ledger, retrieves and re-ranks, returns a full top-10, and spends
the turn's one question on the attribute most likely to move the next retrieval.

The whole system runs **in-process, in-memory, with no network required**. One
optional layer can call a hosted LLM when it is available; it is off unless
credentials are present, and its absence changes nothing but the ordering of a
shortlist.

---

# Submission

Everything the organizer needs is in this section. The phase plan after it is
internal team history, retained for provenance.

## What ships

```text
agent.py            the submission entry point -- exports `Agent`
src/                the system (18 modules)
requirements.txt    empty by design: the graded core needs nothing but the stdlib
requirements-optional.txt   the three optional layers' dependencies
README.md           this file
```

Everything else is development tooling that never reaches the organizer:

- **`starter/`** — the superseded first-generation system, retained unmodified
  as the historical control the rebuild is measured against. **Not part of the
  submission.**
- **`evaluator/`** — the organizer's kit, vendored byte-identical and never edited.
- **`cli/`, `tests/`, `scripts/`, `bakeoff/`, `evaluation-data/`** — dev tooling,
  test suite, scoring harnesses, and measurement rigs.

> ⚠️ `evaluator/local_evaluator.py:12` hardcodes `from starter.agent import Agent`.
> Running the vendored evaluator directly scores the **superseded** system.
> Score the submission only through `scripts/evaluate_src.py`.

## Setup and installation

### 1. Python

**Python 3.10 or later.** Verified on 3.11.9 and 3.14.2. SQLite must be built
with **FTS5** (standard on CPython's official builds) — the retrieval index
depends on it.

### 2. Dependencies

```bash
pip install -r requirements.txt        # a no-op: the graded core has no dependencies
```

The core is standard library only — `json`, `re`, `sqlite3`, `dataclasses`,
`typing`. This is deliberate: `submission_rules.md` reserves the organizer's
right to run the submission under CPU, memory, timeout and network restrictions,
and fewer dependencies is less that can fail when it counts.

**The agent runs, scores, and is fully functional with nothing else installed.**
The three optional layers below each degrade silently to a null implementation
when their dependency is absent.

### 3. The catalog

Place the organizer's catalog at `data/catalog.jsonl` and the public sessions at
`data/public_set.jsonl`. Both are gitignored (~50k rows, distributed as a release
asset).

```bash
python -c "from agent import Agent; print('degraded:', Agent('data/catalog.jsonl').degraded)"
```

`degraded: False` means the index built. **If the catalog is missing the agent
still starts and still returns schema-valid responses — it just returns zero
recommendations every turn and scores exactly `0.00000`.** `Agent.__init__`
swallows the load failure by design and there is no logging anywhere in `src/`,
so nothing warns you. A `TechnicalScore` of exactly zero is almost always this.

### 4. Optional layers (not required)

```bash
pip install model2vec sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
pip install google-genai

# vendor the two local checkpoints -- fetched once, then never network again
python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M').save_pretrained('data/models/potion-base-8m')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').save('data/models/ms-marco-MiniLM-L-6-v2')"
```

Confirm which implementation actually loaded — the null fallback is silent:

```bash
python -c "from src.rerank import load_reranker; print(load_reranker().name)"
python -c "from src.semantic import load_semantic_decoder; print(load_semantic_decoder().name)"
python -c "from src.llm_rerank import load_llm_reranker; print(load_llm_reranker().name)"
```

Expect `cross-encoder/ms-marco-MiniLM-L-6-v2`, `rung3_centroid`, and
`gemini-3.5-flash`. Any `null*` name means that layer is inert on this machine.

### 5. Environment variables

**One, and it is optional:**

| Variable | Required? | Effect if unset |
|---|---|---|
| `GEMINI_API_KEY` | **No** | The LLM escalation layer loads `NullLlmReranker` and never fires. Every other layer is unaffected. Nothing warns, by design. |

The key is read by `google.genai.Client()` from the environment. It is never
read, logged, or committed by any file in this repository.

## One command to run the agent in the official harness

```python
from agent import Agent          # Agent(catalog_path="data/catalog.jsonl")
```

The organizer's evaluator imports this class and calls `reset()` / `respond()`
in-process. No CLI, no server, no network on the graded path. All paths resolve
relative to the repository root.

## Reproducing our results

```bash
# full test suite -- 390 tests. Check the count before believing green.
python -m unittest discover -s tests -p "test_*.py" -t .

# score the submission. --bracket both reports the leaky/scrubbed spread.
python scripts/evaluate_src.py --catalog data/catalog.jsonl --dataset data/public_set.jsonl --bracket both
```

`scripts/evaluate_src.py` drives the organizer's own unmodified `evaluate()`
over the 200 public sessions and appends a row to the tracked run log
`results_src.md`. It prints `degraded` on every run.

On Windows the interpreter is `python`, not `python3`.
`docs/windows-dev-setup.md` is the verified setup guide.

## Architecture

`src/agent.py::Agent.respond()` is a never-raise wrapper around one pass of
`src/pipeline.py::run_turn`. Twenty steps, every one of them individually
guarded:

```text
decode (Tier 1 regex)  ->  Tier 2 semantic fallback  ->  contradiction check
  ->  override guard  ->  ledger append  ->  slot fill  ->  ask bookkeeping
  ->  query = concat(ledger)  ->  BM25 (pool 300)  ->  partition by shown-set
  ->  hydrate top-50 window  ->  cross-encoder rerank  ->  verbatim-overlap gate
  ->  LLM escalation (vague branch only)  ->  assemble top-10  ->  record shown
  ->  choose ask_attribute  ->  schema coercion
```

### The three layers, by blast radius

Layer membership encodes the **runtime safety contract** — whether a component
runs every turn and what it costs when it fails. It is not a build order.

| Layer | Members | If it fails |
|---|---|---|
| **1 — offline core** | BM25 index, Tier 1 frame decode, constraint ledger, slot state, overlap gate, shown-set, ask policy, response builder | Swallowed into an empty response — that turn scores 0. Fail on *every* turn and the run is exactly `0.00000`. |
| **2 — adaptive orchestration** | ask-yield ordering (`src/askyield.py`, **off**) | Falls back to the fixed schedule. The whole attribute-selection band is worth ≈0.004. |
| **3 — optional, deletable** | cross-encoder rerank, Tier 2 semantic fallback, LLM escalation | Caught by its own try/except; Layer 1's ordering stands. Score unchanged. |

The evaluator wraps `respond()` in a bare `try/except` and swallows failures into
an empty response — **`TechnicalScore` 0.00000, silently, no crash, no traceback,
looks like a clean run.** That single fact governs every design decision here:

- `respond()` never raises; it returns `src/contract.py::empty_response()` on any
  exception.
- `__init__` and `reset()` are **not** wrapped by the evaluator
  (`local_evaluator.py:306` and `:228`) — a raise in either kills the *entire
  run*, not one session. Both are guarded here.
- Every outgoing payload passes through `src/contract.py::validated()`, which
  coerces each field to its schema-valid empty form rather than letting an
  invalid value through.

### Component notes

- **Intent is a decode, not an estimate.** The simulator emits a closed set of
  eight f-string reply templates, so `src/frames.py` matches them with anchored
  whole-frame regexes. It splits the two declines on the single token
  `additional`: "I don't have **a** preference" leaves the bucket live (re-ask
  later), while "I don't have an **additional** preference" proves it empty
  (retire permanently).
- **The ledger *is* the query.** Every disclosed reply is appended verbatim, and
  the concatenation of those raw strings is what gets searched — no parsing sits
  between the customer's words and retrieval. Append-only, enforced by the
  absence of any deletion method, and never erased even on intent override.
- **Slots are scheduling-only.** The typed slot view drives the override check
  and the question schedule, and never touches retrieval. A parsing bug can
  corrupt *which question we ask*; it cannot corrupt *what we search*. Asserted
  structurally in `tests/test_src_layering.py`.
- **Retrieval is BM25, single route.** SQLite FTS5 over an in-memory index built
  once at construction. Terms are quoted as phrases and OR-joined over ≤40 unique
  stopword-filtered terms, so a stray FTS5 operator in a customer reply cannot
  break the query. Dense/embedding fusion was measured twice and **rejected**
  (−0.206 at top-100, −0.065 at top-50): the target is BM25's rank 1 in 87 of 176
  hit sessions but sits around dense rank 72, so blending dilutes a strong list
  with a weak one.
- **Never repeat a shown product.** A session ends the instant our list contains
  the target, so anything still on screen in a running session is confirmed
  wrong — excluding it cannot lose the target. It is an *ordering* preference,
  not a filter: `partition()`, never `filter()`, so the top-10 is always full.
  Across an intent override the evaluator's hit check was suppressed early, so
  everything shown before the override goes back in play.
- **The ask policy never returns `null` and never returns `other`.** A
  seven-attribute fixed schedule for turns 1–7, then a fallthrough ladder
  re-evaluated independently on each free turn.

### The routed LLM escalation

The one place a language model touches the system, and it sits in **ranking**,
never in intent parsing.

**94.5% of the simulator's disclosed constraint strings are verbatim substrings
of the target product's own listing.** In those sessions the customer is
effectively reading text off the product page: exact term matching is close to
the ideal algorithm, and a reasoning model adds nothing while costing full
latency. So the LLM is **routed, not blanket** — gated on a deterministic, local,
zero-cost check (`src/overlap.py`) that asks whether the disclosed strings appear
as literal substrings in the retrieved listings.

| overlap | route |
|---|---|
| **any literal overlap** | BM25 → cross-encoder → return. Lexical is near-optimal; nothing to gain. |
| **zero literal overlap** ("vague") | BM25 → cross-encoder → **LLM** → return. Exact matching is blind; inference is the only thing that can help. |

The trigger is **lexical blindness, not customer vagueness**. *"Something that
won't soak through in heavy rain"* is perfectly specific and has near-zero
lexical overlap; *"waterproof nylon shell"* is copied off the listing. The first
is this layer's case, the second is not.

Its safety contract is the same as every Layer 3 component: its own try/except, a
strict response schema, a permutation check that discards any result which is not
a re-ordering of its input, and a fallback to **the cross-encoder's ordering** —
never BM25's, never an empty list. The worst case of the whole layer is "we score
exactly what we score without it."

## Model, cost, token, latency and network disclosure

Required by `submission_rules.md` ("Model Policy", "Reproducibility
Requirements") and `competition_specification.md` ("Model and API Policy").

### Configuration

| Layer | Model | Where it runs | Dependency | Network | Default |
|---|---|---|---|---|---|
| Retrieval, ledger, intent Tier 1, slots, overlap gate, ask policy | **none** | in-process | stdlib only | none | **always on** |
| Cross-encoder rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params) | **local**, vendored at `data/models/` | `sentence-transformers`, `torch` | **none at inference** | on when the checkpoint is present |
| Intent Tier 2 fallback | `minishlab/potion-base-8M` via `model2vec` (static embeddings, numpy-only) | **local**, vendored at `data/models/` | `model2vec` | **none at inference** | on when the checkpoint is present |
| LLM ranking escalation | `gemini-3.5-flash`, `thinking_budget=0` | **hosted API (Google)** | `google-genai` + `GEMINI_API_KEY` | **yes, when it fires** | on when the key and package are present |
| Ask-yield adaptive ordering | none | in-process | stdlib | none | **off** (`ADAPTIVE_ENABLED = False`) |

**No generative model anywhere in intent classification.** Tier 1 is regex; the
Tier 2 fallback behind it is an *encoder* (one forward pass of a frozen static
embedding model, nearest-centroid over the eight known reply shapes), chosen over
a fine-tuned head on held-out numbers: 0 wrong out of 168 against a comparison
model's 29, which wins under this project's asymmetric-cost rule even at lower
combined recovery.

### Does this submission require network access?

**No.** The offline core plus both local checkpoints is the complete system minus
one ranking enhancement. Concretely:

| Condition | Behaviour |
|---|---|
| No dependencies installed at all | Pure stdlib BM25 pipeline. Full score, no degradation in schema or coverage. |
| Checkpoints vendored, no API key | Cross-encoder and Tier 2 run locally. **Zero network calls at any point.** |
| Network disabled at scoring time | `load_llm_reranker()` returns `NullLlmReranker` — it never probes an endpoint at construction, precisely so it cannot hang on a rig with the network off. If a call is somehow attempted and fails, `safe_rerank` catches it and keeps the cross-encoder's order. |

The system **cannot fail for want of a key**. It has no offline/online modes —
it has one path with one optional re-ordering step.

### Cost, tokens, latency

| Disclosure | Value |
|---|---|
| **Token usage — core** | `0` prompt, `0` completion, every turn. Truthfully, because no model is called. |
| **Token usage — LLM layer, when it fires** | ~4,640 prompt (top-50 window), ~30 completion (10 integer indices, no rationale). Measured on the one successful live smoke call: 212 prompt / 9 completion on a 5-candidate synthetic query. Reported honestly through `usage.prompt_tokens` / `usage.completion_tokens`; `(0, 0)` on every turn it does not fire. |
| **Estimated model cost** | **$0.00** for the core and both local checkpoints. The LLM layer is ~**$0.011 per turn it fires on**, and against this simulator it fires on ≈5.5% of turns by construction. |
| **Latency — core** | Index build ~1.16 s once at construction; ~**19 ms per turn** end to end. 200 sessions / 571 turns in 9.7 s. |
| **Latency — with the cross-encoder** | ~**1.2 s per turn** (single rig, top-50 window). This is a timeout/disqualification risk against a per-turn budget the organizers have never published — it is **not** a score cost, since Efficiency is turn-based and wall-clock never enters `TechnicalScore`. |
| **Latency — LLM layer** | ~0.14–0.42 s added on the turns it fires. Pinned to `thinking_budget=0`: reasoning mode is disqualifying on latency (Gemini 3.5 Flash at `high` effort is 15.28 s to first token). |
| **Memory** | One in-memory SQLite FTS5 index over the 50,000-product catalog. No vector store, no external index — the spec's "must run entirely in-memory" constraint is satisfied by construction. The two local checkpoints add their own footprint when loaded (~22M and ~8M parameters respectively). |
| **API credentials** | One optional environment variable, `GEMINI_API_KEY`. Never committed, never logged, never read by any file in this repo — `google.genai.Client()` reads it from the environment directly. |

### Output contract

`respond()` returns `message` (str), `ask_attribute` (one allowed attribute,
never `null` and never `other`), `recommendations` (up to 10 unique
`parent_asin`, ordered best to worst), and `usage` with non-negative counts.
Every field is coerced through `src/contract.py::validated()` before it leaves.

## Results

Scored with the organizer's own unmodified `evaluate()` over the 200 public
sessions.

**Read both brackets, never one.** `public_set.jsonl` carries no real
`intent_card`, so the evaluator falls back to building the simulated customer's
"hidden" preferences out of the *target product's own listing* — the 94.5%
verbatim-overlap figure above. Leaky is as-shipped; scrubbed patches that leak.
The organizer's held-out set should land between them.

### Offline core (stdlib only — the configuration that needs nothing installed)

| bracket | hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| leaky (upper bound) | 0.9950 | 0.7059 | 2.860 | **0.872057** |
| scrubbed (lower bound) | 0.6600 | 0.2519 | 6.410 | **0.497383** |

Against the superseded `starter/` system, measured under the same conditions on
the same day:

| bracket | `starter/` | `src/` | delta |
|---|---|---|---|
| leaky | 0.692586 | 0.872057 | **+0.179471** |
| scrubbed | 0.198439 | 0.497383 | **+0.298944** |

The gain is *larger* with the leak removed, which is the opposite of a
measurement artifact. The organizer's published BM25 baseline is HitRate@10
`0.125`.

### With the cross-encoder enabled

⏳ **Measurement in progress.** The rows above are the offline core with all
three Layer 3 seams inert — the configuration that was current when they were
recorded (31 Aug 2026). The cross-encoder was enabled on 1 Sep 2026 and **there
is no committed full-public-set score for the enabled configuration yet**; the
run is slow because the checkpoint scores 50 pairs per turn on CPU.

What is measured about the cross-encoder is component-level, not end-to-end:
**+0.047 `TechnicalScore`** with the CI excluding zero, at the top-10 and top-20
windows (`bakeoff/results-part4.json`); at top-50 the delta is +0.017 and its CI
*includes* zero. On the ESCI rig — 600 real Amazon-customer queries with human
relevance labels — the same checkpoint moves MRR@10 from 0.6686 to 0.7173.

**Do not quote the rows above as the score of the shipped configuration** until
this row is filled in. That is exactly the kind of drift this repo's run log
exists to prevent.

## Limitations, and what we would improve with more time

**Both numbers above are this simulator.** The bracket *direction* is the signal,
not either endpoint. We report the spread rather than the flattering end.

**The LLM escalation has never been measured at scale.** Exactly two live calls
have ever been made against it, both smoke tests on 1 Sep 2026. The first found a
real bug (an over-strict response-length check that rejected a usable response;
`safe_rerank` degraded correctly and nothing broke). The second, after the fix,
succeeded end-to-end. **That is a correctness check, not a benchmark**, and no
claim about its aggregate quality or `TechnicalScore` impact should be read into
it. The measurement harness exists (`bakeoff/followup_llmrr_esci.py`) and the
sweep is the first thing we would run with more time.

**The paraphrase probe is the layer's real falsifier and it has not run.** The
honest claim for the escalation is *real-world robustness*, not local score:
against this simulator it fires on ~5.5% of turns by construction, so its
expected `TechnicalScore` delta is ≈0. We disclose that up front rather than let
a reader infer a gain. What would substantiate it is perturbing the disclosed
strings off verbatim and showing the routed layer recovers what BM25 + the
cross-encoder lose. Without that, it is a designed mechanism with a measured
gate, not a proven win.

**The overlap gate re-sorts after the reranker, and overrides its top pick on
roughly half of reranked turns** (40% leaky, 53% scrubbed, measured on 30
stratified public sessions with a pass-through reranker). On the vague queries
the LLM exists to serve, the gate is ranking on exactly the signal that band
lacks. Settling the component order — gate as annotation first, or as a
tie-breaker among equal-overlap candidates only — is cheap, needs no API key, and
should happen before any further model spend.

**Recall caps the whole ranking stack.** A *perfect* top-50 reranker reaches
0.8431 on the vague band of 600 real ESCI queries; the remaining ~15.7% is
retrieval failure, where the target is not in the window at any size. Nothing in
the rerank or escalation layers touches that. If a paraphrase probe shows targets
leaving BM25's top-50, the binding constraint is recall and the answer is query
expansion, not a smarter reranker.

**The cross-encoder's ~1.2 s/turn is unbudgeted.** The organizers have not
published a per-turn timeout. It costs nothing on the metric, but it is a real
disqualification risk and we cannot size it.

**Tier 1's frame decode is exact against the current simulator, and that is a
narrow guarantee.** The organizers reserve the right to add paraphrasing. The
Tier 2 encoder fallback exists for exactly that case and is live, but it fires
zero times against the shipped simulator by construction — so it is a designed
robustness property, not a measured one.

**One reply shape decodes conservatively:** a message combining real content with
a decline phrase is treated as a decline, losing its content. Deliberate — on a
paraphrased set a missed decline is the costlier error.

**The never-repeat rule shipped ahead of the rank-distribution readout the design
asked for first** (`docs/todo.md` item 7). It cannot cost anything — that much is
structural — but it is not yet proven to be what earns the gain.

**Ask-yield adaptive ordering is built but off.** Its premise is under review: only
40 of 200 public sessions reach turn 7 at all, so the band a dynamic schedule
could act on is small, and the whole attribute-selection band measures ≈0.004.

### What we would do with more time, in order

1. Run the LLM sweep through `bakeoff/followup_llmrr_esci.py` and replace the
   two smoke calls with a real number.
2. Build the paraphrase holdout — kept out of the working tree, decrypted only at
   scoring time — and report retrieval survival and rerank survival *separately*,
   so a recall collapse cannot masquerade as a ranking failure.
3. Settle the gate/reranker ordering before spending anything further on models.
4. Measure the cross-encoder at top-20 against top-50 to buy back latency
   headroom, and confirm the ASIN-vs-integer token ratio the output-contract
   argument rests on.

## Team contributions

| Member | Contribution |
|---|---|
| **Darren Sim** (`darrensimmx`) | Architecture and design of record; constraint ledger and retrieval spine; ask policy and the seven-slot schedule; the overlap gate and routed LLM escalation; benchmark tracking and the leak-bracket methodology. |
| **Raphael Lim** (`Raphael2908`) | The clean-room `src/` rebuild; the bracket-mixing fix in `scripts/evaluate_src.py`; docs/pipeline cross-repo sync tooling; attribute-weighting investigation. |
| **Reagan Chow** | Cross-encoder checkpoint comparison at depth 50; ZeroEntropy `zerank-1-small` evaluation and the 50-session sampling rig. |
| **Jamison Teng** | P1 offline-safety hardening: guarding the evaluator-unwrapped surfaces (`__init__`, `reset`) and the response-schema coercion layer. |

*Derived from this repository's commit history; see `git shortlog -sne`.*

## Where the reasoning lives

Planning and design rationale live in a **separate repository**,
[`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs).
That repo is the *why*; this one is the *how*. When a change needs a rationale —
why BM25 and not dense, why the seven-slot schedule and never `other` — the
answer is in `project/standing-findings.md` there, or in `docs/artifacts` here.

- [`report.md`](report.md) — the LLM escalation's window sizing, output contract,
  and model selection, with its own revision log.
- [`docs/todo.md`](docs/todo.md) — decisions deliberately left open, with evidence.
- [`docs/hard-rules.md`](docs/hard-rules.md) — the normative rules every component obeys.
- [`docs/windows-dev-setup.md`](docs/windows-dev-setup.md) — the verified setup guide.
- [`docs/benchmark-tracking-plan.md`](docs/benchmark-tracking-plan.md) — how scores
  get tracked so each change's effect is visible in git history.

Competition kit (organizer-owned, read-only):
[`TechJam2026/techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search).

---

# Internal phase plan (superseded)

Team history, retained for provenance. The current design lives in
`docs/artifacts`; open decisions live in `docs/todo.md`.

## Sequencing rationale

Ship a minimal, safe, *scoring* pipeline first, then layer in the features the
evidence supports — not the ones that sound impressive.

1. **Get something that scores at all**, safely, before making it smarter. The
   single biggest lever was not a model: it was accumulating every disclosed
   constraint into the retrieval query, unconditionally, every turn.
2. **Verify the retrieval assumption before building on it.** Retrieval gates 80%
   of `TechnicalScore` (HitRate 50% + MRR 30%). The bake-off settled it: dense
   fusion loses, cross-encoder rerank wins at +0.047 with the CI excluding zero.
3. **Fill the turns a fixed schedule cannot reach** (ask-yield). Sequenced after
   retrieval because its yield signal reads retrieval's candidate-pool churn —
   the dependency runs one way.
4. **Add contradiction detection last.** Its ceiling on 80% of `TechnicalScore` is
   ≤0.0078. It ships because it is in the rubric and it is the innovation play,
   not because it is where early effort goes. What shipped is the *slot-value*
   check — exact string comparison, hard trigger, zero model.
5. **Harden for submission** — prove the offline path with networking actually
   disabled, and package per the organizer's reproducibility rules.

## Phases

| Phase | Scope | Status |
|---|---|---|
| **0 — Environment & baseline** | Vendor `evaluator/` unmodified; download the catalog; reproduce the published BM25 baseline (HitRate@10 ≈ 0.125). | ✅ done |
| **1 — Foundational pipeline** | A real, scoring `Agent`: constraint ledger, fixed schedule, BM25 over the accumulated ledger string, always-return top-10, plus two dev CLIs. | ✅ done (`starter/`, since superseded by `src/`) |
| **2 — Retrieval verification & rerank** *(was Phase 3)* | Re-run the retrieval bake-off; add cross-encoder rerank only if the numbers justify it, and only off the critical path. | ✅ done — dense rejected, CE shipped 1 Sep 2026 |
| **3 — Ask-yield adaptive ordering** *(was Phase 2)* | Replace the fixed order with one that adapts to observed yield. | ⏸️ built, disabled. Premise under review — only 40/200 sessions reach turn 7. |
| **4 — Intent classifier & contradiction detection** | Per-turn Buying/Browsing/Override decode; slot-value contradiction diff; Tier 2 semantic fallback. | ✅ done — Tier 1 regex + Tier 2 rung-3 centroid, both live |
| **5 — Hardening & submission** | Offline verification with networking disabled; the method/limitations report; package per `submission_rules.md`. | 🟡 in progress — this README is part of it |

## Phase numbering

**This repo's `Phase 0`–`Phase 5` is the single canonical execution plan.** Cite
it as "Phase N" in issues, PRs, branch names, and benchmark labels.

Two other schemes exist in the planning repo and are **not** interchangeable —
the same token means different things:

| This repo | Planning repo `G1–G5` (architecture gates) | Planning repo `1–9` (superseded) |
|---|---|---|
| Phase 0 — Environment & baseline | — | 1 (partial) |
| Phase 1 — Foundational pipeline | G1 offline safety · G2 ledger + ask · G3 fixed schedule | 1, 4 |
| Phase 2 — Retrieval verification & rerank | — | 2, 3, 9 |
| Phase 3 — Ask-yield | G4 ask-yield | — |
| Phase 4 — Intent classifier & contradiction | G5 intent override | 5, 6, 7, 8 |
| Phase 5 — Hardening & submission | — | — |

The planning repo's key was relabelled `P1–P5` → `G1–G5` on 28 Aug 2026 to end a
direct collision. `G1–G5` describes *architecture gates* (what must be green
before what) and deliberately has no slot for environment, retrieval, or
hardening; it is not a delivery schedule.

**Phases 2 and 3 were swapped on 28 Aug 2026** (retrieval was Phase 3, ask-yield
was Phase 2). Pre-28-Aug references use the old numbering; where a number and a
work item disagree, **the work item named alongside it is authoritative**.

## Cross-repo sync checks

`.github/workflows/docs-sync.yml` runs `scripts/check_docs_sync.py` on every push
to `main`, weekly, and on demand. It fetches `darrensimmx/techjam2026-docs` and
checks a curated table of facts against this repo's actual code — a regression
net for specific claims that have already drifted once. `techjam2026-docs` runs
the mirror check, `scripts/check_pipeline_sync.py`, against this repo.

**One-time setup (not done by CI itself):** both repos are private, so each
workflow needs read access to the other. Create one fine-grained PAT scoped to
read-only **Contents** on both repos, then add it as a repository secret named
`TECHJAM_CROSS_REPO_TOKEN` in *each* repo's Settings → Secrets and variables →
Actions. Without it the workflow still runs and reports which checks it could not
fetch, rather than failing silently.

Extend `FACTS` (or the docs repo's mirror table) whenever a review catches a claim
in one repo that does not match the other's actual state.
