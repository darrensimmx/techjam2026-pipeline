# Findings — TechJam Statement 4 (Conversational Search)

Working notes from source analysis of `techjam-conversational-search`, cross-checked
against `statement4diagrams-v3.html`.

> **Status — added 2026-08-28 when this file was brought under version control.**
> These are working notes written before a catalog was available locally, so §7
> and §8 list several things as unmeasured that now are. Corrections and
> additions since:
>
> - **The catalog is no longer a blocker.** `§8.1` ("Download `catalog.jsonl` —
>   nothing runs end-to-end without it") is handled: the blob is already in the
>   local object store on `origin/benchmark-tracking`, and
>   `python3 .claude/skills/run-sol/bench.py setup` restores it in ~2s.
> - **`§7`'s unverified score figures now have a real number.** Current `main`
>   scores `0.692586` (hit@10 0.80, MRR 0.525619, MTTC 4.255) against the
>   organizer's `0.10671` baseline. See `docs/ledger-freeze-regression.md`.
> - **`§4`'s "no timeout anywhere" is independently re-confirmed** against
>   `evaluator/local_evaluator.py` — no `timeout`, `signal`, `alarm`, or
>   `threading.Timer` appears anywhere in the harness.
> - **`§8.2` (validate the lexical ceiling) and `§8.3` (ship
>   `classify(msg) -> Event`) remain open.**
>
> The reference to `statement4diagrams-v3.html` throughout is to a file that
> lives outside this repo; §6's corrections are recorded here for the record.

**Provenance markers** — every claim below is tagged:

- `[verified]` — confirmed by reading or executing the organizer's code this session
- `[diagram]` — asserted in v3 of the architecture doc, **not** independently re-checked
- `[approx]` — approximate figure, not measured here

All line references are `evaluator/local_evaluator.py` unless stated otherwise.

---

## 0. Scope constraints that shape everything

`[verified]` `docs/competition_specification.md:11-13`

- **In scope:** keyword / dense / hybrid retrieval, Buying-Browsing routing, query
  rewriting, semantic reranking, conversation state, clarification strategy,
  anonymized-profile use, "legally accessible LLM APIs **or local models**".
- **Out of scope:** catalog modification, identifiers outside the frozen catalog,
  private-label reconstruction, real transactions, mandatory UI, **full-model
  training**, multimodal systems, infrastructure-heavy vector DBs.

"Full-model training" being out is the constraint with teeth: it rules out
fine-tuning any pretrained encoder. Frozen encoder + linear head or cosine
threshold is fine; SetFit's standard contrastive body fine-tune probably is not.

`[verified]` The starter agent has **zero ML dependencies** — `json`, `re`,
`sqlite3`, `pathlib` only (`starter/agent.py:1-6`). Everything is ours to add.

`[verified]` `data/catalog.jsonl` is **not in the repo** — gitignored, downloaded
separately from a GitHub Release (50,000 rows). Nothing runs end-to-end until
it lands.

---

## 1. The anonymized preference profile

Delivered **once** per session via `reset(session_id, user_profile)`, before any
customer message, and never updated.

`[verified]` Schema — `docs/agent_api_contract.json`, all five fields required,
`additionalProperties: false`:

| Field | Type | Observed across the 200 public sessions |
|---|---|---|
| `purchase_frequency` | string | **Constant** — `"3-4 prior purchases"`, all 200 |
| `average_prior_rating` | number \| null | 5.0 (134), 3.0 (22), 4.0 (21), 1.0 (14), 2.0 (9) — never null |
| `rating_style` | string | `usually positive` (134), `critical` (45), `mixed` (21) |
| `preference_tags` | string[] | 1–4 tags from a 9-word vocabulary |
| `summary` | string | One templated sentence |

`[verified]` Tag vocabulary and counts: fit 163, material 154, comfort 144,
style 101, durability 47, performance 26, warmth 18, weather 12,
"general shopping" 1. Tags per profile: 4 tags (121), 2 (43), 3 (30), 1 (6).

`[verified]` Session labels: `category_bucket` is `clothing` for all 200.
Scenario mix buying 80 / browsing 80 / intent_override 30 / boundary 10.
Difficulty easy 80 / medium 90 / hard 30.

### Redundancy — two of five fields carry no information

`[verified]` **`summary` is fully derivable.** It is exactly
`f"Prior purchases emphasize {', '.join(tags)}; ratings are {rating_style}."`
— checked all 200, zero mismatches. Do not spend tokens parsing it.

`[verified]` **`rating_style` and `average_prior_rating` are a deterministic
pair.** 5.0 → positive, 4.0 → mixed, {3.0, 2.0, 1.0} → critical. One signal,
not two.

`[verified]` **`purchase_frequency` is constant** in the public set. No
discriminative value. Cannot confirm the private 800 sessions differ.

### It is a weak retrieval signal

`[verified]` 125 distinct profiles across 200 sessions; the most common profile
repeats 26 times and maps to different hidden targets. Every session is
`clothing`, and the tags concentrate in a four-word head
(fit / material / comfort / style).

**Usable for:** tie-breaks among near-equal candidates when a rarer tag
(`warmth`, `weather`, `performance`) matches product text; choosing which
attribute to ask first. **Not usable for:** narrowing 50k products.

`[verified]` Nothing scores our use of it — the starter ignores it
(`starter/agent.py:73`) and the evaluator only passes it through (line 231).

---

## 2. Intent classification

### The profile is not an intent-classifier input

`[verified]` Measured two ways against `scenario_type`:

- **Permutation test.** Raw mutual information looked promising for
  high-cardinality features (`tag_set` 0.441 bits) but that is cardinality bias:
  shuffled labels give 0.414 ± 0.037 on the same feature. Excess ≈ 0.03 bits,
  inside noise, for every field.
- **5-fold CV accuracy**, majority baseline 40.0%: `rating_style` 37.0%,
  `n_tags` 43.0%, `tag_set` 33.0%, whole profile 35.5%. **At or below baseline.**

Structurally it could not work anyway: `reset()` fires before the first message,
and 15% of sessions flip intent at turn 3 or 4. A static session-level object
cannot represent a mid-session event.

### "Buying vs Browsing" is the wrong axis

`[verified]` `scenario_type` is read on the reply path in exactly two places —
`initial_message` (154-163) and the boundary branch (168). `customer_reply` is
otherwise scenario-independent.

`[verified]` Drove one target through both scenarios with an identical ask
schedule. **Streams converge at turn 3 and the final disclosed sets are
identical.** The only difference: `hard_constraints[0]` is disclosed at turn 1
(buying) vs turn 2 (browsing).

So Buying-vs-Browsing is a **one-turn timing difference, not a session
property**. Re-running that classifier every turn re-derives a label that stops
meaning anything after turn 2.

> **Action:** drive track selection off **ledger state**, not a turn-1 label:
> `if ledger: hybrid + filter else: dense only`. If a "browsing" label latches at
> turn 1 and stays dense-only, 40% of sessions ignore the accumulating ledger —
> which contradicts the doc's own Phase 2 headline.

### What should run every turn: event detection

`[verified]` The simulated customer emits exactly **eight** templates. A
substring classifier gets **8/8** — no ambiguity, no tie-breaks:

| Event | Discriminating substring | Action |
|---|---|---|
| `OVERRIDE` | `ignore my earlier preference` | clear slot, rebuild query |
| `BOUNDARY_REFUSAL` | `I don't have a preference for` | mark attribute dead, advance |
| `NO_INFO` | `no additional preference for` | attribute exhausted |
| `BAD_ASK` | `Ask me about one specific attribute` | we sent `ask_attribute: null` |
| `DISCLOSURE` | `For that, what matters is:` | append to ledger |
| `OPEN_CONSTRAINED` | `A key requirement is:` | turn 1, buying |
| `OPEN_VAGUE` | `still exploring` | turn 1, browsing **or boundary** |
| `OPEN_OTHER` | (fallthrough) | turn 1, **intent_override** |

Two consequences worth exploiting:

`[verified]` **Boundary is unobservable at turn 1** — it shares the browsing
template verbatim (163). Treat as browsing until a clarification comes back
refused.

`[verified]` **Override is detectable at turn 1, free.** Override sessions open
as `I'm looking for {cat}. {old_value}`, matching neither other template, so they
fall through. That tells us at turn 1 that the turn-1 constraint is the one that
will be discarded — and that hits do not count until the override lands
(line 252: `if override_applied and target in ranked`). Do not over-weight that
constraint. v3 parks override at Phase 8; this signal is available at turn 1.

`[verified]` The override regex also catches the empty-`new_value` fallback
message (264), which shares the `ignore my earlier preference` substring.

---

## 3. Offline intent classifiers — options

Decision: go offline. Given §0, the question is which **frozen** encoder gets a
linear head or cosine threshold, not which model to fine-tune.

| Option | Size | Extra training | Notes |
|---|---|---|---|
| Substring / regex | 0 | none | 8/8 today `[verified]` |
| TF-IDF → LogisticRegression | ~50 KB | linear head only | trains <1s; label cost zero (templates known) |
| Model2Vec `potion-base-2M` | 1.8M params | none | static embeddings, no transformer forward pass |
| Model2Vec `potion-base-8M` | 7.5M params | none | sweet spot if a real encoder is wanted |
| Model2Vec `potion-base-32M` | 32.3M params | none | distilled from `bge-base-en-v1.5` |
| MiniLM bi-encoder | ≈90 MB `[approx]` | none | likely already in the stack |
| SetFit | 90 MB+ | fine-tunes body | probably out of scope per §0 |

Model2Vec figures are from the maintainers' current model table (Context7,
`/minishlab/model2vec`). It bridges into sentence-transformers via
`StaticEmbedding.from_model2vec(...)`.

### Recommendation

**Do not build a model yet.** Build the interface — one
`classify(msg) -> Event` with the regex behind it — and leave an embedding
fallback as a drop-in.

- Regex is at 100%. A model cannot beat it, only tie it while adding load time
  and an exception path.
- The only thing that breaks regex is the organizer adding NL paraphrasing
  (`competition_specification.md:38`), flagged as possible, not committed. Note
  it **cannot decide correctness** — hits stay exact code matches.
- Training data is free on demand: the eight templates are known, so paraphrase
  each 50 ways and train a 50 KB logistic regression in an afternoon *if*
  paraphrasing appears.

If an encoder ships anyway for the write-up: **reuse the browsing-track dense
bi-encoder.** Embed the eight templates as prototypes at startup, cosine-match,
threshold. Zero extra megabytes, and "training" means computing eight centroids.
Shipping a second model pays twice for disk, load time, and failure surface.

---

## 4. Evaluator failure semantics

`[verified]` The v3 doc's central risk claim is accurate — lines 239-242:

```python
try:    response = agent.respond(session_id, user_message, turn, TOP_K)
except Exception:
        response = {"message": "", "ask_attribute": None, "recommendations": []}
```

Two things v3 does **not** say:

`[verified]` **The shape check is equally silent** (243-244). A response that is
not a dict, or whose `message` is not a `str`, is replaced by the same empty
dict. A malformed payload zeroes the turn exactly like a raised exception.

`[verified]` **The evaluator enforces no timeout anywhere.** Grepped — nothing.
The bare `except` catches a *crash* but not a *hang*; a stalled network call
hangs the entire run indefinitely. Our own timeout is not belt-and-braces, it is
the only guard.

`[verified]` **Token usage is not scored.** `reported_token_usage` (289-291) sits
outside `recommended_technical_score`. No token penalty — an LLM costs score
nothing, only risk.

---

## 5. Retrieval — why BM25 works, and where it is weak

`[verified]` **Constraints are mined from the target's own indexed text.**
`intent_card()` (52-71) takes `features` and `details` verbatim, plus a
material/colour regex over `searchable_text()`. So disclosed strings are, by
construction, strings present in the target document. This is near-exact string
recovery, not semantic matching — the structural reason concatenation is the
score lever.

`[verified]` **Constraint ordering is deliberate** (59-63):

```python
candidates.insert(0, material)            # "cotton"
candidates.insert(1, f"color: {color}")   # "color: blue"
candidates.append(f"budget around ${p}")  # appended last
hard_constraints = cleaned[:2]            # → usually [material, "color: X"]
soft_preferences = cleaned[2:4]           # → the real feature strings
```

**Turn 1 is much weaker than it looks.** A buying session opens with
`coarse_category` plus a bare material word — *"I'm looking for Shirts T-Shirts.
A key requirement is: cotton."* `cotton` matches an enormous slice of a 50k
clothing catalog. Browsing gives the category alone. The strings that actually
identify the product are the `soft_preferences` (verbatim feature/detail text),
and they only arrive on turns 2-4 when the right attribute is asked.

> BM25's job is not "find it from the query" — it is **"accumulate enough
> verbatim strings that the conjunction becomes unique."**

`[verified]` **`coarse_category`** (126-134) returns the last two category path
segments, excluding the `Clothing, Shoes & Jewelry` top level — e.g.
`"Shirts T-Shirts"`. The product title is *not* leaked at turn 1 (235); it
appears in constraints only as a fallback when a product has no
features/details (65-66).

### Highest-leverage implementation detail

`[verified]` **Mirror `SEARCH_FIELDS` exactly in the index** (line 22):

```python
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
```

That tuple is what the organizer mines constraints from. Indexing only
title + description loses every constraint sourced from `details` — and
`details` is a dict flattened to `"{key}: {value}"` (40-46), so
`"Fit type: Regular fit"` is a literal disclosed string. Index it flattened the
same way and BM25 recovers it nearly free.

### Known noise injectors

Under a strict "no parsing" ledger:

- `"color: blue"` contributes a junk `color` token (`blue` is the signal).
- `"budget around $24.99"` contributes `budget` / `around` plus a price string
  that will not match text fields. `[verified]` Rare — appended last (63), so it
  only reaches `soft_preferences` on products with ≤3 feature/detail entries.

`[diagram]` v3 reports typed slots vs raw string measured +0.000000 across 200
paired sessions, so this may not be worth revisiting — but these two formats are
where it would show up if anywhere.

`[diagram]` v3's "84% of disclosed constraint strings are substrings of the
target's indexed text" is near-tautological by the construction above. The
interesting part is the ~16% that are not — precisely the two synthesized
formats named here.

---

## 6. Corrections to `statement4diagrams-v3.html`

| Panel | Current | Should be |
|---|---|---|
| 1, box 3 | "Intent Classifier — Buying vs Browsing, re-runs each turn" | Event classifier (§2 table). Buying/Browsing does not survive turn 2. |
| 1, tracks | Branch on turn-1 intent label | Branch on ledger state |
| 2, box 2 | "Intent Classifier + State Machine" as a separate component | Collapse into the Constraint Ledger; keep the box for event detection + state |
| 1, box 7 / 2 | LLM Semantic Rerank as a networked optional layer | See §7 — resolve the Pillar I question |
| 3, Phase 8 | Intent override at Phase 8 | Turn-1 detection is free; the *handling* can stay late |

---

## 7. Open / unverified

- **Does Pillar I demand a *prompted LLM* specifically, or just *semantic
  reranking*?** Not answerable from the repo — the text is in the external
  problem statement. This is the only thing keeping a network dependency in an
  otherwise fully offline architecture. If it just wants semantic reranking, the
  Phase 9 cross-encoder becomes the Phase 6 implementation, the entire
  "OPTIONAL — network" container disappears, and §0 already permits
  "local models". Strictly better design: same rubric coverage, no hang risk,
  no fallback edge to test.
- **Will the organizer add NL paraphrasing?** `competition_specification.md:38`
  flags it as possible. Determines whether §3's tier 2 is ever needed.
- **Whether freezing SetFit's body counts as "full-model training".** Worth
  clearing with organizers rather than discovering at judging.
- `[diagram]` v3's score figures (0.16 → 0.75, 0.74679 ask schedule, comparator
  0.17–0.23) are unverified here — no catalog yet.
- `[approx]` The MiniLM ≈90 MB figure was not checked against source.

---

## 8. Next steps

1. **Download `catalog.jsonl`** — nothing runs end-to-end without it.
2. **Validate the lexical ceiling without running a session.** For each of the
   200 public samples: build the intent card, concatenate all four constraints
   plus the coarse category, check what rank BM25 gives the known
   `ground_truth.parent_asin`. ~20 lines, and it bounds the pure-lexical path
   before anything is built on top.
3. **Ship `classify(msg) -> Event`** behind a swappable interface, regex-backed.
4. **Re-read Pillar I** and settle §7's first bullet before building the
   reranker.
