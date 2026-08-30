# TBD — open questions

**Status:** 🟡 open. **Binds nothing.** Entries here are unresolved on purpose.

A question lives here while reasonable people on this team disagree, or while the
evidence is real but incomplete. Moving an entry out means one of two things: it
becomes a rule in `docs/hard-rules.md` (it binds `starter/`), or it becomes a
closed entry in `docs/findings.md` (we decided not to build it). **Deleting an
entry without doing one of those two is not allowed** — that is how a decision
gets silently re-made three weeks later.

Represent the side you disagree with at its strongest. An entry that only argues
one way is not an open question, it is a finding in the wrong file.

Opened 30 Aug 2026.

---

## Should the ledger be cleaned on an intent override?

**Opened:** 30 Aug 2026 · **Owner:** raphael · **Blocks:** nothing yet — Phase 4
would inherit whichever answer wins

**The claim.** Append-only is wrong: when the customer overrides a preference,
the superseded constraint should be removed from (or demoted in) the query rather
than left to compete with its replacement. `starter/ledger.py` currently appends
every non-content-free reply verbatim and never invalidates anything.

### Evidence for cleaning

- **The organizer's briefing says so, in as many words.** The Statement 4 deck
  ("Four customer behaviors test different Agent skills", ~31:57) gives an
  override example — turn 1 "black running shoes", turn 3 "Actually, make them
  casual white sneakers" — and scores it: *"Weak Agent: appends contradictory
  words. Strong Agent: replaces black → white and running → casual, then
  reranks."* An earlier slide asks for an agent that "keeps **active**
  constraints", which implies constraints can go inactive.
- **Half the competition score is human-judged.** Devpost's Official Rules put
  four criteria at 25% each; Technical Execution and Innovation & Problem Insight
  are read by people, who have seen that slide. A submission that visibly ignores
  the stated behavior needs to explain itself well.
- **The local score may be resting on a leak.**
  `scripts/leak_controlled_benchmark.py:14` measures that **94% of disclosed
  constraint strings are exact substrings of the target's own indexed text**. If
  append-only wins only because the simulator recites the answer's spec sheet
  back to the agent, that advantage may not survive final scoring — and the
  measurement that "erasure loses score" would be measuring the leak, not the
  policy.

### Evidence against cleaning

- **The simulator's override is not a contradiction.** `behavior_for()`
  (`evaluator/local_evaluator.py:74`) sets `old_value = soft[-1]` (`:79`) and
  `new_value = hard[0]` (`:80`), both from `intent_card(product)` where
  `product = products[target]` (`:208-209`) — the ground-truth listing. Both
  values describe the **same** product. Materialized examples from the public set:

  | sample | `old_value` (deleted by "replace") | `new_value` |
  |---|---|---|
  | `public_0002` | `Buckle closure` | `leather` |
  | `public_0003` | `Stainless Steel Band` | `Water Resistant` |
  | `public_0004` | `Long torso camisole…` | `polyester` |

  There is no "black" to replace with "white". Replacement deletes an accurate
  description of the item being retrieved.
- **The simulator can re-disclose `old_value` later.** `initial_message()`
  emits it at `:160-162` **without** adding it to `disclosed` — unlike the buying
  branch, which does add (`:156-158`). So it remains eligible to come back as a
  fresh constraint from `customer_reply()`.
- **`docs/hard-rules.md:75` (rule 6) already forbids erasure** on exactly this
  basis, and records that literal slot erasure loses score.
- **Override is currently the best-scoring scenario**: hit@10 0.833, MRR 0.657,
  against buying 0.825 / 0.507 and browsing 0.788 / 0.494. Whatever is wrong with
  append-only is not visible in the scenario it supposedly breaks.

### The middle option nobody has measured

**Weighted demote** — multiply the superseded constraint's terms by a factor
(~0.3) instead of dropping them. It is the only policy that is defensible under
both simulators: if `old_value` still describes the target (as it does today) most
of its signal survives; if a future simulator emits a genuine contradiction, a
demoted term is close to inert. It also gives the writeup something honest to say
to a judge holding that slide.

Note this is *not* the same as the weighted-attribute sweep deprioritized in
`docs/findings.md` — that one is about retrieval column weights. This is a
per-constraint recency weight in the ledger. Do not let the shared word merge them.

### The two sides are not arguing about the same score

Worth stating plainly, because it explains why the argument does not resolve: the
repo's evidence measures `recommended_technical_score`. The claim is largely about
the human-judged 50%, which **nothing local measures at all**. The honest outcome
may be "cleaning costs X technical points and buys an unmeasurable amount of judge
credit" — a values call, not a measurement, and it should be recorded as one rather
than dressed up as settled.

### A synthesis that may dissolve this entirely

Keep the ledger append-only **for retrieval**, and acknowledge the override in the
outgoing `message`: *"Got it — dropping the buckle-closure preference and
prioritising leather."*

This costs **exactly zero** technical points. `customer_reply()`
(`evaluator/local_evaluator.py:166`) never receives `message`, and `evaluate()`
reads it only for an `isinstance(..., str)` check at `:243` — `sessions.append()`
at `:269-276` stores `sample_id`, `scenario_type`, `hit`, `first_hit_turn`,
`best_rank`, `reciprocal_rank`, and nothing else. The field is free.

And `message` is precisely what a human judge reads in a transcript. So the
proposal buys the entire human-judged benefit of "visibly handles the override"
without touching the query that the measurements are about.

**Caveat, and it is real:** the local evaluator neither stores nor scores
`message`, so any judged transcript comes from the organizer's harness, not this
one. This cannot be validated locally — only reasoned about. It is cheap enough
that reasoning may be sufficient.

### What would settle it

Three arms over the 30 `intent_override` sessions — append-only (current),
literal replace, weighted demote:

```bash
python3 .claude/skills/run-sol/bench.py eval
python3 scripts/leak_controlled_benchmark.py
```

**The second run is the one that decides it.** The first only tells you which
policy wins on a leaky simulator; the leak-controlled run tells you whether that
ranking is a property of the policy or of the leak. If the ordering flips under
leak control, cleaning wins and rule 6 needs revisiting. If it holds, append-only
survives on its merits and this entry moves to `docs/findings.md`.

**Use a paired test, not a point difference.** n=30 is small and
`bakeoff/results-part1.json` puts `current.r1_bootstrap.sd` at 0.025785 across the
full 200 — a 30-session subgroup delta sits inside that noise. The wins/losses/ties
and `delta_ci` machinery in `bakeoff/part3_fusion.py` is the right instrument;
"the override row went up" is not.

Third question, cheaper than either: **count how often `old_value` gets
re-disclosed** under rule 5's mandated re-ask of the burned attribute. If it comes
back often, erasure is self-defeating in this harness regardless of who is right
in principle.

Do not run either against `evaluation-data/` — that set is test-only.

---

## What is the right granularity for attribute weighting?

**Opened:** 30 Aug 2026 · **Owner:** raphael · **Blocks:** any weighting work,
including the sweep that `docs/findings.md` defers

**The claim.** The FTS5 column weights are the wrong unit. A real shopper cares
about some attributes more than others, and which ones shifts mid-conversation —
so the agent needs per-attribute weighting that adapts as the customer talks.
Tuning six column weights against this evaluator optimises for the harness, not
for a shopper.

### Evidence for — the granularity objection is measured, not theoretical

All six askable attributes collapse into **one** column. Measured across the 200
public targets, 800 disclosed constraint strings:

| column | constraints found in it | current weight |
|---|---|---|
| `features` | **749 (93.6%)** | 2.5 |
| `title` | 43 (5.4%) | 6.0 |
| `description` | 28 (3.5%) | 1.0 |
| `details` | 8 (1.0%) | 2.5 |
| `categories` | 0 (0.0%) | 4.0 |
| `store` | 0 (0.0%) | 1.5 |

Reproduce: materialize each sample with `materialize_hidden_fields()`, then test
each `hard_constraints`/`soft_preferences` string for substring membership in
each column of the target listing.

Two things follow. First, the inherited weights are close to inverted for this
workload — the column carrying nearly all the customer's text is ranked below two
that carry almost none, and `categories` (weight 4.0) never matches anything a
customer says, because `intent_card()` (`evaluator/local_evaluator.py:52`) builds
constraints from `features` and `details` only. Second, and more important:
**material, colour, size, style, use_case and feature all land in the same
column**, so no column weight vector can express "this customer cares about
colour more than size." There is one slider for six attributes.

`bm25()` accepts column weights only — FTS5 offers no per-term weighting — so the
mechanism the claim needs does not exist at that layer at all.

### Evidence against — it cannot be validated here, at all

- The simulator has **no notion of hard vs soft**. `customer_reply()`
  (`evaluator/local_evaluator.py:174-185`) returns up to two matching constraints
  with no strength marker, drawn from `hard_constraints + soft_preferences`
  concatenated. Nothing rewards weighting them differently and nothing punishes
  weighting them badly.
- The 93.6% figure **is the leak**. `intent_card()` lifts sentences out of the
  target's `features` and the customer recites them back — the same mechanism as
  the 94% substring measurement at `scripts/leak_controlled_benchmark.py:14`. So
  "up-weight `features`" is mechanically "up-weight the field the answer is being
  read out of." Under real customer paraphrase that 93.6% collapses.
- Therefore any local number produced by any weighting scheme — column, attribute,
  adaptive — is measuring the harness. **This entry cannot be closed by
  `bench.py eval`.** Saying so is the entry's main job.

### If it is built, build it at the right layer

Not column weights. Two places, both already scaffolded:

- **Score fusion** — one query per attribute group, combined with per-attribute
  weights. `bakeoff/part3_fusion.py` already implements weighted fusion and
  `bakeoff/bm25_scores.py:32` exists to get comparable magnitudes; both are
  currently pointed at BM25-vs-dense instead of attribute-vs-attribute.
- **Rerank** — retrieve wide, rescore candidates on weighted per-attribute match.
  Stronger, because it can ask "does this match the size they stated?", which a
  column weight cannot. It is also where the headroom is: recall@100 is 0.96
  against recall@10 of 0.80, oracle@100 = +0.2478.

**Where a model would earn its place** — not choosing the numbers, but classifying
constraint *strength*: "I need a size 11 wide" (hard, weight heavily or filter) vs
"something casual, I guess" (soft). Bounded, genuinely hard under paraphrase, and
the same Tier-2 shape the glossary already scopes. First pass needs no model —
strength is usually lexically marked (`need`, `must`, `only`, a number with a unit
vs `prefer`, `ideally`). Ship rules behind an interface; swap a classifier in if
paraphrasing arrives.

### What would settle it

Nothing local, and that is the finding. What can be done:

1. **Bound the ceiling before building.** Run the attribute-fusion arm through
   `bakeoff/part3_fusion.py`'s paired wins/losses/ties + `delta_ci` machinery in
   both brackets. If it wins only under the leaky bracket, that is a clean
   negative and costs an afternoon.
2. **Decide it as a product question.** 50% of the competition is human-judged.
   "Column weights cannot distinguish size from colour, so we built per-attribute
   weighting, and we are telling you it does not move our local number because our
   local number is measuring a simulator leak" is a defensible submission — and a
   stronger Innovation & Problem Insight story than a tuned number nobody can
   justify. Record it as a values call, not a measurement.

---

## How to add an entry

```markdown
## <the question, as a question>

**Opened:** <date> · **Owner:** <name> · **Blocks:** <what, or "nothing yet">

**The claim.** <one sentence, in the proposer's own framing>

### Evidence for
### Evidence against
### What would settle it
```

Every bullet carries a `file.py:line` pin or a named source. "I think" is not
evidence; "the slide says, at 31:57" is.
