# Hard rules — implementation constraints

**Status:** ✅ current. **Normative for `starter/`.**

**Authoritative source: `techjam2026-docs/project/hard-rules.md`.** That file carries the
evidence, the measurements, and the scope limits. This file is the short list of what
those rules *bind in this repo* — it deliberately does not restate the reasoning, so it
cannot drift from the evidence. If the two disagree, the docs repo wins.

All rules were established 28 Aug 2026 against
`evaluator/local_evaluator.py` and the 200-sample public set.

## The rules

1. **Always send a real `ask_attribute`. Never `null`.** A null ask returns the
   "Those options are not quite right yet" template, which the ledger drops — the query
   is unchanged and the turn teaches nothing. Asking costs nothing in the scoring, so a
   null ask is weakly dominated. Measured: the 160 null turns currently sent across
   turns 7-10 gained 0 constraints and produced 0 hits.
   *Binds:* `starter/scheduler.py` must not return `None` while any askable attribute
   remains.

2. **Always return the full top-10, every turn.** Sending fewer is *legal* (the contract
   has `maxItems: 100` and no `minItems`), and a top-1-then-top-10 policy measures
   **+0.018602** — but it is **declined** as harness-gaming, on the same standard that
   rejected the `"other"` short-circuit at +0.042. Do not implement it.
   *Binds:* `starter/agent.py` recommendation slice.

3. **Parse customer replies with regex. Never a model.** Every customer utterance is one
   of 8 f-strings in `local_evaluator.py`. A classifier cannot beat a substring check
   already at 100%, and would put a model dependency on the critical path — the
   "don't score zero" failure mode.
   *Binds:* `starter/ledger.py`. No model import, no network call, on this path.

4. **Split the two declines.** `_CONTENT_FREE_PATTERNS[0]` currently matches both the
   boundary refusal and genuine exhaustion via `(?:a|an\s+additional)`. They mean
   opposite things: the refusal returns at `local_evaluator.py:169` *before* the
   constraint filter runs (bucket never opened), while exhaustion returns at `:183`
   *after* it ran and found nothing (bucket verified empty). Discriminate on the literal
   token `additional`. Both still get dropped from the query; only the classification
   differs.
   *Binds:* `starter/ledger.py:23-31`.

5. **Re-ask a burned attribute; never re-ask an exhausted one.** A refusal means the
   customer declined to look, not that the bucket is empty — re-asking recovers
   constraints in 25 of the 40 sessions that have a burned ask. An exhausted attribute
   can never refill, because `disclosed` only grows. An attribute that returned exactly
   2 may have been truncated by the `[:2]` cap and is also worth re-asking.
   *Binds:* `starter/scheduler.py` tail policy, `starter/ledger.py` yield tracking.

6. **Accumulate constraints verbatim. Never erase on intent override.** `old_value` and
   `new_value` are both generated from the same target listing, and `old_value` is never
   added to `disclosed` — so the "abandoned" preference still describes the target and
   still helps retrieval. Implementing Pillar II's literal "slot erasure" loses score.
   *Binds:* `starter/ledger.py` — append-only, no invalidation path.

7. **Do not build a scenario classifier.** Type is determined for free by the first ask:
   the opening message identifies buying and intent_override, and browsing vs boundary
   resolves on the first non-null `ask_attribute`, because the boundary branch at
   `local_evaluator.py:168` is the first branch in `customer_reply` and must fire.
   *Binds:* nothing to build. Read it off the reply.

## Scope

All of the above is stated against `evaluator/local_evaluator.py`.
`competition_specification.md` reserves the right to add natural-language paraphrasing.
If added, reply *frames* change — but hits are exact `parent_asin` matches, so
paraphrasing can degrade parsing only, never scoring, and rule 5's bias toward treating
ambiguous declines as refusals caps that cost at one re-ask on an otherwise idle turn.
