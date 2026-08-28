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

5. **After a refusal, ALWAYS re-ask that attribute — unconditionally.** Not
   conditionally, not "if the ledger looks thin". A refusal means the customer declined
   to look, not that the bucket is empty: it returns at `local_evaluator.py:169` before
   the constraint filter at `:178` runs. Re-asking recovers constraints in 8 of the 10
   boundary sessions and 25 of the 40 sessions with a burned ask. You cannot predict
   which will pay and you do not need to — the re-ask lands on an idle turn where the
   alternative is a null ask that freezes the query, so a re-ask that recovers nothing
   costs **zero**.
   *Timing:* later, not immediately — the refusal consumed its turn either way, and an
   immediate re-ask would displace a never-probed attribute. Put it on the first idle
   turn (turn 7+ under the current schedule).
   *Priority:* a burned attribute outranks an overflow candidate (one that returned
   exactly 2 and may hold a third under the `[:2]` cap). At most one attribute per
   session is ever burned, so this is a single slot, not a queue.
   *The mirror image is equally binding:* **never re-ask an exhausted attribute.** The
   filter ran and found nothing, and `disclosed` only grows, so it can never refill.
   Provably worthless, not merely unlikely.
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
