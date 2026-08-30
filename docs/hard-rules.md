# Hard rules — implementation constraints

**Status:** ✅ current. **Normative for `starter/`.**

**Authoritative source: `techjam2026-docs/project/hard-rules.md`.** That file carries the
evidence, the measurements, and the scope limits. This file is the short list of what
those rules *bind in this repo* — it deliberately does not restate the reasoning, so it
cannot drift from the evidence. If the two disagree, the docs repo wins.

All rules were established 28 Aug 2026 against
`evaluator/local_evaluator.py` and the 200-sample public set.

## The rules

**Rule 0 — TechnicalScore is not the objective. Never rank a change by score delta
alone.** Added 30 Aug 2026. `TechnicalScore` is one objective *input* to the Technical
Execution row, not the row and not the rubric — the specification says so under the
scoring formula, and judges separately assess *"code quality, architecture, reliability,
and the effective use of models or APIs."* So a change measuring ~0 or negative can still
be correct, and one measuring positive can still be wrong. Precedents on the record:
`"other"` declined at **+0.004**, clock-gated withholding declined at **+0.018602**, and
the ledger content-free filter **kept at −0.030232** because the score it gave up was
bought with noise tokens that will not transfer to a private set.
*Binds:* every change to `starter/`. State the architectural and feasibility read
alongside any number. "It measures ~0" does not justify dropping robustness work —
thread safety, degraded-mode signals and offline verification are Feasibility evidence,
scored, just not by the formula. Evidence: docs repo `project/hard-rules.md` → A0.

1. **Always send a real `ask_attribute`. Never `null`.** A null ask returns the
   "Those options are not quite right yet" template, which the ledger drops — the query
   is unchanged and the turn teaches nothing. Asking costs nothing in the scoring, so a
   null ask is weakly dominated. Measured: the 160 null turns currently sent across
   turns 7-10 gained 0 constraints and produced 0 hits.
   *Binds:* `starter/scheduler.py` must not return `None` while any askable attribute
   remains.

2. **Ship the full top-10 every turn — as policy, not because the harness requires it.**
   Sending fewer is *legal* (the contract has `maxItems: 100` and no `minItems`), so the
   permission is structural and the count is our choice. A top-1-then-top-10 policy
   measures **+0.018602** and is **declined** as harness-gaming — whether or not it is
   disclosed — on the same standard that rejected the `"other"` short-circuit at
   **+0.004**. Do not implement a gate on the turn clock. A K chosen from a *measured
   confidence signal* is a different object: **open, not forbidden**, and parked until
   the architecture lands (docs repo `open-questions.md` item 11).
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

5. **After an ask that got no answer, ALWAYS re-ask that attribute — unconditionally.**
   Not conditionally, not "if the ledger looks thin". **Two events burn an ask, and this
   rule binds both** — an earlier phrasing said only "after a refusal", which left the
   larger of the two outside the rule:
   - *boundary refusal* (10 sessions) — returns at `local_evaluator.py:169` before the
     constraint filter at `:178` runs. The customer declined to look; the bucket is not
     empty.
   - *the override turn* (**30 sessions**) — `evaluate()` takes the
     `not override_applied and turn + 1 == override["turn"]` branch and **never calls
     `customer_reply`**, so that turn's ask is never read at all.

   Read the burned attribute from state; do not hardcode it. It is `FIXED_SCHEDULE[0]`
   for a refusal but `FIXED_SCHEDULE[1]` or `[2]` for an override, and all three couplings
   move if the schedule changes. Re-asking recovers constraints in 25 of the 40 sessions
   with a burned ask (8 of 10 boundary, 17 of 30 override). You cannot predict
   which will pay and you do not need to — the re-ask lands on an idle turn where the
   alternative is a null ask that freezes the query, so a re-ask that recovers nothing
   costs **zero**.
   *Timing:* later, not immediately — the refusal consumed its turn either way, and an
   immediate re-ask would displace a never-probed attribute. Put it on the first idle
   turn (turn 7+ under the current schedule).
   *Priority:* a burned attribute outranks an overflow candidate (one that returned
   exactly 2 and may hold a third under the `[:2]` cap). At most one attribute per
   session is ever burned — a session carries one `scenario_type` and both
   `boundary_used` and `override_applied` are one-shot — so this is a single slot
   (`Optional[str]`), not a queue.
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

## Naming — read before writing a component name

Added 29 Aug 2026. Authoritative source: `techjam2026-docs/project/glossary.md`.

- **intent classifier** — the shipping component: a regex **frame decode** (which reply
  branch emitted this string) plus a semantic fallback for the paraphrase case. In this
  repo the Tier 1 half is `starter/ledger.py`'s `_CONTENT_FREE_PATTERNS`. Tier 2 is not
  approved to build.
- **decline split** — refusal vs. exhaustion on the token `additional`. **An output of
  the frame decode, not a component.** Do not give it its own module or diagram box.
  Rule 5 above is the rule; `_CONTENT_FREE_PATTERNS[0]` currently collapses both.
- **scenario classifier** — reserved for the thing rule 7 says not to build. Scenario type
  is a free byproduct of the frame decode.
- **intent trajectory** — the per-turn label sequence and drift check, 🔴 demoted
  29 Aug 2026. Never call it "the intent classifier."
- **trajectory** unqualified in `bakeoff/` means the *query/rank* trajectory across turns
  — unrelated to intent trajectory. Qualify it in new code.

## Scope

All of the above is stated against `evaluator/local_evaluator.py`.
`competition_specification.md` reserves the right to add natural-language paraphrasing.
If added, reply *frames* change — but hits are exact `parent_asin` matches, so
paraphrasing can degrade parsing only, never scoring, and rule 5's bias toward treating
ambiguous declines as refusals caps that cost at one re-ask on an otherwise idle turn.
