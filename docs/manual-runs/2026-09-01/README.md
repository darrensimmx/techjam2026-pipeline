# Manual turn-by-turn runs against `src/` — 1 Sep 2026

Hand-driven sessions against the **submission** agent (`agent.py` → `src/`), one
pair per evaluator scenario type. Not a scoring run: no number here is
comparable to `results_src.md`. The question was narrower — *does the latest
`src/` behave correctly when a human drives it, turn by turn, including in
phrasings the vendored simulator never emits?*

Reproduce any of these with:

```bash
python3 scripts/manual_src_session.py --catalog data/catalog.jsonl \
    --messages docs/manual-runs/2026-09-01/<name>.json --top 5 --target <parent_asin>
```

`src/` was at 88b3b91 (`Rebuild the agent as a clean-room src/ tree`), catalog
`data/catalog.jsonl`, `degraded=False`, `top_k=10`, 10 turns each. The full
suite was green at 390 tests on the same tree.

| transcript | scenario | phrasing | target | first hit | best rank |
|---|---|---|---|---|---|
| `buying_template.log`      | buying          | harness templates | B07QMS8TX8 | turn 2 | 1 |
| `buying_natural.log`       | buying          | free-form human   | B07QMS8TX8 | turn 1 | 1 |
| `browsing_template.log`    | browsing        | harness templates | B07Q46M2J2 | turn 2 | 6 |
| `browsing_natural.log`     | browsing        | free-form human   | B07Q46M2J2 | turn 5 | 8 |
| `boundary_template.log`    | boundary        | harness templates | B0BN6CCHB7 | turn 4 | 2 |
| `boundary_drain.log`       | boundary        | 9 declines in a row | –      | –      | – |
| `override_template.log`    | intent_override | harness templates | B0C65TFM9F | turn 1 | 1 |
| `override_late_natural.log`| intent_override | late + paraphrased | B09YMTWDXJ | turn 3 | 1 |
| `boundary_hedge_probe.log` | probe           | retired-vs-hedge  | –          | –      | – |
| `probe.log`                | probe           | malformed input   | –          | –      | – |

## What held

Across 80 graded turns, every hard contract invariant held:

- exactly 10 in-catalog recommendations on every turn, including on a fully
  drained ask ladder and on malformed input;
- `ask_attribute` always a member of `ALLOWED_ATTRIBUTES`, never `null`, never
  `other` — 80/80;
- no product recommended twice in a session, except where an override's
  `restore_all()` legitimately put earlier items back in play;
- the ledger append-only and never shrinking, including across an override
  (hard rule 6); content-free replies appended nothing;
- the exact four-key schema every turn, and no raise from `None`/int/dict
  messages, a non-integer turn, FTS5 operators, or an un-`reset()` session.

Two mechanisms were observed *changing the outcome*, not merely setting a flag:

- **the decline split** (`src/frames.py`) — in `boundary_template.log`, "I don't
  have **a** preference for material" left `material` live and it was re-asked
  at turn 8 and answered, while "I don't have an **additional** preference for
  color" retired `color` for the rest of the session;
- **the override guard** (`src/shown.py`) — `suppressed=True` from turn 1 kept
  the pre-override top-10 unrecorded, so turn 4, the first *scored* turn,
  returned the target at rank 1 rather than a second-best list.

`buying_natural.log` and `browsing_natural.log` are the strongest evidence for
the `src/slots.py` layering claim: under free-form phrasing every turn decoded
as `unknown`, so slots, retirement and scenario detection all went dark for ten
turns — and both sessions still hit, because the ledger appends verbatim and the
concatenation is the query. A parsing failure corrupted *which question we
asked* and never *what we searched*.

## What did not

Three findings, none a contract violation, ordered by how much they would cost
on the private set:

1. **Override detection has no paraphrase tolerance.** `_F8_OVERRIDE` is
   anchored on the literal "ignore my earlier preference". In
   `override_late_natural.log` turn 6, "Actually, scrap that — … what I really
   want is a water resistant watch" decoded as `unknown`: no shown-restore, no
   slot contradiction. Correct against the vendored harness, which emits only
   the eight templates; the organizers reserve the right to paraphrase, and
   this is the one frame with no fallback (Tier 2 is inert).
2. **Every utterance enters the query at equal weight**, so a chatty customer
   dilutes it. `buying_natural.log` turns 5 and 7 (sizing chat, a shift-length
   aside) pushed the top-10 to pajama sets and a novelty t-shirt; a
   category-overlap proxy against the target reads `10,6,9,5,3,3,3,5,10,6`
   versus `10,10,10,10,9,9,9,9,6,6` for the template run. No cost here — the
   hit was banked on turn 1 — but it is where a late-hit session would lose.
   The same root cause makes the Tier-1.5 hedge miss "no strong preference on
   color really" and append the whole content-free sentence to the query.
3. **The hedge rung ignores `retired`.** `src/askpolicy.py:259` filters
   `HEDGE_ORDER` by `asked` only, while the schedule (`:224`), the overflow
   branch (`:248`) and the floor (`:265`) all filter `retired` as well.
   Unreachable under the vendored simulator, whose replies always name the
   attribute just asked; reachable when a customer volunteers "I don't have an
   additional preference for brand", since `src/frames.py` binds exhaustion to
   the attribute *named in the text*, not to `last_ask`. Cost is one wasted ask
   on something the customer already declined. `boundary_hedge_probe.log`.

Asking `brand` and `category` on turns 8–9 is **not** on this list: both are in
`ALLOWED_ATTRIBUTES`, and `src/askpolicy.py` step 2.iii prices them at zero
deliberately and says not to optimise them away.
