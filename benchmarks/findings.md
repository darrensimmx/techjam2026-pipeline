# Findings

Living log, append-only. A dated entry each time a `scripts/benchmark.py`
run reveals something worth remembering — a surprising number, a
hypothesis confirmed or reversed, a tuning lever that mattered or didn't.
Pairs with `history.jsonl`: that file has the numbers, this file has the
story behind them. Never delete an entry; mark it superseded if a later
run contradicts it.

## 2026-08-27 — Phase 1 baseline beats the planning repo's projected ask-yield ceiling

> **Superseded (see 2026-08-28 below).** This run's 0.7228 is inflated by a
> leak in the customer simulator, not by the agent understanding intent.
> Under a leak-controlled re-run the score drops to ~0.20 — nowhere near the
> 0.68–0.75 ceiling this entry claims Phase 1 already clears. Left as
> originally written below, per this file's append-only convention; read the
> 2026-08-28 entry before citing anything on this page.

**Run:** `phase1-baseline`, commit `ecacc52`, see `history.jsonl`.

**Result:** HitRate@10 **0.85**, MRR 0.525, MTTC 3.99, TechnicalScore
**0.7228** — against the real 50k-row catalog and the full 200-session
public set (not the synthetic fixture CI uses for fast checks).

**Why this matters:** the planning repo's `decisions/standing-findings.md`
projects ask-yield's full dynamic-ordering system (Phase 2, not built yet)
landing around 0.68–0.75 TechnicalScore, based on an old review doc's
numbers that were explicitly flagged as unverified ("needs a sanity
re-run" — `decisions/open-contradictions.md` item 5). Phase 1 — just the
fixed six-attribute schedule plus unconditional constraint accumulation
into the BM25 query, nothing adaptive — already clears that projected
ceiling on HitRate@10 alone, before ask-yield exists at all.

**This is a real re-run of that unverified number, and it disagrees with
the prior estimate.** Worth carrying back to the planning repo
(`techjam2026-docs`) rather than leaving it sitting here only — it directly
answers `decisions/open-contradictions.md` item 5 and should update
`standing-findings.md`'s "one real lever" section, which currently cites
0.16 / 0.23 / 0.75 as unverified figures from the review doc rather than a
fresh run.

**Per-scenario detail** (the reason to always look past the overall number):

| scenario | n | hit_rate@10 | mrr | mttc |
|---|---|---|---|---|
| boundary | 10 | 0.600 | 0.533 | 6.20 |
| browsing | 80 | 0.863 | 0.487 | 3.96 |
| buying | 80 | 0.875 | 0.514 | 3.41 |
| intent_override | 30 | 0.833 | 0.657 | 4.87 |

- **Boundary is the weak point** — lowest hit rate, worst MTTC, despite
  being the scenario the fixed schedule should handle *most* cheaply
  (customer declines, we retire the attribute and move on). n=10 is small
  enough that this could be noise, but it's the first place to look if
  Phase 2 doesn't move the overall number as expected.
- **Intent Override already has the best MRR (0.657)** despite zero
  override-specific handling existing yet. Consistent with the planning
  repo's own finding that the override scenario's ceiling is dominated by
  ordinary HitRate/MRR, not by detecting the override itself — the
  detection ceiling story is about a different, much smaller lever
  (≤0.0078), not this number.
- **Browsing and Buying are both strong** (>0.86 hit rate) — accumulation
  alone is doing most of the work the planning repo's review predicted.

**Open question for Phase 2:** does ask-yield's dynamic ordering move
Boundary specifically (its actual weak point), or does it only fill turns
7–10 for Browsing/Buying as originally designed? Track this explicitly in
the next benchmark run rather than assuming.

## 2026-08-28 — The 2026-08-27 baseline is inflated by simulator leakage; a controlled re-run gives a much lower bracket

**Run:** `scripts/leak_controlled_benchmark.py`, branch
`investigate/leak-controlled-benchmark`. Not a `history.jsonl` row — this
changes the *customer simulator*, not the agent, so it isn't directly
comparable to the other rows there without this note.

**What was wrong with the 2026-08-27 entry:** `data/public_set.jsonl` never
carries a real `intent_card`/`behavior` — 0 of 200 rows do. The organizer's
own held-out per-session customer data isn't in the public set. The vendored
evaluator's `materialize_hidden_fields()` falls back, for every session, to
`intent_card()`, which builds the simulated customer's disclosed preferences
by lifting sentences straight out of the **target product's own listing** —
full feature bullets, detail values, title fragments. `customer_reply()` then
recites those sentences back to the agent, turn by turn.

Measured: **94% of disclosed constraint strings are exact substrings of the
target's own indexed text.** Confirmed mechanistically too — an agent frozen
to only ever read turn 1 (every later disclosure ignored) already scores
near the organizer's *official published* baseline (HitRate@10 0.185 vs.
0.125). Essentially the entire 2026-08-27 score comes from turns 2–10,
exactly where the simulator reads out more of the answer's own spec sheet. A
BM25 agent doesn't need to understand the customer when the customer is
quoting the product page it's trying to find.

**Controlled re-run:** patched only `intent_card()` — `customer_reply()`,
`initial_message()`, `behavior_for()` are untouched and work unmodified on
the new card — to disclose atomic attribute values only: a material word, a
color word, a short structured detail value (size/fit/style/use_case/
department/occasion/season), a budget number. Never a multi-word span copied
from `features`/`description`/`title`. `evaluator/local_evaluator.py` itself
is never edited; the patch applies to the imported module object at runtime
and is restored before exit. Same 200-session public set, same real 50k
catalog.

| condition | hit_rate@10 | mrr | mttc | TechnicalScore | leak (3-gram overlap) |
|---|---|---|---|---|---|
| 2026-08-27 baseline (leaky, unpatched) | 0.800 | 0.526 | 4.25 | 0.6926 | 0.348 |
| controlled (atomic values only) | 0.255 | 0.100 | 8.95 | **0.1984** | 0.106 |
| organizer's official published baseline | 0.125 | 0.068 | 9.81 | 0.1067 | — |

(The 0.6926 above differs slightly from the 0.7228 recorded on 2026-08-27 —
the ledger's content-free filter changed in between, see
`fix/spine-doc-divergences`. Re-run both conditions together if an exact
delta ever matters.)

**Read this as a bracket, not a replacement number.** Real customers
plausibly do say "I want it in cotton" — legitimate disclosure, not a leak —
but they don't recite marketing copy verbatim. `leaky` is an upper bound
(free credit from the simulator quoting the answer); `controlled` is a lower
bound. The organizer's real held-out evaluator, presumably backed by genuine
customer profiles rather than text extracted from the answer, should land
somewhere between them.

**What this means for the 2026-08-27 entry's central claim:** "Phase 1
already clears the planning repo's projected ask-yield ceiling (0.68–0.75)"
does **not** hold under the controlled condition (0.1984 — nowhere close).
The controlled score does clear the organizer's own official baseline by
roughly 2x, so the accumulate-every-constraint lever has real value — just
far less than the leaky number implied. **Do not carry the 0.7228 (or 0.85
HitRate@10) figure back to `techjam2026-docs`/`standing-findings.md` as a
re-verification of that ceiling without this caveat attached.**

**Open question for Phase 2/3, revised:** the 2026-08-27 entry flagged
Boundary as a puzzling weak point given the fixed schedule should handle it
most cheaply. It's not puzzling under this mechanism — Boundary is the one
scenario that discloses the *least* leaked text (one attribute is always
declined outright), so it was always going to score closer to the true,
unleaked number than the other three scenarios. Worth re-checking whether
ask-yield's dynamic ordering (Phase 2) meaningfully helps under the
*controlled* simulator, not just the leaky one — Phase 2 will otherwise be
evaluated locally against the same leaky harness by default, and could look
like it's working when it's really just getting better at exploiting the
same leak.
