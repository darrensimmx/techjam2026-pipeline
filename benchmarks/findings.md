# Findings

Living log, append-only. A dated entry each time a `scripts/benchmark.py`
run reveals something worth remembering — a surprising number, a
hypothesis confirmed or reversed, a tuning lever that mattered or didn't.
Pairs with `history.jsonl`: that file has the numbers, this file has the
story behind them. Never delete an entry; mark it superseded if a later
run contradicts it.

## 2026-08-27 — Phase 1 baseline beats the planning repo's projected ask-yield ceiling

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
