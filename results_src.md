# `src/` benchmark results

Appended by `scripts/evaluate_src.py` on every run. Newest run first. `score` is
`recommended_technical_score` over whatever `--dataset` was passed:
`0.50*hit@10 + 0.30*mrr + 0.20*efficiency`.

Reference points -- organizer weak-BM25 baseline `0.106710`; superseded
`starter/` system @ `70165ff` `0.692586` (see `results.md`).

A `*` after the commit means tracked files were modified in the worktree, so
that row does not correspond to the commit alone. `degraded` means
`Agent.degraded` was True: the index did not build, and the score is a data
problem rather than a solution problem.

The `[...]` prefix in a note names the bracket. `[both -> leaky]` means both arms
ran and the row records the leaky one. **Every row's `score` must be derivable
from its own columns** as `0.50*hit@10 + 0.30*mrr + 0.20*((11 - mttc)/10)`; a row
where it is not, is not a measurement. `vs starter` and `vs baseline` are `--` on
a scrubbed row, because both reference constants were measured under the leak.

> **Two rows below are marked `BRACKET-MIXED` and must not be quoted.** Until
> 31 Aug 2026, `scripts/evaluate_src.py` rebound `result` inside the bracket loop,
> so a `--bracket both` run paired the **leaky score** with the **scrubbed
> metrics**. Both marked rows read `0.872057` beside metrics that compute to
> `0.497383`. They are kept rather than deleted because a deleted row is not a
> corrected one — the numbers themselves were fine, only their pairing was wrong,
> and both true values appear on correctly-paired rows elsewhere in this table.
> Fixed at `scripts/evaluate_src.py` `main()`; rows from 02:21 on are sound.

| when (UTC) | commit | branch | score | vs starter | vs baseline | hit@10 | mrr | mttc | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-31 21:02 | `cb5817e*` | `enable-layer3-live` | **0.913500** | -- | -- | 1.0000 | 0.925000 | 4.2 | [leaky] live gemini smoke test, 10 sessions (4 intent_override/3 boundary/2 browsing/1 buying) [dataset: llm_smoke_set.jsonl] [10 sessions -- NOT the 200-session set] |
| 2026-08-31 02:21 | `6e4c32b*` | `src-rebuild` | **0.872057** | +0.179471 | +0.765347 | 0.9950 | 0.705855 | 2.86 | [both -> leaky] verify bracket-mixing fix [dataset: public_set.jsonl] |
| 2026-08-31 02:11 | `6e4c32b*` | `src-rebuild` | **0.872057** | +0.179471 | +0.765347 | 0.9950 | 0.705855 | 2.86 | [leaky] PR verification: leaky arm alone, consistent row [dataset: public_set.jsonl] |
| 2026-08-31 02:10 | `6e4c32b` | `src-rebuild` | **0.872057** | +0.179471 | +0.765347 | 0.6600 | 0.251944 | 6.41 | [both] PR verification run against docs/artifacts [dataset: public_set.jsonl] **BRACKET-MIXED -- do not quote; see header. Score is leaky, metrics are scrubbed.** |
| 2026-08-30 17:03 | `70165ff*` | `main` | **0.872057** | +0.179471 | +0.765347 | 0.6600 | 0.251944 | 6.41 | [both] src/ clean-room rebuild: full Layer 1, seams inert [dataset: public_set.jsonl] **BRACKET-MIXED -- do not quote; see header. Score is leaky, metrics are scrubbed.** |
| 2026-08-30 17:02 | `70165ff*` | `main` | **0.497383** | -0.195203 | +0.390673 | 0.6600 | 0.251944 | 6.41 | [both] src/ clean-room rebuild: full Layer 1, seams inert [dataset: public_set.jsonl] |
| 2026-08-30 17:01 | `70165ff*` | `main` | **0.869625** | +0.177039 | +0.762915 | 1.0000 | 0.632083 | 2.0 | [both] smoke [dataset: set.jsonl] |
| 2026-08-30 16:56 | `70165ff*` | `main` | **0.832292** | +0.139706 | +0.725582 | 1.0000 | 0.510972 | 2.05 | harness smoke: 251-product synthetic catalog, 40 sessions [dataset: syn_dataset.jsonl] |
| 2026-08-30 16:55 | `70165ff*` | `main` | **0.872057** | +0.179471 | +0.765347 | 0.9950 | 0.705855 | 2.86 | src/ rebuild, first full public-set run from the verification harness [dataset: public_set.jsonl] |
