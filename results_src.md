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

| when (UTC) | commit | branch | score | vs starter | vs baseline | hit@10 | mrr | mttc | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-30 17:03 | `70165ff*` | `main` | **0.872057** | +0.179471 | +0.765347 | 0.6600 | 0.251944 | 6.41 | [both] src/ clean-room rebuild: full Layer 1, seams inert [dataset: public_set.jsonl] |
| 2026-08-30 17:02 | `70165ff*` | `main` | **0.497383** | -0.195203 | +0.390673 | 0.6600 | 0.251944 | 6.41 | [both] src/ clean-room rebuild: full Layer 1, seams inert [dataset: public_set.jsonl] |
| 2026-08-30 17:01 | `70165ff*` | `main` | **0.869625** | +0.177039 | +0.762915 | 1.0000 | 0.632083 | 2.0 | [both] smoke [dataset: set.jsonl] |
| 2026-08-30 16:56 | `70165ff*` | `main` | **0.832292** | +0.139706 | +0.725582 | 1.0000 | 0.510972 | 2.05 | harness smoke: 251-product synthetic catalog, 40 sessions [dataset: syn_dataset.jsonl] |
| 2026-08-30 16:55 | `70165ff*` | `main` | **0.872057** | +0.179471 | +0.765347 | 0.9950 | 0.705855 | 2.86 | src/ rebuild, first full public-set run from the verification harness [dataset: public_set.jsonl] |
