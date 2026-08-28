# The ledger-freeze regression (`3bc061f`)

Measured 2026-08-28 against the 200-session public set with the verified 50k
catalog. Reproduce any row here with:

```bash
python3 .claude/skills/run-sol/bench.py bisect ecacc52 94e8916 3bc061f d567541 main
```

## Summary

`3bc061f` ("Fix ledger content-free filter; widen scheduler signature") dropped
`recommended_technical_score` from **0.722818 to 0.692586** — hit rate 0.85 →
0.80, ten sessions out of 200 flipping from hit to miss.

The commit's stated premise is **correct**. It is not a careless change, and a
straight revert is the wrong response. What it lacked was an end-to-end
measurement, which no test in the suite performs.

## Bisect

| commit | hit@10 | mrr | mttc | score | |
|---|---|---|---|---|---|
| `ecacc52` | 0.8500 | 0.525395 | 3.990 | 0.722818 | benchmark-tracking plan |
| `94e8916` | 0.8500 | 0.525395 | 3.990 | 0.722818 | P1 offline safety — neutral |
| **`3bc061f`** | **0.8000** | 0.525619 | 4.255 | **0.692586** | **−0.030232** |
| `d567541` | 0.8000 | 0.525619 | 4.255 | 0.692586 | acceptance checks — neutral |
| `2ba2747` | 0.8000 | 0.525619 | 4.255 | 0.692586 | current `main` |

`ecacc52` reproduces the recorded `phase1-baseline` score exactly, which
validates both the historical number and the measurement method. Runs are
bit-identical across repeats, so these deltas carry no noise.

## Mechanism

`Agent._respond` builds its query directly from the ledger:

```python
query = state.disclosed_constraints or message_text
matches = self._index.search(query, _limit(top_k))
```

`Bm25Index.search` is deterministic. So when `SessionState.record_message`
filters a reply, `disclosed_constraints` does not change — and the query, and
therefore the entire top-10, are **identical to the previous turn**.

`3bc061f` widened the filter from one content-free reply template to all three.
Once a session exhausts its real disclosures, *every* remaining simulator reply
is content-free, so the ledger freezes and the agent re-issues a dead query
until turn 10. It cannot convert, no matter how many turns remain.

Hits by first-hit turn:

| first-hit turn | before | after |
|---|---|---|
| turns 1–5 | 162 | 160 |
| **turns ≥ 6** | **8** | **0** |
| misses | 30 | **40** |

Every late-turn conversion disappeared.

## The effect is non-additive

Filtering either new template alone is nearly free; filtering both is not:

| filtered templates | score |
|---|---|
| boundary only (pre-`3bc061f`) | 0.722818 |
| boundary + "no additional preference" | 0.721767 |
| boundary + "those options are not quite right" | 0.719719 |
| all three (current `main`) | **0.692586** |
| nothing at all | 0.721519 |

Roughly ten times the sum of its parts. The old, leaky filter was accidentally
acting as **query diversification** — the noise tokens perturbed the query each
turn, which re-rolled the BM25 ranking and occasionally surfaced the target.

## What to do about it

Do not revert. The three replies really do disclose nothing — verified against
the simulator's own templates at `evaluator/local_evaluator.py:166-184` — so
re-admitting them only restores a noise-driven lottery that happens to pay out.

The commit did not introduce the defect; it **exposed** one that was always
there: the agent has no strategy once its six-attribute schedule is exhausted.
Before, random noise disguised that. The fix belongs in Phase 2 (ask-yield):
treat "the ledger stopped growing" as the trigger for a deliberate exploration
or diversification step.

Note that the boundary and intent_override scenarios are byte-identical across
the regression; only buying and browsing moved.

## Process consequence

`tests/test_ledger_scheduler.py` passes in **both** states. Unit tests
structurally cannot observe a ranking-quality change, so any future edit to the
ledger filter, the schedule, or the query construction must be measured
end-to-end before merge:

```bash
python3 .claude/skills/run-sol/bench.py check
```
