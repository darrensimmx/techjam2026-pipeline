# Part 0 — the decision rule, written before the arms ran

Committed deliberately ahead of the Part 3 / 4 / 5 results so the thresholds
cannot be fitted to them afterwards. Part 1 (the rerank ceiling) and Part 2 (the
dense ceiling) are gates on *whether to build* an arm, not ship decisions, so
running them first does not compromise this.

The evidence for that ordering is the filesystem: this file is written while
`results-part3.json`, `results-part4.json` and `results-part5.json` do not yet
exist, and its mtime predates all three. Committing it as its own commit, ahead
of the results, would make that guarantee stronger and is worth doing before
this branch is pushed.

---

## 1. What `submission_rules.md` actually constrains

The prompt asks us to quote the size limit and the offline requirement, and to
say at once if bundled model weights are barred or would push us over — because
that might settle the whole question before an arm runs.

**There is no numeric size limit.** Read at source, the entire size constraint is
one adjective:

> *"You may include: … lightweight local assets required by your agent"* — `:36-41`

and one reservation:

> *"The organizer reserves the right to run your submission under CPU, memory,
> timeout, and network restrictions."* — `:100-101`

So bundled weights are **not barred**, and no stated megabyte figure can rule
them in or out. This does *not* settle the question — but it changes its shape:
the binding constraint is CPU/timeout, not disk. A 91 MB cross-encoder is not
disqualified by any published rule; ~1 s of CPU per turn against an unpublished
per-turn timeout is the actual risk, and it is a judgement call, not a lookup.

**Offline is a real requirement, and it is asymmetric.**

> *"For official final scoring, organizer policy may disable network access."* — `:59`

Combined with `local_evaluator.py:239-242`, where `respond()` is wrapped in a
bare `except Exception` that swallows the failure into an empty response, a
network-dependent arm does not error under offline scoring — it silently scores
zero for that turn. That asymmetry is why the LLM rerank arm (K2) is refused
rather than measured: the number it produces with networking on is not the
number it would score.

## 2. There is no seed to vary

The prompt asks for BM25-only under 5 seeds, to size every margin against the
baseline's own seed-to-seed spread.

**That variance does not exist here.** `local_evaluator.py` exposes no seed
argument, and its only RNG (`materialize_hidden_fields`, `:210-212`) is seeded
from `f"{sample_id}\0{scenario_type}"` — content-derived, not global. The
shipped agent is deterministic. Measured, three runs under different
`PYTHONHASHSEED`:

| PYTHONHASHSEED | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| 0 | 0.8 | 0.525619 | 4.255 | 0.692586 |
| 1 | 0.8 | 0.525619 | 4.255 | 0.692586 |
| 12345 | 0.8 | 0.525619 | 4.255 | 0.692586 |

Seed variance is exactly **0.000000**. A rule reading "beat the seed-to-seed
variance" would therefore accept *any* positive difference, including one
session moving from rank 3 to rank 2. That is the opposite of what the rule was
for, so it is replaced rather than satisfied.

**The substitute** is the sampling variance that does exist: the 200 public
sessions are a sample, and the private set is a different draw. A bootstrap over
sessions (2000 draws) gives the baseline's spread:

| ledger | R1 TechnicalScore | bootstrap sd | 95% interval |
|---|---|---|---|
| current (HEAD) | 0.692586 | 0.025785 | [0.642451, 0.743510] |
| legacy (94e8916) | 0.722818 | 0.022729 | [0.677544, 0.766168] |

So **one baseline sd is ≈ 0.023–0.026 TechnicalScore.** Every margin below is
read against that.

## 3. The rule

An arm ships only if **all four** hold. Any single failure is a "don't ship".

**(a) The gain is real.** The *paired* bootstrap 95% CI on ΔTechnicalScore
against R1 excludes zero. Paired, not two independent CIs: the same resampled
session indices are scored under both arms so session-difficulty variance
cancels. Two overlapping marginal CIs are not evidence of no difference, and
this project has already been burned once by reading a difference off two
separately-quoted aggregates.

**(b) The gain is big enough to be worth carrying.** The point estimate exceeds
**+0.020**, i.e. roughly one baseline bootstrap sd. Below that, a difference is
inside the noise the private set will resample anyway, and this repo's own
Bucket 2 rule already says a difference of that size is not evidence of a better
choice. An arm that clears (a) but not (b) is recorded as *measured, real, and
too small to justify the complexity* — a distinct finding from "no effect".

**(c) It does not regress many sessions.** At most **5% of sessions** (10 of
200) end at a worse reciprocal rank than under R1. If an arm exceeds this, it
ships only when the aggregate gain exceeds **3 baseline sd (+0.070)** — the
threshold at which a broad win is large enough to buy out a concentrated loss.
Stated now so it cannot be introduced later as an excuse. Win/loss/tie is
reported for every arm regardless.

**(d) It is offline and affordable.** Weights are bundled and local; no network
call sits on the response path. Added wall-clock is reported per turn, and an
arm costing **more than 1.0 s per turn** on this rig is *not* shippable on the
strength of its score alone — it is recorded as **conditional**, blocked on the
organizers publishing a per-turn timeout. It cannot be talked into a ship
decision by its TechnicalScore, because a timeout does not reduce the score
gracefully: it zeroes the turn.

## 4. Committed in advance

- If Part 2's union recall is not meaningfully above BM25's own recall, **Part 3
  is reported as unnecessary and the dense question is closed**, whatever a
  tuned fusion weight later shows. A fusion arm cannot retrieve what neither
  input retrieved; a near-tie produced by tuning is not a reason to reopen it.
- If R4's weight sweep peaks at or adjacent to `w(dense) = 0`, **that is the
  finding and it gets reported as such** — not as "best weight 0.1, +0.002".
- If Part 1's oracle gap is small, Part 4 is skipped. *(It was not: measured
  +0.2105 at top-50. Part 4 runs.)*
- K2, the LLM rerank, is **not built**, for the reason in §1. That refusal is
  the finding, and it is reported rather than left as a gap.
- A negative result is the expected outcome and is written up as plainly as a
  positive one. "BM25 wins, the rest is not worth the complexity or the offline
  risk" is a complete answer to this bake-off.

## 5. Two baselines, not one

`3bc061f` ("Fix the ledger content-free filter") is described as a bug fix and
per-turn it is one — but it costs **−0.030232** TechnicalScore against the
commit before it, which is larger than every effect this bake-off is measuring.
The bake-off's own fixed conditions pin *"every disclosed string concatenated
into the query, every turn"*, which is the pre-`3bc061f` behaviour.

So every arm is measured against **both** ledgers and both are reported. An arm
that only clears the rule under one of them has not cleared it.
