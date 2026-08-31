# Retrieval bake-off harness

Executes `features/retrieval-rerank/bakeoff-prompt.md` from the planning repo
([`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs)).
That prompt had been run once before, stalled after 9 commands, and produced no
output files — this directory is the re-run that produces them.

**Findings live in the planning repo. Code lives here.** Nothing in this folder
is part of the submission: `starter/` is the shipped agent and is untouched by
everything here, and the vendored `evaluator/` is imported, never modified
(`submission_rules.md:51` bars *"code that modifies evaluator files"*).

## The one idea that makes this affordable

In `evaluator/local_evaluator.py:238-268` the next customer message is a
function of exactly three things — `response["ask_attribute"]`, the `disclosed`
set, and `boundary_used`. It is **not** a function of
`response["recommendations"]`. Recommendations only control *when the session
stops* (`:252-255`).

So the query trajectory is independent of retrieval quality. Run the shipped ask
policy once with an agent that returns no recommendations, capture all 200×10
queries, and every retrieval or rerank arm becomes a pure re-ranking of cached
candidates — no evaluator re-run per arm. That is what turns an 11-point weight
sweep from a day into a second.

This is **proved, not assumed**. `simulate.py` replays the BM25 baseline and must
reproduce a real `evaluate()` run to 6 decimals and on all 200 individual
sessions. It does, on both ledger variants:

| ledger | replayed | real `evaluate()` | per-session mismatches |
|---|---|---|---|
| current (HEAD) | 0.692586 | 0.692586 | 0 / 200 |
| legacy (94e8916) | 0.722818 | 0.722818 | 0 / 200 |

If that check ever fails, every number downstream of it is void.

## Why two ledgers

`3bc061f` widened the ledger's content-free filter. It is described as a bug
fix, and per-turn it is one — but it costs **−0.030232** TechnicalScore
(HitRate@10 0.85 → 0.80). The bake-off's own fixed conditions pin *"every
disclosed string concatenated into the query, every turn"*, which is the legacy
behaviour, and the gap between the two ledgers is larger than most of the
effects under test. So both are captured and every ceiling is reported against
both. See the planning repo's `project/standing-findings.md`.

## Files

| file | what it does |
|---|---|
| `capture.py` | replays the ask policy, writes `trajectories-{current,legacy}.json` (queries + BM25 top-100 per turn) |
| `simulate.py` | evaluator-exact replay scorer, paired bootstrap, and the `validate()` proof above |
| `overlap.py` | verbatim-overlap rate per session — the number the retrieval one-pager rests on |
| `bm25_scores.py` | BM25 scores (not just order) for the weighted-fusion arm |
| `dense.py` | encodes the 50k catalog + every captured query with two bi-encoders |
| `esci.py` | pulls real human Amazon queries that target products in our own catalog |
| `part1_ceiling.py` | Part 1 — the rerank ceiling (oracle rerank) |
| `part2_dense.py` | Part 2 — the dense ceiling (union recall, dense-only rescues) |
| `part3_fusion.py` | Part 3 — R1–R4 retrieval arms, including the weight sweep |
| `part4_rerank.py` | Part 4 — K0/K1 cross-encoder rerank, cost and wall-clock |
| `part4_checkpoint_comparison.py` | Part 4 follow-up — *which* cross-encoder. Six checkpoints over `data/TechJam_32_Sessions.jsonl`, reranking BM25's top-50. See the caveats below. |
| `part5_realqueries.py` | Part 5 — the same arms on ESCI's human-authored queries |

### `part4_checkpoint_comparison.py` — read this before citing it

This is the harness behind the checkpoint table now quoted in `src/rerank.py`,
`requirements-optional.txt` and `docs/todo.md` item 4. Cherry-picked onto this
branch from PR #21 (`feature/add-zeroentropy-zerank-1-small`) together with its
input, `data/TechJam_32_Sessions.jsonl`; PR #21 itself is not merged and the two
scratch `.md` files it also carries were deliberately left behind.

Three caveats, none of which invalidate the table but all of which change how
much it can be leaned on:

- **The run is reproducible; the run is not archived.** No
  `results-checkpoint-comparison.json` was ever committed, so the numbers in the
  docs have a harness but no artifact. Re-running it regenerates them — that is
  the difference between this and the centroid comparison, which has no harness
  at all.
- **`MODELS` holds six arms; the quoted table has four.** `distilroberta` and
  `zerank-1-small` (the arm PR #21 exists to add) are not in it. Do not present
  the table as the full sweep.
- **32 sessions, one seed, no confidence interval** — unlike `part4_rerank.py`,
  which bootstraps. It settles *which* checkpoint, not *how much* the rerank is
  worth; the latter is still `part4_rerank.py`'s +0.047 and its own open
  reconciliation debt (`docs/todo.md` item 4).

## Reproducing

`data/catalog.jsonl` is required (see `data/README.md`). The bake-off's extra
dependencies are deliberately **not** in the repo's `requirements.txt` — the
shipped agent is standard-library only and stays that way:

```sh
python -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/Scripts/python -m pip install sentence-transformers pandas pyarrow

python bakeoff/capture.py --ledger current
python bakeoff/capture.py --ledger legacy
python bakeoff/simulate.py            # must print VALIDATION PASSED
python bakeoff/overlap.py
python bakeoff/bm25_scores.py
.venv/Scripts/python bakeoff/dense.py
.venv/Scripts/python bakeoff/esci.py
python bakeoff/part1_ceiling.py
.venv/Scripts/python bakeoff/part2_dense.py
.venv/Scripts/python bakeoff/part3_fusion.py
.venv/Scripts/python bakeoff/part4_rerank.py
.venv/Scripts/python bakeoff/part4_checkpoint_comparison.py   # ~40 min, 6 arms, CPU
.venv/Scripts/python bakeoff/part5_realqueries.py
```

`cache/` and `trajectories-*.json` are regenerable and gitignored; the
`results-*.json` files are the artifacts worth keeping.

## External data

Real human shopping queries come from the Shopping Queries Dataset (ESCI):

> Reddy, Màrquez, Valero, Rao, Zhang, Sanz, Nag, Nagaraj, Karim, Rowe, Nio,
> Zhu. *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
> Product Search.* arXiv:2206.06588, 2022.
> <https://github.com/amazon-science/esci-data> — Apache-2.0.

Used because the public set's customer utterances are built by copying strings
out of the target product's own listing (`local_evaluator.py:52-71`), so a BM25
win on it is consistent both with "BM25 is the right retriever" and with "the
simulator hands BM25 the answer key", and the public set cannot separate those.
Inventing paraphrases to fill the gap would mean inventing the distribution that
decides the answer. ESCI supplies human-authored queries instead.

**Correction, 29 Aug 2026 — the original wording here called ESCI a
"low-verbatim-overlap retrieval set". Measured, it is not.** On the
query-token-coverage measure ESCI is *higher* than the public set (mean 0.815 /
median 1.000 vs 0.724 / 0.714), because that measure is confounded by query
length: a three-token human query has all its tokens somewhere in a long
listing, while a ten-turn accumulated ledger dilutes. What actually differs is
*phrase-level* copying — 94.5% of the simulator's constraint strings are
verbatim substrings of the target listing, and no human query is a copied
phrase. The claim ESCI supports is the narrower one: queries nobody here
authored, not generated from the target document. Kept rather than quietly
reworded, because the metric was added expecting it to support the story and it
did not.

The plan to run ESCI queries against *our own* index was abandoned: 71 shared
ASINs of 599,151 × 50,000, and zero after the us+Exact filter. `esci.py` builds
a standalone 20,000-product corpus in our catalog's schema instead.
