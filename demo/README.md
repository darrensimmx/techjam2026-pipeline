# `demo/` — two CLIs that show the agent working

Dev tooling. **Not part of the submission**, never on the graded path, and it
changes nothing about `src/`.

```text
terminal 2                              terminal 1
python -m demo.backend                  python -m demo.frontend --bracket leaky --step
   the reasoning                           the conversation
```

Start the backend first — it waits for a run to appear. Then start the frontend.
The frontend appends a JSONL trace to `demo/runs/`; the backend tails it and
renders each turn's nineteen pipeline stages as they land.

## The two-terminal recipe

```bash
# terminal 2 — the pipeline explainer, waits for a run
python3 -m demo.backend

# terminal 1 — the conversation, one turn per keypress
python3 -m demo.frontend --bracket leaky --sample-id public_0001 --step
```

Nobody types the customer's replies. The conversation is **scripted replay**:
the evaluator's own simulated customer drives it from `data/public_set.jsonl`,
so what you see is a session the scorer would actually have produced.

Useful variations:

```bash
# a random hard session of a given scenario, reproducibly
python3 -m demo.frontend --bracket scrubbed --scenario intent_override --seed 7

# hands-free, 2s per turn
python3 -m demo.frontend --bracket leaky --delay 2

# re-render a finished run afterwards
python3 -m demo.backend --replay demo/runs/<run>.jsonl --speed 2

# just the query and the picks
python3 -m demo.backend --only C,F
```

`python3` on macOS/Linux, `python` on Windows (`docs/windows-dev-setup.md`).

## `--bracket` is required, and here is why

`public_set.jsonl` carries no `intent_card`, so the vendored evaluator builds
the simulated customer's "hidden" preferences **out of the target product's own
listing** and recites them back turn by turn. 94.5% of disclosed constraint
strings are exact substrings of the target's indexed text. Locally that inflates
everything.

| arm | what the customer discloses |
|---|---|
| `--bracket leaky` | **upper bound.** The organizer's own behaviour: multi-word spans lifted from the target's listing. |
| `--bracket scrubbed` | **lower bound.** Atomic attribute values only — a material word, a colour word, a budget number. |

Every other tool in this repo defaults to `leaky`, which is fine for a labelled
table. It is not fine for a screen someone films, so here the flag has no
default. Four things then keep the label honest:

1. A coloured banner in both headers.
2. A `[leaky]` / `[scrubbed]` tag on every status line, every score line, and
   every summary — enforced by a test.
3. In leaky mode, block E prints `rate 1.000 under a LEAKY card: every disclosed
   string is literally in the pool text` — the leak shown as a *measurement*
   rather than a disclaimer.
4. The banner reads `session_open.hidden_card.source` — **what actually ran** —
   never the CLI flag, so the two cannot disagree.

Nothing is ever appended to `results_src.md`. A one-session demo is not a
measurement.

## What the backend shows

`src/pipeline.py::_run_turn` is nineteen stages, rendered as eight blocks:

| block | stages | what you see |
|---|---|---|
| A | 1-5 | the utterance, which of the eight evaluator templates produced it, the Tier-1 decode (frame, segments, decline), whether Tier 2 ran |
| B | 6-9 | override guard, the verbatim ledger append, slot fills, ask bookkeeping and any retirement |
| C | 10-12 | the query and where it came from, **the literal FTS5 MATCH string**, the BM25 pool with real scores and titles, the fresh/seen split |
| D | 13-15 | the window through hydrate → rerank → gate, whether the reranker is inert, which products the overlap gate moved and by how much |
| E | — | `overlap.measure()` — an instrument `src/overlap.py:138` already ships and nothing calls in production |
| F | 16-17 | the final ten, each with its BM25 score and its **provenance** (`window#0`, `rest#3`, `seen#0`) |
| G | 18-19 | the AskState before the call, which rung of the ladder fired and why, cross-checked against what the policy actually returned |
| H | — | ground truth: the target's pool rank and picks rank, whether the hit counted, per-stage timings |

Two conventions:

- **`[derived]`** — the tracer computed this rather than observing it. Four
  values are derived: the MATCH string, the ask rung, the window/rest split, and
  the overlap report. Two of those carry a cross-check, and a failed cross-check
  prints as a loud `MISMATCH` rather than being quietly dropped.
- **Narrow terminals** (< 90 columns) drop the title column and say so, rather
  than wrapping into mush.

## How it observes without touching `src/`

`_run_turn` keeps every intermediate in a local and returns only a `TurnPlan`,
so there is nothing to read from outside. Rather than add nineteen emit sites to
the file whose docstring reads *"Every step below runs on every turn and is
FORBIDDEN TO RAISE"*, `demo/tracer.py` wraps the stage functions at runtime and
restores them in a `finally`. The submission stays byte-identical and the graded
path pays nothing.

This reaches every stage — including the three that look lossy — because each
stage receives the previous stage's output as an argument:

- the pre-Tier-2 decode **is** `_tier1`'s return;
- the inline `fresh[:50]` split at `:137` is reconstructed and then
  cross-checked against `_hydrate`'s argument and `_assemble`'s;
- the three orderings rebound onto `window` at `:138-140` are each visible from
  two independent points.

**The one rule.** If a recorder raised inside a stage, `run_turn`'s outer except
would swallow it into `_degraded_plan()` and the demo would silently show a
*different* agent than the one being scored. So every wrapper calls the original
first and returns its result unconditionally, with all recording in its own
`try/except` that counts failures instead of propagating them. `_degraded_plan`
is patched as a canary, and `tests/test_demo_trace.py` injects an
always-raising recorder and asserts the recommendations are unchanged.

## Tests

```bash
python3 -m unittest tests.test_demo_tracer_targets   # static, fast, no catalog
python3 -m unittest tests.test_demo_trace            # end-to-end, synthetic catalog
```

Both are registered in `.github/workflows/ci.yml` — a test file that is not
listed there does not run. The two that carry the weight:

- **`TracingDoesNotChangeBehaviour`** — the same sessions run with and without
  the tracer must produce identical recommendations, ask attributes and messages.
- **`DriverMatchesTheRealEvaluator`** — `demo/driver.py` copies the vendored
  drive loop, so it is checked against the real `evaluate()` for identical
  `(hit, first_hit_turn, best_rank)`.

`verify_targets()` checks every patch site's parameter names **in order** —
a reordered `_assemble(window, rest, seen, limit)` would pass an existence check
while silently inverting every provenance label on screen. If it fails, the
frontend refuses to start and prints the diff.

## Troubleshooting

| symptom | cause |
|---|---|
| `TRACER CANNOT ATTACH TO src/pipeline.py` | a stage function changed shape; fix `demo/tracer.py:PATCH_TARGETS` |
| `AGENT IS DEGRADED` | `data/catalog.jsonl` is missing or unreadable — a data problem, not a regression |
| `SMALL CATALOG (N rows)` | you pointed it at `tests/fixtures/catalog.jsonl`; 6 products against `top_k=10` makes any ranker look perfect |
| `POOL EMPTY` twice running | the index has rows but the query matched nothing |
| `RUNG MISMATCH` | `demo/askrung.py` has drifted from `src/askpolicy.py::_select`; trust the policy row |
| `_degraded_plan FIRED` | the trace is **not** the scored agent — file it |
| escape codes printed literally on Windows | use Windows Terminal, or pass `--no-color` |
| backend says `waiting for a run` | start the frontend, or pass `--run <path>` |
