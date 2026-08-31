# techjam2026-pipeline

**Statement 4 — conversational e-commerce search.** A multi-turn shopping agent
that asks one useful clarifying question per turn and finds the customer's
hidden target product within ten turns.

---

# Submission

Everything the organizer needs is in this section. The phase plan that follows
it is internal team history and is **superseded** — the source of truth for the
current design is `docs/artifacts`.

## What ships

```text
agent.py            the submission entry point -- exports `Agent`
src/                the system (18 files, standard library only)
requirements.txt    intentionally empty
docs/todo.md        decisions deliberately left open, with their evidence
```

`starter/` is the superseded first-generation system. It is retained unmodified
as the historical record and as the baseline the rebuild is measured against.
**It is not part of the submission.** `evaluator/` is the organizer's kit,
vendored byte-identical and never edited.

## Setup and reproduction

```bash
# Python 3.10 or later. Verified on 3.14.2 / SQLite 3.50.4 (FTS5 required).
pip install -r requirements.txt        # a no-op: there are no dependencies

# place the organizer's catalog at data/catalog.jsonl, then:
python3 -m unittest discover -s tests -p 'test_*.py' -t .   # 390 tests
python3 scripts/evaluate_src.py --bracket both              # score it
```

One command to run the agent in the official harness:

```python
from agent import Agent          # Agent(catalog_path="data/catalog.jsonl")
```

There are **no environment variables** and **no non-obvious setup**. All paths
resolve relative to the repository root.

## Model choice, cost, tokens, latency, network

| Disclosure | Value |
|---|---|
| **Models used** | **None.** No LLM, no SLM, no neural model of any kind runs on the graded path. |
| **Network access** | **None required, and none attempted.** Enforced by an AST test over every `src/` module (`tests/test_src_no_network.py`). |
| **API credentials** | None. The system cannot fail for want of a key. |
| **Offline verified** | Full 200-session run under `sandbox-exec -f scripts/no-network.sb` with networking revoked: **identical score, 0.872057**. A control probe in the same sandbox confirms the block is real, not vacuous. |
| **Reported token usage** | `0` prompt, `0` completion, every turn — truthfully, because no model is called. |
| **Estimated model cost** | **$0.00.** |
| **Latency** | Index build **1.16 s** once at construction; **~19 ms per turn** end to end. 200 sessions / 571 turns complete in **9.7 s**. |
| **Memory** | One in-memory SQLite FTS5 index over the 50,000-product catalog. |

Retrieval is BM25 over SQLite FTS5 — standard library only. Dense/embedding
fusion was measured twice and **rejected** (−0.206 at top-100, −0.065 at
top-50): the target is BM25's rank 1 in 87 of 176 hit sessions but sits around
dense rank 72, so blending dilutes a strong list with a weak one.

Three optional layers exist as **typed seams with inert null implementations** —
a cross-encoder reranker, a semantic intent fallback, and an LLM re-ranking
escalation. All three are **disabled**, each behind a flag checked *before* its
dependency, so installing `requirements-optional.txt` changes nothing. Which
implementation fills each seam is an open decision, recorded with its evidence
in `docs/todo.md`. **If any is ever enabled, this table must be updated** — the
LLM seam is the only place a language model is even proposed, and it sits in
*ranking*, never in intent parsing.

## Method

`respond()` is a never-raise wrapper around one pass:

**decode → contradiction check → ledger append → query → BM25 → rerank seam →
overlap gate → never-repeat selection → ask policy → schema coercion**

- **Intent (Tier 1).** The simulator emits a closed set of sentence shapes, so
  intent is an anchored-regex *frame decode*, not an estimate. It splits the two
  declines on the single token `additional`: "I don't have **a** preference"
  means the bucket was never opened (re-ask later), while "I don't have an
  **additional** preference" proves it empty (retire permanently).
- **The ledger is the query.** Every disclosed reply is appended verbatim, and
  the concatenation of those raw strings *is* what gets searched. Append-only,
  enforced by the absence of any deletion method.
- **Slots are scheduling-only.** The typed slot view never touches retrieval, so
  a parsing bug can corrupt *which question we ask* but can never corrupt
  *what we search*. Asserted structurally in `tests/test_src_layering.py`.
- **Ask policy.** A seven-attribute fixed schedule for turns 1–7, then a
  fallthrough ladder re-evaluated independently on each free turn. Never `null`,
  never `other`.
- **Never repeat a shown product.** The session ends the instant our list
  contains the target, so anything still on screen in a running session is
  confirmed wrong. Excluding it cannot lose the target. It is an *ordering*
  preference, not a filter — the top-10 is always full.

## Results

Scored with the organizer's own unmodified `evaluate()` over the 200 public
sessions.

| bracket | hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| leaky (upper bound) | 0.9950 | 0.7059 | 2.860 | **0.872057** |
| scrubbed (lower bound) | 0.6600 | 0.2519 | 6.410 | **0.497383** |

**Read both, never one.** `public_set.jsonl` carries no real intent card, so the
evaluator falls back to building the "hidden" customer preferences out of the
*target product's own listing* — 94.5% of disclosed constraint strings are
verbatim substrings of the target's indexed text. Leaky is as-shipped; scrubbed
patches that leak. The organizer's held-out set should land between them.

Against the superseded `starter/` system, measured under the same conditions:
**+0.179 leaky, +0.299 scrubbed.** The gain is *larger* with the leak removed,
which is the opposite of a measurement artifact.

## Limitations

- Both numbers above are this simulator. The bracket *direction* is the signal.
- The never-repeat rule shipped ahead of a rank-distribution readout the design
  asked for first (`docs/todo.md` item 7). It cannot cost anything — that much
  is structural — but it is not yet proven to be what earns the gain.
- Tier 1's frame decode is exact against the current simulator. The organizers
  reserve the right to add paraphrasing; the semantic fallback for that case is
  designed and seamed but **not built**, so paraphrased input degrades to a
  content-bearing unknown rather than being understood.
- One reply shape decodes conservatively: a message combining real content with
  a decline phrase is treated as a decline, losing its content. Deliberate — on
  a paraphrased set a missed decline is the costlier error.

---

# Internal phase plan (superseded)

Everything below predates the `src/` rebuild and is kept for team history. The
current design lives in `docs/artifacts`; open decisions live in `docs/todo.md`.


Code for TikTok TechJam 2026, Statement 4: a conversational shopping agent that
asks useful follow-up questions and finds the customer's hidden target product
within 10 turns.

Planning and reasoning live in a separate repo —
[`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs).
Read that first if you want the *why*. This repo is the *how*: the actual
`Agent` implementation, the dev tooling around it, and this phase plan.

Competition kit (organizer-owned, read-only reference — contract, evaluator,
data): [`TechJam2026/techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search).

## What we're building

The organizer's evaluator imports one class and calls it in-process — no
network, no CLI, on the graded path:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",   # one of a fixed enum, or null
            "recommendations": [{"parent_asin": "B000...", "score": 1.0}],  # up to 10, best-to-worst
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

A session ends the moment the target appears in `recommendations`, or after
turn 10. Full schema: `docs/agent_api_contract.json` in the competition repo.

That `Agent` — pure local Python, zero network calls, never throws — is the
entire submission. Everything else in this repo (the two chat CLIs, tests,
tooling) exists to build and sanity-check it; none of it ships to the
organizer.

## Overall plan

Ship a minimal, safe, *scoring* pipeline first, then layer in the two
adaptive features that are documented as actually moving the number, in the
order the evidence supports — not the order that sounds impressive. Concretely:

1. **Get something that scores at all**, safely, before making it smarter.
   The single biggest lever the docs repo identifies isn't a clever model —
   it's accumulating every disclosed constraint into the retrieval query,
   unconditionally, every turn (0.16 → 0.75 in prior review). That alone is
   worth more than any model choice in this project, so it comes first.
2. **Verify the retrieval assumption before building on top of it further** —
   the BM25-only decision is prior-evidence-backed; the bake-off that checks
   it was the single highest-priority open item in the planning repo. It
   gates 80% of TechnicalScore (HitRate 50% + MRR 30%), and the one confirmed
   effect it produced — cross-encoder rerank at +0.047, CI excluding zero — is
   an order of magnitude larger than the whole attribute-selection band
   (≈0.004). Everything else is measured on top of whatever this settles.
3. **Fill the turns a fixed schedule can't reach** (ask-yield dynamic
   ordering) — the six-attribute schedule runs dry after turn 6. Sequenced
   after retrieval on purpose: ask-yield's yield signal is defined partly as
   *churn in the ranked candidate pool*, so it reads retrieval's output and
   would need re-tuning if the spine changed underneath it. The dependency
   runs one way.
4. **Add contradiction detection last** — its ceiling on the metric that's 80%
   of TechnicalScore is ≤0.0078 by the planning repo's own analysis. It still
   ships (it's in the rubric, it's the innovation play), it just isn't where
   early effort goes. **Scope narrowed 29 Aug 2026:** what ships here is the
   *slot-value* contradiction check — exact string comparison, hard trigger,
   zero model. The per-turn intent labeller and its drift check are 🔴 demoted
   (`techjam2026-docs/features/intent-trajectory/`): the reply-frame decode
   already supplies the per-turn signal for free, and `customer_reply` is a
   deterministic function of session state, so the drift cannot occur.
   **"Intent classifier" now names something else in the docs repo** — the
   shipping regex frame decode plus its semantic fallback,
   `techjam2026-docs/features/intent-classifier/`. See that repo's
   `project/glossary.md` before using either term.
5. **Harden for submission** — prove the offline fallback actually works
   with networking *disabled*, not mocked, and package per the organizer's
   reproducibility rules.

Each phase gets its own branch, opened off `main` once the prior phase is
merged.

## Phases

### Phase 0 — Environment & baseline
**Goal:** prove the environment works before writing any agent logic.
**Deliverables:** competition kit's `evaluator/` vendored in unmodified,
`data/catalog.jsonl` downloaded, unmodified starter agent reproduces the
published baseline (HitRate@10 ≈ 0.125) via `python3 -m evaluator.local_evaluator`.
**Status:** not started.

### Phase 1 — Foundational pipeline
**Goal:** a real, scoring `Agent` — constraint ledger, fixed six-attribute
schedule, BM25 over the accumulated ledger string, boundary handling, always
returns top-10 — plus two CLIs so a human can manually chat with it turn by
turn. This is the minimum viable version of the whole system.
**Deliverables:** `starter/agent.py` + `ledger.py` + `scheduler.py` +
`retrieval.py`; `cli/agent_server.py` + `cli/client.py`; evaluator re-run
showing a jump from baseline toward the ~0.17–0.23 stateful-BM25 range.
**Branch:** `foundational-pipeline`.
**Status:** in progress — see that branch.

### Phase 2 — Retrieval verification & optional rerank
*(Renumbered 28 Aug 2026 — was Phase 3. See "Phase numbering" below.)*
**Goal:** re-run the stalled retrieval bake-off to confirm (or correct) the
BM25-only decision; add local cross-encoder rerank only if the numbers
justify it, and only as an optional layer that's never on the critical path.
**Status:** verification done, rerank decision open. The bake-off has results
(dense loses, rerank wins) — planning repo issue #7. The remaining call is
whether to ship cross-encoder rerank at +0.047 for ~1.2s/turn, which is
blocked on an unpublished organizer timeout — planning repo issue #9.

### Phase 3 — Ask-yield adaptive question ordering
*(Renumbered 28 Aug 2026 — was Phase 2. See "Phase numbering" below.)*
**Goal:** replace the fixed six-attribute order with one that adapts to what's
actually taught the most so far, filling turns 7–10 where the fixed schedule
has no rule left.
**Deliverables:** ask-yield layer swapped in behind the same scheduler
interface used in Phase 1, with a try/except and fallback to the fixed order
on failure.
**Status:** not started — and the premise is under review. Planning repo
issue #4 challenges "the schedule runs dry after turn 6"; measured against
the vendored evaluator, only 40/200 public sessions reach turn 7 at all. Do
not start this before that investigation reports.

### Phase 4 — Intent classifier & contradiction detection
**Goal:** classify each turn Buying/Browsing/Override and detect when a
customer contradicts an earlier disclosed constraint, via a ledger-based
slot-value diff.
**Deliverables:** heuristic phase, then trained encoder classifier (training
needs a real machine/Colab — blocked in sandboxed environments with no
network to pytorch.org).
**Status:** not started.

### Phase 5 — Hardening & submission
**Goal:** meet the organizer's actual bar, not just "it runs."
**Deliverables:** run the full evaluator with networking *actually disabled*
and confirm a non-zero score; write the method/limitations report and the
latency/token/cost disclosure; package per `docs/submission_rules.md`
(entry file, requirements, one-command run instructions).
**Status:** not started.

## Phase numbering

**This repo's `Phase 0`–`Phase 5` is the single canonical execution plan.** Cite
it as "Phase N" in issues, PRs, branch names, and benchmark labels.

Two other numbering schemes exist in the planning repo and are **not**
interchangeable with this one — the same token means different things:

| This repo | Planning repo `G1–G5` (architecture gates) | Planning repo `1–9` (superseded) |
|---|---|---|
| Phase 0 — Environment & baseline | — | 1 (partial) |
| Phase 1 — Foundational pipeline | G1 offline safety · G2 ledger + ask · G3 fixed schedule | 1, 4 |
| Phase 2 — Retrieval verification & rerank | — | 2, 3, 9 |
| Phase 3 — Ask-yield | G4 ask-yield | — |
| Phase 4 — Intent classifier & contradiction | G5 intent override | 5, 6, 7, 8 |
| Phase 5 — Hardening & submission | — | — |

The planning repo's key was relabelled `P1–P5` → `G1–G5` on 28 Aug 2026 to end a
direct collision: its `P2` meant "ledger + ask" (shipped here in Phase 1) while
this repo's `Phase 2` meant something entirely different. `G1–G5` describes
*architecture gates* (what must be green before what) and deliberately has no
slot for environment, retrieval, or hardening; it is not a delivery schedule.

**Phases 2 and 3 were swapped on 28 Aug 2026** (retrieval was Phase 3, ask-yield
was Phase 2). Retrieval gates 80% of TechnicalScore and ask-yield's yield signal
reads retrieval's candidate-pool churn, so the dependency runs retrieval →
ask-yield. Pre-28-Aug references to "Phase 2 / ask-yield" and "Phase 3 /
retrieval" use the old numbering; the work item named alongside the number is
authoritative where they disagree.

## Repo layout (target, mirrors the competition kit)

```
techjam2026-pipeline/
  starter/
    agent.py          # the graded Agent class
    ledger.py           # disclosed_constraints accumulation
    scheduler.py         # fixed six-attribute order (Phase 1) -> ask-yield (Phase 3)
    retrieval.py         # BM25 over the full ledger string
  evaluator/             # vendored from the competition repo, never edited
  data/
    catalog.jsonl         # downloaded from the competition repo's release, gitignored
    public_set.jsonl       # 200 labeled dev sessions
  cli/
    agent_server.py       # dev-only: hosts one Agent, speaks newline-delimited JSON on stdio
    client.py            # dev-only: REPL for manual turn-by-turn chat
  tests/
  README.md
  requirements.txt
```

## Cross-repo sync checks

`.github/workflows/docs-sync.yml` runs `scripts/check_docs_sync.py` on every
push to `main`, weekly, and on demand. It fetches `darrensimmx/techjam2026-docs`
(private) and checks a small, curated table of facts against this repo's
actual code — not general prose parsing, a regression net for specific claims
that have already drifted once (see the script's `FACTS` table and the docstring
on each check for the incident it guards). `techjam2026-docs` runs the mirror
check, `scripts/check_pipeline_sync.py`, against this repo.

**One-time setup (not done by CI itself):** both repos are private, so each
workflow needs read access to the other. Create one fine-grained PAT scoped to
read-only **Contents** on both `techjam2026-pipeline` and `techjam2026-docs`,
then add it as a repository secret named `TECHJAM_CROSS_REPO_TOKEN` in *each*
repo's Settings → Secrets and variables → Actions. Without it the workflow
still runs and reports which checks it couldn't fetch, rather than failing
silently.

Extend `FACTS` (or the docs repo's mirror table) whenever a review — human or
Claude — catches a claim in one repo that doesn't match the other's actual
state; that's what turns a one-off correction into a standing guard.

## Related

- Planning source of truth: [`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs)
  — see `pipeline/implementation-plan.md` there for the detailed DoD checklist
  behind Phase 1.
- [`docs/benchmark-tracking-plan.md`](docs/benchmark-tracking-plan.md) — how
  Hit Rate/MRR/MTTC get tracked across phases so each change's effect is
  visible in git history, not just in someone's terminal.
- Competition kit: [`TechJam2026/techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search)
