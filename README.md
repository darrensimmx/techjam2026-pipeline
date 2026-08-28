# techjam2026-pipeline

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
turn 10. The evaluator—not the agent—controls this stop, and the agent cannot
see which recommendation is the hidden target. Withholding the current top 10
to chase a higher rank on a later turn would therefore sacrifice an immediate
hit and worsen MTTC, with no guarantee of improving MRR. Full schema:
`docs/agent_api_contract.json` in the competition repo.

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
2. **Fill the turns a fixed schedule can't reach** (ask-yield dynamic
   ordering) — the six-attribute schedule runs dry after turn 6.
3. **Verify the retrieval assumption before building on top of it further** —
   the BM25-only decision is prior-evidence-backed but has never been
   re-verified end to end; that bake-off is flagged as the single
   highest-priority open item in the planning repo.
4. **Add the intent classifier / contradiction detection last** — its ceiling
   on the metric that's 80% of TechnicalScore is ≤0.0078 by the planning
   repo's own analysis. It still ships (it's in the rubric, it's the
   innovation play), it just isn't where early effort goes.
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

### Phase 2 — Ask-yield adaptive question ordering
**Goal:** replace the fixed six-attribute order with one that adapts to what's
actually taught the most so far, filling turns 7–10 where the fixed schedule
has no rule left.
**Deliverables:** ask-yield layer swapped in behind the same scheduler
interface used in Phase 1, with a try/except and fallback to the fixed order
on failure.
**Status:** not started.

### Phase 3 — Retrieval verification & optional rerank
**Goal:** re-run the stalled retrieval bake-off to confirm (or correct) the
BM25-only decision; add local cross-encoder rerank only if the numbers
justify it, and only as an optional layer that's never on the critical path.
**Status:** not started.

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

## Repo layout (target, mirrors the competition kit)

```
techjam2026-pipeline/
  starter/
    agent.py          # the graded Agent class
    ledger.py           # disclosed_constraints accumulation
    scheduler.py         # fixed six-attribute order (Phase 1) -> ask-yield (Phase 2)
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

## Related

- Planning source of truth: [`darrensimmx/techjam2026-docs`](https://github.com/darrensimmx/techjam2026-docs)
  — see `pipeline/implementation-plan.md` there for the detailed DoD checklist
  behind Phase 1.
- [`docs/benchmark-tracking-plan.md`](docs/benchmark-tracking-plan.md) — how
  Hit Rate/MRR/MTTC get tracked across phases so each change's effect is
  visible in git history, not just in someone's terminal.
- Competition kit: [`TechJam2026/techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search)
