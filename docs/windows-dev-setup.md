# Windows dev setup — running and testing this repo

Verified on Windows 11, Python 3.11.9. Commands are given for **PowerShell**
(the default in Windows Terminal / VS Code) and, where they differ, for **Git
Bash**.

There are no third-party dependencies — `requirements.txt` is deliberately
empty (comments only). Standard library only.

---

## 1. One-time setup

### Prerequisites

- **Python 3.11+** from [python.org](https://www.python.org/downloads/windows/).
  Tick *"Add python.exe to PATH"* in the installer.
- **Git for Windows** (gives you Git Bash — needed only for the `.sh` script, see §5).

Check:

```powershell
python --version
```

If `python` opens the Microsoft Store instead of running, disable the App
Execution Alias: *Settings → Apps → Advanced app settings → App execution
aliases* → turn off the `python.exe` and `python3.exe` entries.

> On Windows the interpreter is `python`, not `python3`. Every docstring and
> README line in this repo says `python3 -m ...` — that is the macOS/Linux
> spelling and **will fail here**. Substitute `python`.

### Clone and enter

```powershell
git clone https://github.com/darrensimmx/techjam2026-pipeline.git
```

```powershell
cd techjam2026-pipeline
```

### Virtual environment (optional but recommended)

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Git Bash equivalent:

```bash
source .venv/Scripts/activate
```

`.venv/` is already in `.gitignore`.

### Get the catalog (required for the CLIs and the evaluator, NOT for tests)

`data/catalog.jsonl` is ~50,000 rows and is **not** on `main` (gitignored,
distributed as a GitHub Release asset).

The quickest route: the unmerged `benchmark-tracking` branch has
`data/catalog.jsonl.gz` and `data/SHA256SUMS` committed.

```powershell
git checkout origin/benchmark-tracking -- data/catalog.jsonl.gz data/SHA256SUMS
```

Otherwise download `catalog.jsonl.gz` from the
[competition kit's releases page](https://github.com/TechJam2026/techjam-conversational-search/releases).
Either way, decompress it:

```powershell
tar -xzf catalog.jsonl.gz
```

```powershell
Move-Item catalog.jsonl data\catalog.jsonl
```

Verify against the published `SHA256SUMS`:

```powershell
Get-FileHash data\catalog.jsonl -Algorithm SHA256
```

**Without this file the agent still starts and still returns schema-valid
responses — it just returns zero recommendations on every turn and scores
0.0.** `Agent.__init__` swallows the load failure by design and nothing prints
a warning. See §6.

### Vendored model weights (Tier 2 centroid, cross-encoder rerank)

Same convention as the catalog above: gitignored (`data/models/`), not on
`main`, fetched once locally. Without either, the corresponding layer degrades
silently to its null implementation — `NullSemanticDecoder` /
`NullReranker` — exactly as if the layer were still disabled.

```powershell
pip install model2vec sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M').save_pretrained('data/models/potion-base-8m')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').save('data/models/ms-marco-MiniLM-L-6-v2')"
```

Sanity check both loaded (not the null fallback):

```powershell
python -c "from src.semantic import load_semantic_decoder; print(load_semantic_decoder(enabled=True).name)"   # rung3_centroid
python -c "from src.rerank import load_reranker; print(load_reranker().name)"                                  # cross-encoder/ms-marco-MiniLM-L-6-v2
```

Sanity check that the catalog actually loaded:

```powershell
python -c "from starter.retrieval import Bm25Index; print(Bm25Index('data/catalog.jsonl').search('waterproof leather boots', 5))"
```

Expect five `parent_asin` strings. An empty list means the catalog is missing
or malformed.

---

## 2. Running the test suite

**Always run from the repo root.** Every test resolves paths relative to it,
and `starter` / `evaluator` / `cli` are imported as top-level packages.

### The command that actually works

```powershell
python -m unittest tests.test_agent_contract tests.test_cli_integration tests.test_evaluator_smoke tests.test_ledger_scheduler tests.test_offline tests.test_p1_offline_safety -v
```

Expected: `Ran 22 tests` … `OK`.

### Or just use discovery

```powershell
python -m unittest discover
```

Also `Ran 22 tests` … `OK` — the same 22, since the explicit list above names
every test module. This works because `tests/__init__.py` exists — do not delete
it. Without it `tests/` is not an importable package, discovery walks straight
past it, and the run reports `Ran 0 tests in 0.000s` followed by **`OK`**: green
output meaning "found nothing", not "everything passed". If you ever see a
suspiciously fast `OK`, check the test count before believing it.

### Mirroring the two CI jobs

`.github/workflows/ci.yml` runs on `ubuntu-latest` in two jobs. To reproduce
them job-for-job:

```powershell
python -m unittest tests.test_agent_contract tests.test_ledger_scheduler tests.test_offline tests.test_p1_offline_safety -v
```

```powershell
python -m unittest tests.test_cli_integration tests.test_evaluator_smoke -v
```

Job 2 spawns `cli/agent_server.py` as a real subprocess via `sys.executable`,
so it uses whichever interpreter is active — activate your venv first, or it
runs against the system Python.

### Running one class or one test

```powershell
python -m unittest tests.test_p1_offline_safety.TestRespondSafety -v
```

```powershell
python -m unittest tests.test_agent_contract.TestAgentContract.test_never_raises_on_malformed_input -v
```

---

## 3. Manual chat with the agent (the terminal workflow)

### Option A — one terminal (the normal way)

`cli/client.py` spawns its own `cli/agent_server.py` subprocess. You do **not**
need a second terminal:

```powershell
python -m cli.client --catalog data\catalog.jsonl
```

You get an interactive REPL, up to 10 turns, `/quit` to stop early:

```
Session 8f3a... -- up to 10 turns. Type /quit to stop early.

[turn 1] you: I'm looking for waterproof hiking boots
[turn 1] agent: Do you have a material preference?
           asking about: material
           top-10: ['B09...', 'B07...', ...]
```

To try it without downloading the catalog, point it at the 6-row fixture:

```powershell
python -m cli.client --catalog tests\fixtures\catalog.jsonl
```

### Option B — two terminals (driving the server by hand)

Useful when you want to see the raw JSON protocol or replay a fixed script.

**Terminal 1** — start the server. It reads newline-delimited JSON on stdin and
writes one JSON object per line to stdout:

```powershell
python -m cli.agent_server --catalog data\catalog.jsonl
```

It sits there with no prompt and no banner. That is normal — it is blocked on
`stdin`. Type or paste one JSON object per line:

```json
{"op":"reset","session_id":"s1","user_profile":{"purchase_frequency":"3-4 prior purchases","average_prior_rating":4.5,"rating_style":"usually positive","preference_tags":["comfort"],"summary":"manual"}}
```

```json
{"op":"respond","session_id":"s1","user_message":"waterproof leather boots","turn":1,"top_k":10}
```

Stop it with **Ctrl+C** (or Ctrl+Z then Enter to close stdin cleanly).

**Terminal 2** — note there is no client that attaches to an *already running*
server; `client.py` always spawns its own. So the second terminal is for
scripted piping instead. PowerShell:

```powershell
'{"op":"reset","session_id":"s1","user_profile":{}}', '{"op":"respond","session_id":"s1","user_message":"waterproof leather boots","turn":1,"top_k":10}' | python -m cli.agent_server --catalog tests\fixtures\catalog.jsonl
```

Git Bash:

```bash
printf '%s\n' '{"op":"reset","session_id":"s1","user_profile":{}}' '{"op":"respond","session_id":"s1","user_message":"waterproof leather boots","turn":1,"top_k":10}' | python -m cli.agent_server --catalog tests/fixtures/catalog.jsonl
```

Expected output:

```json
{"ok": true}
{"message": "Do you have a material preference?", "ask_attribute": "material", "recommendations": [{"parent_asin": "T0001"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
```

> `agent_server.py` does not guard `json.loads` or the `request[...]` lookups.
> One malformed line or one missing key kills the process with a traceback, and
> `client.py` then dies with `agent_server exited unexpectedly`. Send
> well-formed JSON.

---

## 4. Running the full evaluator

Needs `data/catalog.jsonl` (§1) and `data/public_set.jsonl` (committed).

```powershell
python -m evaluator.local_evaluator --catalog data\catalog.jsonl --dataset data\public_set.jsonl --output results.json
```

Prints the aggregate block and writes per-session detail to `results.json`
(gitignored). The number to watch is `recommended_technical_score`.

Instrumented variant — same run, plus the five P1 acceptance criteria checked
per call:

```powershell
python scripts\verify_offline_safety.py
```

Writes `results_offline.json`. Read §6 before trusting its `[PASS]` lines.

---

## 5. Offline verification does NOT run on Windows

`scripts/verify_offline_safety.sh` is **macOS only**. It depends on
`sandbox-exec` (macOS Seatbelt) with the `scripts/no-network.sb` profile to
revoke networking at the kernel level. There is no `sandbox-exec` on Windows;
Git Bash fails at step 1b with `command not found`.

What you *can* run on Windows:

```powershell
python scripts\netprobe.py
```

The control probe on its own. Exit 1 = network reachable (expected when
unsandboxed). Exit 0 = every operation denied.

```powershell
python scripts\verify_offline_safety.py
```

The instrumented evaluator run, **without** a network block. This covers
criteria 1–3. It does **not** establish criteria 4/5 "with networking revoked".

For a real kernel-level block, run the evaluator in a container with no
network adapter — the closest Windows equivalent to `sandbox-exec -f
scripts/no-network.sb`:

```powershell
docker run --rm --network none -v ${PWD}:/app -w /app python:3.11-slim python -m evaluator.local_evaluator
```

The `.sh` driver and the `.sb` profile do not cover this path — you are
supplying the block yourself, so say so explicitly when reporting the result.

---

## 6. Gotchas, and green output that means nothing

| Symptom | Cause | Fix |
|---|---|---|
| `python3 : not recognized` | Docstrings/README use the POSIX spelling | Use `python` |
| `unittest discover` says `OK` instantly with 0 tests | `tests/__init__.py` was deleted | Restore it (§2) |
| `ModuleNotFoundError: starter` | Not in the repo root | `cd` to the repo root |
| `recommendations: []` every turn, no error | `data/catalog.jsonl` missing; `Agent.__init__` swallows it and sets a null index | Fetch the catalog (§1), then run the `Bm25Index` check |
| `verify_offline_safety.py` fails criterion 4 with `0/768 turns returned recommendations` | Same null-index cause as the row above | Fetch the catalog (§1) |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `.\scripts\verify_offline_safety.sh` fails | Bash + macOS-only `sandbox-exec` | See §5 |

### Line endings

Git for Windows may check `scripts/*.sh` out with CRLF, which breaks the
shebang under Git Bash / WSL (`bad interpreter: /usr/bin/env bash^M`). If you
intend to run them under WSL:

```bash
git config core.autocrlf input
```

Then re-checkout the working tree so the change takes effect.

---

## 7. What a green test run does and does not prove

Read this before reporting "all tests pass."

- **`tests/fixtures/catalog.jsonl` has 6 products and `top_k` is 10.** Any
  query matching at least one term returns essentially the whole fixture
  catalog, so `test_evaluator_smoke`'s `assertTrue(...["hit"])` is satisfied
  even by a ranker that ignores the query entirely — swapping in a
  reverse-sorted, query-blind ranker still yields `hit: True,
  first_hit_turn: 1`. The test proves the plumbing runs; it says nothing about
  retrieval quality.
- **`test_cli_integration` is a spine smoke test, not a termination proof.** It
  drives 10 turns because that is the competition's session bound, but the
  agent has no turn-limit logic — the caller owns the stop condition, and under
  grading that caller is the organizer's evaluator. The test checks that the
  endpoint answers, the payloads are contract-shaped, and retrieval is still
  returning results at turn 10. Early termination on a hit is covered through
  the real evaluator in `test_evaluator_smoke.py`.
- **`test_offline`** is an AST check for banned import *names* in `starter/*.py`
  only. It executes nothing, does not cover `evaluator/` or `cli/`, and would
  not catch a dynamic import. The real networking-disabled proof is Phase 5
  (§5).

Retrieval quality is only measurable against the real 50k catalog via §4. There
is no committed results artifact to compare against (`results*.json` are all
gitignored), so record whatever number you get somewhere durable.
