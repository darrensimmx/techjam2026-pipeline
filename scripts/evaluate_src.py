"""Score the NEW `src/` system with the vendored, unmodified evaluator.

## Why this script has to exist

`evaluator/local_evaluator.py` hard-codes `from starter.agent import Agent` at
line 12, so `python3 -m evaluator.local_evaluator` scores the SUPERSEDED Phase-1
system, not this one. The evaluator is vendored from the competition kit and is
never edited (a diff drift check enforces byte-identity), so that import cannot
be changed.

The way through is that `evaluate()` takes the agent as its FIRST PARAMETER
(local_evaluator.py:216). Import `evaluate`, `catalog_index` and `load_jsonl`,
construct OUR `Agent` from the repo-root `agent.py`, and hand it in. No
evaluator edit, no `starter/` touch. `scripts/leak_controlled_benchmark.py`
already uses exactly this pattern; this script is the plain, unpatched version
of it.

Note that importing `evaluator.local_evaluator` still executes its line-12
`from starter.agent import Agent`, so `starter/` must remain importable. That
import is never *used* here -- the agent we score is the one we pass in.

## Two reasons a number out of here can lie to you

**A score of exactly 0.0 is usually a data problem, not a solution problem.**
If `data/catalog.jsonl` is missing, `Agent.__init__` degrades to a null index by
design and every turn returns an empty but schema-valid response: no exception,
no warning, and a `recommended_technical_score` of 0.00000 that reads as a
catastrophic regression. `Agent.degraded` is checked BEFORE scoring and shouted
about, and the catalog row count and index size are printed, so the cases are
told apart on sight.

Measured caveat on that check. A MISSING catalog never actually reaches the
`degraded` branch here: the evaluator's own `catalog_index()` opens the same path
and raises `FileNotFoundError` first (local_evaluator.py:116), so the silent-zero
story from CLAUDE.md describes the ORGANIZER's harness -- which builds the agent
without a catalog_index of its own -- not this script. The explicit
`preflight_catalog()` below restores the intended unmissable message for that
case. `agent.degraded` still earns its place for the narrower failure it does
catch: a catalog that exists and parses as JSONL but that `Bm25Index` cannot
index (no FTS5, every row rejected). And a third case slips past both -- an index
with rows but no searchable TEXT reports `degraded=False` and still scores
0.00000 -- so the index size is printed and the zero alarm names it.

**An unrecorded number is a lost number.** `results*.json` are gitignored and
there is no committed baseline artifact, so every run appends one line to
`results_src.md` (newest first) with the timestamp, commit, branch, score and
`--note`. The commit carries a trailing `*` when tracked files were modified,
because such a row does not correspond to that commit alone.

Usage:
    python3 scripts/evaluate_src.py --note "ws-a index wired up"
    python3 -m scripts.evaluate_src --catalog tests/fixtures/catalog.jsonl \
        --dataset data/public_set.jsonl --output /tmp/results_src.json --note smoke

Exit status is 1 when the agent is degraded or the score rounds to 0.00000, so
the silent-zero case is loud in automation as well as on a terminal.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as _vendored  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.leak_controlled_benchmark import intent_card_scrubbed  # noqa: E402

from agent import Agent  # noqa: E402  -- the submission entry point, not src.agent

# The organizer's published weak-BM25 starter baseline (HitRate@10 0.125,
# MRR 0.068034, MTTC 9.81), restated through the evaluator's own formula
# (local_evaluator.py:279-280) so it compares like with like:
#     efficiency = (11.0 - 9.81) / 10.0                    = 0.119
#     score      = 0.50*0.125 + 0.30*0.068034 + 0.20*0.119 = 0.10671
# Derived, not measured locally. Same constant as scripts/verify_offline_safety.py.
BASELINE_TECHNICAL_SCORE = 0.10671

# The superseded `starter/` system over the 200-session public set, recorded in
# results.md at commit 70165ff ("control: pre-rebuild, starter/ untouched").
# This is the number the rebuild has to beat to be worth shipping. It is a LEAKY
# local number (see CLAUDE.md) -- an upper bound, not a score.
STARTER_TECHNICAL_SCORE = 0.692586

RESULTS_MD = ROOT / "results_src.md"
RESULTS_MD_HEADER = """# `src/` benchmark results

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
"""
_TABLE_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

_RULE = "=" * 72
_ALARM = "#" * 72


# --------------------------------------------------------------------------
# git metadata -- best effort, never fatal. A missing git is not a reason to
# throw away a scoring run.
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_commit() -> str:
    """Short SHA, with a trailing `*` when TRACKED files are modified.

    Untracked files (`??`) do not earn the star: a stray scratch file in the
    worktree does not change what the commit builds.
    """
    sha = _git("rev-parse", "--short", "HEAD") or "unknown"
    porcelain = _git("status", "--porcelain")
    dirty = any(line and not line.startswith("??") for line in porcelain.splitlines())
    return f"{sha}*" if dirty else sha


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _delta(value: float, reference: float) -> str:
    return f"{value - reference:+.6f}"


def print_reference_comparison(score: float, bracket_label: str = "leaky") -> None:
    """A bare score is meaningless without the two numbers that bracket it."""
    print()
    print(_RULE)
    print("REFERENCE COMPARISON")
    print(_RULE)
    print(f"  this run (src/, {bracket_label:<8})       {score:.6f}")
    print(f"  organizer weak-BM25 baseline     {BASELINE_TECHNICAL_SCORE:.6f}"
          f"   delta {_delta(score, BASELINE_TECHNICAL_SCORE)}")
    print(f"  starter/ @ 70165ff (superseded)  {STARTER_TECHNICAL_SCORE:.6f}"
          f"   delta {_delta(score, STARTER_TECHNICAL_SCORE)}")
    print()
    print("  Local numbers are inflated by the simulator leak (CLAUDE.md): the")
    print("  vendored evaluator builds the hidden customer card out of the target")
    print("  product's own listing. Both references above were measured under the")
    print("  same leak, so the DELTAS are comparable even though the absolute")
    print("  numbers are upper bounds. Bracket with scripts/leak_controlled_benchmark.py.")
    print(_RULE)


def _catalog_hint(catalog_path: str) -> None:
    print(f"#  Catalog asked for: {catalog_path}")
    print("#  data/catalog.jsonl is gitignored (~50k rows, distributed as a release")
    print("#  asset) -- see docs/windows-dev-setup.md section 1. Confirm with:")
    print("#    python3 -c \"from src.retrieval import Bm25Index; "
          "print(Bm25Index('data/catalog.jsonl').size)\"")


def preflight_catalog(catalog_path: str) -> bool:
    """Report a missing catalog OURSELVES, before catalog_index() raises on it.

    Without this the single most common failure in this repo prints a bare
    FileNotFoundError traceback, which reads like a bug in the harness rather
    than a missing release asset.
    """
    if Path(catalog_path).is_file():
        return True
    print()
    print(_ALARM)
    print("#  CATALOG NOT FOUND -- nothing can be scored")
    print("#")
    print("#  This is a DATA PROBLEM, NOT A SOLUTION REGRESSION. In the organizer's")
    print("#  harness a missing catalog is silent: Agent.__init__ swallows the load")
    print("#  failure by design, every turn returns an empty but schema-valid")
    print("#  response, and the run scores exactly 0.00000 with no warning. Here it")
    print("#  is loud, because the evaluator's own catalog_index() needs the same")
    print("#  file and would otherwise raise FileNotFoundError at you instead.")
    print("#")
    _catalog_hint(catalog_path)
    print(_ALARM)
    return False


def print_degraded_alarm(catalog_path: str) -> None:
    print()
    print(_ALARM)
    print("#  AGENT IS DEGRADED -- agent.degraded is True")
    print("#")
    print("#  The retrieval index did not build. Agent.__init__ swallows the load")
    print("#  failure BY DESIGN and sets a null index; nothing else warns. Every")
    print("#  turn will return an empty but schema-valid response, every session")
    print("#  will miss, and the score below will be exactly 0.00000.")
    print("#")
    print("#  THIS IS A DATA PROBLEM, NOT A SOLUTION REGRESSION.")
    print("#")
    _catalog_hint(catalog_path)
    print(_ALARM)


def index_size(agent: object) -> int | None:
    """The Bm25Index row count, if it is reachable. Never raises.

    Reaching through a private attribute is deliberate: `degraded` is a boolean
    and cannot tell "no index" apart from "an index of 0 rows" apart from "an
    index of 50k rows none of which matched". Only the number can.
    """
    index = getattr(getattr(agent, "_deps", None), "index", None)
    size = getattr(index, "size", None)
    return size if isinstance(size, int) else None


def print_zero_alarm(degraded: bool, rows: int | None) -> None:
    print()
    print(_ALARM)
    print("#  SCORE IS EXACTLY 0.00000")
    print("#")
    if degraded:
        print("#  The agent reported degraded=True above. The index did not build:")
        print("#  fix the catalog before reading anything into this number.")
    else:
        print(f"#  The agent did NOT report degraded (index rows: {rows}), so an index")
        print("#  built and this is a real zero. The usual causes, in order:")
        print("#    - the index has rows but no searchable TEXT, so every MATCH")
        print("#      returns nothing. `degraded` cannot see this -- it only asks")
        print("#      whether the index is empty. Check the row count above against")
        print("#      the catalog row count printed at the top.")
        print("#    - respond() raises on every turn (the evaluator swallows the")
        print("#      exception into an empty response -- no traceback ever prints)")
        print("#    - every response is schema-invalid and gets zeroed just as")
        print("#      silently (local_evaluator.py:243-244)")
        print("#    - recommendations are non-empty but no parent_asin is in the")
        print("#      catalog, so normalize_recommendations() drops all of them")
        print("#  Run `python3 -m unittest tests.test_src_end_to_end -v` -- it")
        print("#  instruments every turn and names which of these it is.")
    print(_ALARM)


def append_results_md(*, score: float, result: dict, note: str, degraded: bool,
                      dataset: str, path: Path = RESULTS_MD) -> None:
    """Append one row, newest first. Creates the file with a header if absent."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    safe_note = " ".join(str(note).split()).replace("|", "\\|") or "(no note)"
    if degraded:
        safe_note = f"DEGRADED (null index) -- {safe_note}"
    if dataset:
        safe_note = f"{safe_note} [dataset: {Path(dataset).name}]"

    row = (
        f"| {when} | `{git_commit()}` | `{git_branch()}` | **{score:.6f}** | "
        f"{_delta(score, STARTER_TECHNICAL_SCORE)} | "
        f"{_delta(score, BASELINE_TECHNICAL_SCORE)} | "
        f"{result.get('hit_rate_at_10', 0.0):.4f} | "
        f"{result.get('mrr', 0.0):.6f} | "
        f"{result.get('mttc', 0.0)} | {safe_note} |\n"
    )

    try:
        if not path.exists():
            path.write_text(RESULTS_MD_HEADER + row, encoding="utf-8")
        else:
            existing = path.read_text(encoding="utf-8")
            if _TABLE_SEPARATOR in existing:
                head, _, tail = existing.partition(_TABLE_SEPARATOR + "\n")
                path.write_text(head + _TABLE_SEPARATOR + "\n" + row + tail, encoding="utf-8")
            else:
                # Someone reshaped the file. Do not guess -- append rather than
                # rewrite, so nothing already recorded can be lost.
                path.write_text(existing.rstrip("\n") + "\n" + row, encoding="utf-8")
    except Exception as error:  # recording must never cost us the run
        print(f"\n!! could not append to {path}: {type(error).__name__}: {error}")
        print(f"!! record this row by hand:\n{row}")
        return
    print(f"\nrecorded in {path}")


# --------------------------------------------------------------------------

@contextmanager
def bracket(name: str):
    """Run the evaluator under one leak bracket.

    `leaky` is the vendored simulator as-shipped. Because public_set.jsonl
    carries no real intent_card, the evaluator falls back to building the
    "hidden" customer preferences out of the TARGET PRODUCT'S OWN LISTING and
    reciting them back turn by turn -- 94.5% of disclosed constraint strings
    are verbatim substrings of the target's indexed text. So a leaky score is
    an upper bound, not a score.

    `scrubbed` swaps in intent_card_scrubbed, which discloses only atomic
    attribute values (a material word, a colour word, a short structured detail,
    a budget number) and never a multi-word span lifted from features,
    description or title. That is the lower bound.

    The organizer's real held-out evaluator, backed by genuine customer
    profiles, should land somewhere between the two. NEVER QUOTE A LOCAL NUMBER
    WITHOUT SAYING WHICH BRACKET IT CAME FROM.

    evaluator/local_evaluator.py is never edited: the patch is applied to the
    imported module object at runtime and restored before this exits, which is
    the same discipline scripts/leak_controlled_benchmark.py uses.
    """
    if name != "scrubbed":
        yield
        return
    original = _vendored.intent_card
    _vendored.intent_card = intent_card_scrubbed
    try:
        yield
    finally:
        _vendored.intent_card = original


def print_bracket_summary(scores: dict) -> None:
    leaky = scores.get("leaky")
    scrubbed_v = scores.get("scrubbed")
    print()
    print(_RULE)
    print("LEAK BRACKET")
    print(_RULE)
    print(f"  leaky (upper bound)     {leaky:.6f}")
    print(f"  scrubbed (lower bound)  {scrubbed_v:.6f}")
    print(f"  spread                  {leaky - scrubbed_v:.6f}")
    print("  The organizer's held-out set should land between these two.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the src/ system with the vendored, unmodified evaluator.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results_src.json")
    parser.add_argument("--bracket", choices=("leaky", "scrubbed", "both"), default="leaky",
                        help="leak bracket to score under; 'both' reports the spread "
                             "(see the bracket() docstring for why this matters)")
    parser.add_argument("--note", default="",
                        help="free-text label recorded alongside this run in results_src.md")
    args = parser.parse_args()

    print(_RULE)
    print("evaluate_src -- src/ system, vendored evaluator, unpatched")
    print(_RULE)
    print(f"  catalog  {args.catalog}")
    print(f"  dataset  {args.dataset}")
    print(f"  output   {args.output}")
    print(f"  note     {args.note or '(none)'}")
    print(f"  commit   {git_commit()} on {git_branch()}")

    if not preflight_catalog(args.catalog):
        return 1

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    print(f"  catalog rows (unique parent_asin): {len(catalog_ids)}")
    print(f"  sessions in dataset:               {len(samples)}")

    agent = Agent(args.catalog)
    degraded = bool(getattr(agent, "degraded", False))
    rows = index_size(agent)
    if degraded:
        print_degraded_alarm(args.catalog)
    else:
        print(f"  agent.degraded: False (index built, {rows} rows)")
        if rows is not None and rows < len(catalog_ids):
            print(f"  !! index holds {rows} of {len(catalog_ids)} catalog rows -- "
                  f"{len(catalog_ids) - rows} were rejected as unindexable")

    brackets = ("leaky", "scrubbed") if args.bracket == "both" else (args.bracket,)
    scores: dict[str, float] = {}
    result = None
    for bracket_name in brackets:
        if len(brackets) > 1:
            print()
            print(_RULE)
            print(f"BRACKET: {bracket_name}")
            print(_RULE)
        with bracket(bracket_name):
            result = evaluate(agent, samples, catalog_ids, categories, products)
        scores[bracket_name] = float(result["recommended_technical_score"])
        if len(brackets) > 1:
            metrics = {k: v for k, v in result.items() if k != "sessions"}
            print(f"  hit@10 {metrics['hit_rate_at_10']:.4f}  mrr {metrics['mrr']:.6f}  "
                  f"mttc {metrics['mttc']:.3f}  score {scores[bracket_name]:.6f}")
    # Both reference scores were measured under the leak, so only a leaky
    # score is comparable to them. Comparing a scrubbed run against a leaky
    # reference mixes brackets -- the exact error this project forbids.
    score = scores.get("leaky", scores[brackets[-1]])
    reference_bracket = "leaky" if "leaky" in scores else brackets[-1]

    # Exactly the shape evaluator/local_evaluator.py:308 prints: headline
    # metrics plus the four per-scenario breakdowns, minus the per-session dump.
    print()
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))

    print()
    print(_RULE)
    print("PER-SCENARIO BREAKDOWN")
    print(_RULE)
    print(f"  {'scenario':<18}{'n':>5}{'hit@10':>10}{'mrr':>10}{'mttc':>9}")
    for name, metrics in sorted(result.get("scenario_metrics", {}).items()):
        print(f"  {name:<18}{metrics['sample_count']:>5}"
              f"{metrics['hit_rate_at_10']:>10.4f}{metrics['mrr']:>10.4f}"
              f"{metrics['mttc'] if metrics['mttc'] is not None else float('nan'):>9.3f}")

    print_reference_comparison(score, reference_bracket)

    if len(scores) > 1:
        print_bracket_summary(scores)

    if round(score, 5) == 0.0:
        print_zero_alarm(degraded, rows)

    try:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nfull per-session result written to {args.output}")
    except Exception as error:
        print(f"\n!! could not write {args.output}: {type(error).__name__}: {error}")

    bracket_note = f"[{args.bracket}] {args.note}".strip()
    append_results_md(score=score, result=result, note=bracket_note,
                      degraded=degraded, dataset=args.dataset)

    return 1 if (degraded or round(score, 5) == 0.0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
