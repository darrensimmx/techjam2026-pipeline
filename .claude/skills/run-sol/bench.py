#!/usr/bin/env python3
"""Driver for the TechJam conversational-search agent.

Everything an agent needs to build, run, score, and regression-check this
project. Standard library only, matching the project's own zero-dependency
policy (requirements.txt).

Run from the repo root:  python3 .claude/skills/run-sol/bench.py <command>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Subprocesses write straight to fd 1; without line buffering our own
# prints land out of order whenever stdout is a pipe rather than a tty.
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parent
CATALOG = ROOT / "data" / "catalog.jsonl"
RESULTS_MD = ROOT / "results.md"
DATASET = ROOT / "data" / "public_set.jsonl"

# The catalog is gitignored and absent from main. It is NOT necessary to
# download it from the GitHub Release: the gzipped blob is already in the
# local object store on this branch.
CATALOG_REF = "origin/benchmark-tracking:data/catalog.jsonl.gz"
CATALOG_SHA256 = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"
CATALOG_ROWS = 50000

BASELINES = json.loads((SKILL_DIR / "baselines.json").read_text())

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""


def run(cmd, **kw):
    kw.setdefault("cwd", ROOT)
    return subprocess.run(cmd, **kw)


def die(msg: str) -> "None":
    print(f"{RED}error:{RESET} {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- setup

def ensure_catalog(quiet: bool = False) -> None:
    """Restore data/catalog.jsonl if missing. Never touches the git index."""
    if CATALOG.exists():
        rows = sum(1 for _ in CATALOG.open())
        if rows == CATALOG_ROWS:
            if not quiet:
                print(f"{GREEN}ok{RESET} catalog present ({rows} rows)")
            return
        print(f"{YELLOW}warn{RESET} catalog has {rows} rows, expected {CATALOG_ROWS}; rebuilding")

    print(f"{DIM}restoring catalog from {CATALOG_REF}{RESET}")
    proc = run(["git", "show", CATALOG_REF], capture_output=True)
    if proc.returncode != 0:
        die(
            "could not read the catalog blob.\n"
            f"  tried: git show {CATALOG_REF}\n"
            "  fetch the branch (git fetch origin benchmark-tracking), or download\n"
            "  catalog.jsonl.gz from the TechJam2026 GitHub Release into data/."
        )
    blob = proc.stdout
    digest = hashlib.sha256(blob).hexdigest()
    if digest != CATALOG_SHA256:
        die(f"catalog sha256 mismatch\n  expected {CATALOG_SHA256}\n  got      {digest}")

    import gzip
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_bytes(gzip.decompress(blob))
    rows = sum(1 for _ in CATALOG.open())
    if rows != CATALOG_ROWS:
        die(f"catalog decompressed to {rows} rows, expected {CATALOG_ROWS}")
    print(f"{GREEN}ok{RESET} catalog restored ({rows} rows, sha256 verified)")


def assert_index_loads() -> None:
    """Guard the project's nastiest failure mode.

    Agent.__init__ swallows a catalog load failure and sets self._index = None.
    Every turn then returns an empty, schema-valid recommendation list and the
    run scores 0.0 with no error anywhere. Fail loudly here instead.
    """
    code = (
        "from starter.retrieval import Bm25Index;"
        "h=Bm25Index('data/catalog.jsonl').search('waterproof leather boots',5);"
        "assert len(h)==5, h; print(' '.join(h))"
    )
    proc = run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        die("BM25 index did not build from data/catalog.jsonl:\n" + proc.stderr.strip())
    print(f"{GREEN}ok{RESET} index builds, sample hits: {proc.stdout.strip()}")


RESULTS_HEADER = """# Benchmark results

Auto-updated by `.claude/skills/run-sol/bench.py` on every `eval` and `check`.
Newest run first. `score` is `recommended_technical_score` over the 200-session
public set: `0.50*hit@10 + 0.30*mrr + 0.20*efficiency`.

Reference points — organizer weak-BM25 baseline `0.106710`; `phase1-baseline`
@ `ecacc52` `0.722818`. See `docs/ledger-freeze-regression.md`.

A `*` after the commit means the worktree had uncommitted changes, so that row
does not correspond to the commit alone.

| when (UTC) | commit | branch | score | delta | hit@10 | mrr | mttc | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""

_ROW_SEPARATOR = "| --- |"


def _git(*args: str) -> str:
    proc = run(["git", *args], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _previous_score() -> "float | None":
    """Score from the newest existing row, for the delta column."""
    if not RESULTS_MD.exists():
        return None
    for line in RESULTS_MD.read_text().splitlines():
        if not line.startswith("| 20"):  # data rows begin with a date
            continue
        cells = [c.strip().strip("*`") for c in line.strip("|").split("|")]
        try:
            return float(cells[3])
        except (IndexError, ValueError):
            return None
    return None


def record_result(res: dict, note: str = "") -> None:
    """Prepend one row to results.md. Newest first, so the latest run reads first."""
    prev = _previous_score()
    score = res["recommended_technical_score"]
    delta = "—" if prev is None else f"{score - prev:+.6f}"
    sha = _git("rev-parse", "--short", "HEAD") or "?"
    # Only tracked modifications count: results.md is itself untracked on its
    # first run, and the catalog/results.json artifacts are always present.
    if _git("status", "--porcelain", "--untracked-files=no"):
        sha += "*"  # worktree dirty: the row is not attributable to this commit
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    row = (f"| {when} | `{sha}` | `{branch}` | **{score:.6f}** | {delta} | "
           f"{res['hit_rate_at_10']:.4f} | {res['mrr']:.6f} | {res['mttc']:.3f} | "
           f"{note} |")

    text = RESULTS_MD.read_text() if RESULTS_MD.exists() else RESULTS_HEADER
    idx = text.find(_ROW_SEPARATOR)
    if idx == -1:  # header missing or hand-mangled; rebuild it
        text = RESULTS_HEADER
        idx = text.find(_ROW_SEPARATOR)
    eol = text.index("\n", idx) + 1
    RESULTS_MD.write_text(text[:eol] + row + "\n" + text[eol:])
    print(f"{DIM}recorded to {RESULTS_MD.name}{RESET}")


# ---------------------------------------------------------------- tests

def cmd_test(args) -> int:
    failures = []

    print(f"\n{DIM}--- project suite (unittest discover) ---{RESET}")
    # Runs against tests/fixtures/catalog.jsonl; no real catalog needed.
    if run([sys.executable, "-m", "unittest", "discover", "-v"]).returncode != 0:
        failures.append("project suite")

    official = (Path(args.official_repo).expanduser() if args.official_repo
                else ROOT.parent / "techjam-conversational-search")
    src = official / "tests" / "test_evaluator.py"
    print(f"\n{DIM}--- organizer's evaluator test ---{RESET}")
    if not src.exists():
        print(f"{YELLOW}skip{RESET} organizer repo not found at {official}")
    else:
        # Drift guard: our evaluator/ is a verbatim vendored copy. If it ever
        # diverges, the organizer's own test is what catches it.
        ours, theirs = ROOT / "evaluator", official / "evaluator"
        diff = run(["diff", "-r", "--exclude=__pycache__", str(ours), str(theirs)],
                   capture_output=True, text=True)
        if diff.returncode == 0:
            print(f"{GREEN}ok{RESET} vendored evaluator/ is byte-identical to the organizer's")
        else:
            print(f"{RED}DRIFT{RESET} vendored evaluator/ differs from the organizer's:")
            print(diff.stdout)
            failures.append("evaluator drift")

        # Both repos ship a `tests` package AND an `evaluator` package, so
        # stacking them on PYTHONPATH resolves the wrong one. Copying the test
        # out and running it standalone makes sys.path[0] a directory that has
        # neither, so PYTHONPATH supplies ours unambiguously.
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "test_official_evaluator.py"
            shutil.copy(src, dst)
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            if run([sys.executable, str(dst), "-v"], env=env).returncode != 0:
                failures.append("organizer evaluator test")

    if failures:
        print(f"\n{RED}FAIL{RESET} " + ", ".join(failures))
        return 1
    print(f"\n{GREEN}PASS{RESET} all suites")
    return 0


# ---------------------------------------------------------------- eval

def score_tree(tree: Path, output: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "evaluator.local_evaluator",
         "--catalog", str(CATALOG), "--dataset", str(DATASET), "--output", str(output)],
        cwd=tree, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        die(f"evaluator failed in {tree}:\n{proc.stderr.strip()}")
    return json.loads(output.read_text())


def fmt(res: dict) -> str:
    return (f"hit@10 {res['hit_rate_at_10']:.4f}  mrr {res['mrr']:.6f}  "
            f"mttc {res['mttc']:.4f}  score {res['recommended_technical_score']:.6f}")


def cmd_eval(args) -> int:
    ensure_catalog(quiet=True)
    assert_index_loads()
    out = ROOT / "results.json"
    print(f"{DIM}running 200 sessions...{RESET}")
    res = score_tree(ROOT, out)
    print(f"\n{fmt(res)}\n")

    print(f"{'scenario':<16}{'n':>5}{'hit@10':>10}{'mrr':>11}{'mttc':>9}")
    for name, m in sorted(res["scenario_metrics"].items()):
        print(f"{name:<16}{m['sample_count']:>5}{m['hit_rate_at_10']:>10.4f}"
              f"{m['mrr']:>11.6f}{m['mttc']:>9.4f}")

    score = res["recommended_technical_score"]
    print()
    for ref in BASELINES["references"]:
        delta = score - ref["technical_score"]
        mark = f"{GREEN}+{delta:.6f}{RESET}" if delta >= 0 else f"{RED}{delta:.6f}{RESET}"
        print(f"  vs {ref['label']:<34} {ref['technical_score']:.6f}  {mark}")
    print(f"\nfull results: {out.relative_to(ROOT)}  {DIM}(gitignored){RESET}")
    if not args.no_record:
        record_result(res, args.note or "")
    return 0


def cmd_check(args) -> int:
    """Regression gate. Non-zero exit if the score drops below the guard."""
    ensure_catalog(quiet=True)
    out = Path(tempfile.mkdtemp()) / "results.json"
    res = score_tree(ROOT, out)
    score = res["recommended_technical_score"]
    guard = args.min if args.min is not None else BASELINES["regression_guard"]
    print(fmt(res))
    if not args.no_record:
        record_result(res, args.note or "")
    if score + 1e-9 < guard:
        print(f"{RED}REGRESSION{RESET} {score:.6f} < guard {guard:.6f} "
              f"({score - guard:+.6f})")
        print(f"{DIM}bisect with: python3 .claude/skills/run-sol/bench.py bisect <rev>...{RESET}")
        return 1
    print(f"{GREEN}OK{RESET} {score:.6f} >= guard {guard:.6f}")
    return 0


def cmd_bisect(args) -> int:
    """Score each revision in an isolated tree. Never mutates the worktree."""
    ensure_catalog(quiet=True)
    revs = args.revs or ["HEAD"]
    print(f"{'rev':<10}{'hit@10':>9}{'mrr':>11}{'mttc':>9}{'score':>11}  subject")
    prev = None
    for rev in revs:
        sha = run(["git", "rev-parse", "--short", rev],
                  capture_output=True, text=True).stdout.strip()
        subj = run(["git", "log", "-1", "--format=%s", rev],
                   capture_output=True, text=True).stdout.strip()
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "t"
            tree.mkdir()
            # git archive writes no metadata and never touches the index or
            # the worktree -- unlike `git checkout <rev> -- <path>`, which
            # stages files and leaves the tree dirty.
            arch = run(["git", "archive", rev], capture_output=True)
            subprocess.run(["tar", "-x", "-C", str(tree)], input=arch.stdout, check=True)
            (tree / "data" / "catalog.jsonl").unlink(missing_ok=True)
            (tree / "data" / "catalog.jsonl").symlink_to(CATALOG)
            res = score_tree(tree, tree / "r.json")
        s = res["recommended_technical_score"]
        mark = ""
        if prev is not None:
            d = s - prev
            if abs(d) > 1e-6:
                mark = f"  {RED if d < 0 else GREEN}{d:+.6f}{RESET}"
        prev = s
        print(f"{sha:<10}{res['hit_rate_at_10']:>9.4f}{res['mrr']:>11.6f}"
              f"{res['mttc']:>9.4f}{s:>11.6f}  {subj[:48]}{mark}")
    return 0


def cmd_setup(args) -> int:
    ensure_catalog()
    assert_index_loads()
    print(f"{GREEN}ok{RESET} ready")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("setup", help="restore the catalog and verify the index builds")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("test", help="project suite + the organizer's evaluator test")
    sp.add_argument("--official-repo", help="path to techjam-conversational-search")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("eval", help="score all 200 public sessions, with comparisons")
    sp.add_argument("--note", help="note recorded alongside this run in results.md")
    sp.add_argument("--no-record", action="store_true", help="skip the results.md row")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("check", help="regression gate; exits 1 if score drops")
    sp.add_argument("--min", type=float, help="override the guard score")
    sp.add_argument("--note", help="note recorded alongside this run in results.md")
    sp.add_argument("--no-record", action="store_true", help="skip the results.md row")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("bisect", help="score one or more revisions in isolated trees")
    sp.add_argument("revs", nargs="*", help="revisions (default: HEAD)")
    sp.set_defaults(func=cmd_bisect)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
