"""Keep docs/pipeline-drawings.html honest against the code it draws.

The drawings hardcode facts that live in `starter/`: column weights, symbol names,
the six-attribute schedule, two numeric caps. This driver detects when those have
moved, and rewrites the mechanically-derivable ones.

It cannot republish the artifact -- that needs Claude's Artifact tool. See
SKILL.md; this script prepares, Claude publishes.

Standard library only, matching the project's own zero-dependency policy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# stdout may be a pipe under a git hook; keep ordering sane against subprocesses.
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parent
PINS = SKILL_DIR / "pins.json"

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def die(msg: str) -> None:
    print(f"{RED}error:{RESET} {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- extraction

def extract_one(rel_path: str, pattern: str, ident: str) -> tuple[str | None, str | None]:
    """Extract exactly one match, or explain why that failed.

    Zero matches or several both mean the code shape moved, which is itself
    drift. Returning None silently would make the check pass forever -- the
    worst available failure mode -- so every miss becomes a reported problem.
    """
    path = ROOT / rel_path
    if not path.exists():
        return None, f"{ident}: {rel_path} does not exist"
    hits = re.findall(pattern, path.read_text(encoding="utf-8"), re.S | re.M)
    if len(hits) != 1:
        return None, (
            f"{ident}: extractor matched {len(hits)} times in {rel_path}, expected exactly 1\n"
            f"      the code shape moved -- fix the pattern in drawings.py"
        )
    return hits[0], None


def actual_weights() -> tuple[str | None, str | None]:
    value, problem = extract_one(
        "starter/retrieval.py", r"bm25\(products,\s*([0-9.,\s]+?)\)", "bm25-weights"
    )
    return (re.sub(r"\s+", " ", value).strip() if value else None), problem


def actual_schedule() -> tuple[list[str] | None, str | None]:
    value, problem = extract_one(
        "starter/scheduler.py", r"FIXED_SCHEDULE\s*=\s*\((.*?)\)", "schedule"
    )
    return (re.findall(r'"([^"]+)"', value) if value else None), problem


def actual_term_cap() -> tuple[str | None, str | None]:
    # Anchored on the full call: a bare \[:(\d+)\] also matches hard_constraints[:2].
    return extract_one(
        "starter/retrieval.py",
        r"dict\.fromkeys\(_terms\(query_text\)\)\)\[:(\d+)\]",
        "term-cap",
    )


def actual_max_recs() -> tuple[str | None, str | None]:
    return extract_one("starter/agent.py", r"_MAX_RECOMMENDATIONS\s*=\s*(\d+)", "max-recs")


def symbol_present(rel_path: str, name: str) -> bool:
    path = ROOT / rel_path
    if not path.exists():
        return False
    src = path.read_text(encoding="utf-8")
    return re.search(rf"(?:^\s*def\s+|self\.){re.escape(name)}\b", src, re.M) is not None


def head_short() -> str | None:
    res = run(["git", "rev-parse", "--short", "HEAD"])
    return res.stdout.strip() if res.returncode == 0 else None


def watched_changed_since(commit: str, watch: list[str]) -> list[str] | None:
    """Watched files changed between `commit` and HEAD, or None if unanswerable."""
    if run(["git", "cat-file", "-e", f"{commit}^{{commit}}"]).returncode != 0:
        return None  # shallow clone, squashed history -- cannot answer
    if run(["git", "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
        return None  # drawn on a line this branch does not contain
    res = run(["git", "diff", "--name-only", f"{commit}..HEAD", "--", *watch])
    if res.returncode != 0:
        return None
    return [line for line in res.stdout.splitlines() if line.strip()]


def mid_rebase() -> bool:
    """True during a rebase/merge/cherry-pick, when post-commit fires per replayed commit."""
    res = run(["git", "rev-parse", "--git-path", "."])
    git_dir = ROOT / res.stdout.strip() if res.returncode == 0 else ROOT / ".git"
    return any(
        (git_dir / marker).exists()
        for marker in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD")
    )


# ---------------------------------------------------------------- checks

def cheap_checks(pins: dict, doc: str) -> list[str]:
    """Text-and-git-only drift checks. No catalog, no scoring."""
    problems: list[str] = []

    weights, problem = actual_weights()
    if problem:
        problems.append(problem)
    elif weights != pins["bm25_weights"]:
        problems.append(
            f"bm25 column weights changed\n"
            f"      pinned: {pins['bm25_weights']}\n"
            f"      actual: {weights}\n"
            f"      Plate 3 renders these as prose ('title N … description N'), so the caption breaks too"
        )
    else:
        parts = [p.strip() for p in weights.split(",")]
        if f"title {parts[1]}" not in doc or f"description {parts[-1]}" not in doc:
            problems.append(
                f"Plate 3's rendered weights disagree with starter/retrieval.py "
                f"(expected 'title {parts[1]} … description {parts[-1]}')"
            )

    schedule, problem = actual_schedule()
    if problem:
        problems.append(problem)
    elif schedule != pins["schedule"]:
        problems.append(
            f"FIXED_SCHEDULE changed\n"
            f"      pinned: {pins['schedule']}\n"
            f"      actual: {schedule}\n"
            f"      Plate 4's whole premise is the six-attribute order"
        )

    for ident, getter, pinned in (
        ("term-cap", actual_term_cap, pins["term_cap"]),
        ("max-recs", actual_max_recs, pins["max_recommendations"]),
    ):
        value, problem = getter()
        if problem:
            problems.append(problem)
        elif value != pinned:
            problems.append(f"{ident}: pinned {pinned}, actual {value} (Plate 3 states it)")

    for rel_path, names in pins["symbols"].items():
        for name in names:
            if not symbol_present(rel_path, name):
                problems.append(f"symbol {name} named in the drawings no longer exists in {rel_path}")

    # Doc-side rot: someone hand-edited the HTML and the pins went stale.
    for literal in pins.get("doc_literals", []):
        if literal not in doc:
            problems.append(f"drawings no longer contain the pinned string {literal!r}")

    pinned_commit = pins.get("commit")
    if pinned_commit:
        changed = watched_changed_since(pinned_commit, pins["watch"])
        if changed:
            listed = ", ".join(changed[:3]) + (" …" if len(changed) > 3 else "")
            problems.append(f"stale since {pinned_commit} — watched files changed: {listed}")
    return problems


def score_check(pins: dict) -> list[str]:
    """The expensive tier. Human-invoked only -- see SKILL.md for why not the hook."""
    catalog = ROOT / "data" / "catalog.jsonl"
    if not catalog.exists():
        return [
            "cannot verify Plate 5: data/catalog.jsonl is absent (gitignored, 60 MB).\n"
            "      A missing catalog scores 0.0 silently, so this check would report\n"
            "      a fake regression. Run: python3 .claude/skills/run-sol/bench.py setup"
        ]
    bench = ROOT / ".claude" / "skills" / "run-sol" / "bench.py"
    if not bench.exists():
        return [
            "cannot verify Plate 5: .claude/skills/run-sol/bench.py is missing "
            "(it lives on branch skill/run-sol-benchmark-harness)"
        ]
    res = run([sys.executable, str(bench), "eval"])
    if res.returncode != 0:
        return [f"bench.py eval failed:\n{res.stdout[-400:]}{res.stderr[-400:]}"]
    results = ROOT / "results.json"
    if not results.exists():
        return ["bench.py eval produced no results.json"]
    got = json.loads(results.read_text(encoding="utf-8"))
    problems = []
    for key, pinned in pins["score"].items():
        actual = got.get("recommended_technical_score") if key == "technical_score" else got.get(key)
        if actual is None:
            problems.append(f"results.json has no {key}")
        elif abs(float(actual) - float(pinned)) > 1e-6:
            problems.append(f"Plate 5 {key}: pinned {pinned}, measured {actual}")
    return problems


# ---------------------------------------------------------------- commands

def cmd_check(args) -> int:
    # Hook mode must never block, never fail, and never speak during a rebase.
    if args.hook:
        if os.environ.get("SOL_SKIP_DRAWINGS_CHECK") or mid_rebase():
            return 0
        try:
            pins = json.loads(PINS.read_text(encoding="utf-8"))
            doc = (ROOT / pins["doc"]).read_text(encoding="utf-8")
            problems = cheap_checks(pins, doc)
        except Exception:
            return 0  # unusual repo state is not the committer's problem
        if problems:
            head = problems[0].split("\n")[0]
            print(f"{YELLOW}drawings:{RESET} {head} — run {DIM}/update-drawings{RESET}")
        return 0

    if not PINS.exists():
        die(f"missing {PINS.relative_to(ROOT)} -- the skill is incomplete")
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    doc_file = ROOT / pins["doc"]
    if not doc_file.exists():
        die(f"missing {pins['doc']} -- restore it with: git checkout -- {pins['doc']}")
    doc = doc_file.read_text(encoding="utf-8")

    problems = cheap_checks(pins, doc)
    if args.score:
        problems += score_check(pins)

    if not problems:
        scope = "including Plate 5 score" if args.score else "cheap tier"
        print(f"{GREEN}ok{RESET} drawings match the code ({scope})")
        return 0
    print(f"{RED}drift{RESET} {pins['doc']} is out of date:\n")
    for problem in problems:
        print(f"  {RED}·{RESET} {problem}")
    print(f"\n{DIM}Refresh what is mechanical: drawings.py refresh{RESET}")
    print(f"{DIM}Then ask Claude to republish — see SKILL.md.{RESET}")
    return 1 if args.strict else 0


def cmd_refresh(args) -> int:
    """Rewrite the mechanically-derivable pins. Prose, SVG and Plate 5 stay manual."""
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    path = ROOT / pins["doc"]
    doc = path.read_text(encoding="utf-8")
    changes: list[str] = []

    weights, problem = actual_weights()
    if problem:
        die(problem)
    if weights != pins["bm25_weights"]:
        old = [p.strip() for p in pins["bm25_weights"].split(",")]
        new = [p.strip() for p in weights.split(",")]
        doc = doc.replace(
            f"title {old[1]} … description {old[-1]}",
            f"title {new[1]} … description {new[-1]}",
        )
        pins["bm25_weights"] = weights
        changes.append(f"bm25 weights -> {weights}")

    schedule, problem = actual_schedule()
    if not problem and schedule != pins["schedule"]:
        pins["schedule"] = schedule
        changes.append(f"schedule -> {schedule}  {YELLOW}(Plate 4 prose needs a human){RESET}")

    for key, getter in (("term_cap", actual_term_cap), ("max_recommendations", actual_max_recs)):
        value, problem = getter()
        if not problem and value != pins[key]:
            pins[key] = value
            changes.append(f"{key} -> {value}")

    head = head_short()
    if head and head != pins.get("commit"):
        doc = re.sub(r"(techjam2026-pipeline @ )[0-9a-f]{7,}", lambda m: m.group(1) + head, doc)
        doc = re.sub(r"(main @ )[0-9a-f]{7,}", lambda m: m.group(1) + head, doc)
        pins["commit"] = head
        changes.append(f"commit stamp -> {head}")

    if not changes:
        print(f"{GREEN}ok{RESET} nothing to refresh")
        return 0

    path.write_text(doc, encoding="utf-8")
    PINS.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    print(f"{GREEN}refreshed{RESET} {pins['doc']} and pins.json:")
    for change in changes:
        print(f"  · {change}")
    print(f"\n{DIM}Now ask Claude to republish — see SKILL.md 'Republish'.{RESET}")
    return 0


HOOK_MARKER = "# sol:update-drawings drift check"
HOOK_BODY = f"""#!/bin/sh
{HOOK_MARKER} -- installed by
# .claude/skills/update-drawings/drawings.py install-hook  (remove with --uninstall)
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
for p in python3 python; do
  command -v "$p" >/dev/null 2>&1 && \\
    "$p" "$root/.claude/skills/update-drawings/drawings.py" check --hook 2>/dev/null && break
done
exit 0
"""


def hooks_dir() -> Path:
    # --git-path is correct inside linked worktrees; --git-dir is not.
    res = run(["git", "rev-parse", "--git-path", "hooks"])
    if res.returncode != 0:
        die("not a git repository")
    return ROOT / res.stdout.strip()


def cmd_install_hook(args) -> int:
    hook = hooks_dir() / "post-commit"
    if args.uninstall:
        if hook.exists() and HOOK_MARKER in hook.read_text(encoding="utf-8"):
            hook.unlink()
            print(f"{GREEN}removed{RESET} {hook.name}")
        else:
            print(f"{DIM}nothing to remove{RESET}")
        return 0
    if hook.exists() and not args.force:
        existing = hook.read_text(encoding="utf-8")
        if HOOK_MARKER not in existing:
            die(
                f"{hook} exists and was not generated by this skill.\n"
                "  Refusing to overwrite. Append this instead, or pass --force:\n"
                '    python3 .claude/skills/update-drawings/drawings.py check --hook 2>/dev/null || true'
            )
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(HOOK_BODY, encoding="utf-8")
    hook.chmod(0o755)
    print(f"{GREEN}installed{RESET} {hook} (warn-only; never blocks a commit)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="drawings.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="detect drift between the drawings and the code")
    c.add_argument("--score", action="store_true", help="also re-score and check Plate 5 (~15s)")
    c.add_argument("--strict", action="store_true", help="exit 1 on drift (never use in CI)")
    c.add_argument("--hook", action="store_true", help="one terse line, always exit 0")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("refresh", help="rewrite the mechanically-derivable values")
    r.set_defaults(func=cmd_refresh)

    i = sub.add_parser("install-hook", help="install the warn-only post-commit hook")
    i.add_argument("--force", action="store_true", help="overwrite a foreign post-commit hook")
    i.add_argument("--uninstall", action="store_true", help="remove the hook")
    i.set_defaults(func=cmd_install_hook)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
