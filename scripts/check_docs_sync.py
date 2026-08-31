"""CI check: verify techjam2026-docs' claims about this repo's code, and its
phase-numbering references, still match reality.

This does not try to parse arbitrary prose. It runs a small, curated table of
facts -- each one a claim this repo or a review has already caught drifting
once (see FACTS below for the incident each guards against) -- checking a
regex in the docs repo's markdown against a predicate over this repo's actual
files. Extend FACTS when a new claim is worth guarding; this is a regression
net for known failure modes, not a general drift detector.

Needs read access to the (private) darrensimmx/techjam2026-docs repo:
    TECHJAM_CROSS_REPO_TOKEN=<token with read access to techjam2026-docs> \\
        python scripts/check_docs_sync.py

Locally, a token with access already on this machine works too:
    TECHJAM_CROSS_REPO_TOKEN=$(gh auth token) python scripts/check_docs_sync.py

Exit 0 = all facts hold (PASS/WARN only). Exit 1 = at least one FAIL.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DOCS_REPO = "darrensimmx/techjam2026-docs"
DOCS_REF = os.environ.get("DOCS_REPO_REF", "main")
REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN = os.environ.get("TECHJAM_CROSS_REPO_TOKEN") or os.environ.get("GITHUB_TOKEN")

_docs_cache: dict[str, str] = {}


def fetch_docs(path: str) -> str:
    """Via the Contents API (not raw.githubusercontent.com): that CDN caches
    for several minutes, which made this check flap against content that had
    already been pushed and fixed. The API reflects main immediately."""
    if path in _docs_cache:
        return _docs_cache[path]
    url = f"https://api.github.com/repos/{DOCS_REPO}/contents/{path}?ref={DOCS_REF}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.raw")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = f"__FETCH_ERROR_{exc.code}__"
    except urllib.error.URLError as exc:
        text = f"__FETCH_ERROR_{exc.reason}__"
    _docs_cache[path] = text
    return text


def read_local(path: str) -> str:
    p = REPO_ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() else ""


_QUOTE_INTRO = re.compile(r"\b(claimed|said|says|note said)\b", re.IGNORECASE)


def _is_quoted_reference(text: str, start: int, end: int) -> bool:
    """True if the match is a reported quote of an already-debunked claim,
    not a live one. This repo's own correction pattern is consistently 'an
    earlier {draft,revision,note} {said,claimed} "X"' -- checking for that
    introducing verb in the ~80 chars before the match is far more reliable
    than checking for adjacent quote marks: the quoted span is often longer
    than the matched phrase itself (e.g. the match is just "shipped as of"
    inside the quoted "shipped as of 28 Aug 2026", so a closing quote mark
    sits well past the match end, not immediately after it)."""
    before = text[max(0, start - 80):start]
    return bool(_QUOTE_INTRO.search(before))


def any_docs_match(
    paths: list[str], pattern: re.Pattern, skip_quoted_references: bool = False
) -> tuple[bool, str]:
    """Returns (matched, evidence) for the first hit. With
    skip_quoted_references=True (use for "is this claimed shipped/built"
    checks), a hit that's a reported quote of an already-debunked claim is
    skipped (see _is_quoted_reference) rather than counted as live. Leave
    False for checks that aren't scanning for a claim at all (e.g. the
    phase-crosswalk description match), where nothing needs filtering.
    Match patterns must otherwise be precise enough to be unambiguous on
    their own -- an earlier version of this used a generic "negation nearby"
    proximity window instead, which both missed real claims (an unrelated
    "not" sitting between the trigger words) and flagged corrected prose
    (nearby "not"/"never" that had nothing to do with the claim). Fetch
    errors = no-match, not a crash."""
    for path in paths:
        text = fetch_docs(path)
        if text.startswith("__FETCH_ERROR_"):
            continue
        for m in pattern.finditer(text):
            if skip_quoted_references and _is_quoted_reference(text, m.start(), m.end()):
                continue
            snippet = re.sub(r"\s+", " ", m.group(0))[:160]
            return True, f"{path}: …{snippet}…"
    return False, ""


# ---------------------------------------------------------------------------
# FACTS: (id, description, check) -> "PASS" | "WARN" | "FAIL", detail
# ---------------------------------------------------------------------------

def check_override_risk_not_overclaimed() -> tuple[str, str]:
    """Guards the incident fixed 28 Aug 2026: docs claimed the override-risk
    preference-stability layer was 'shipped'/'built' when no such code exists
    in starter/ or scripts/ here. Matches the specific phrasings the incident
    actually used (unambiguously positive constructions), not generic
    proximity to "shipped" -- a co-occurrence window is too easily tripped by
    unrelated "not"s in normal prose in both directions (see git history on
    this line for the false-positive/false-negative round that motivated
    this)."""
    claim = re.compile(
        r"\bwe shipped\b|\bshipped as of\b|\bbuilt, log-only, traced\b|"
        r"\bships it\s*—\s*built\b",
        re.IGNORECASE,
    )
    matched, evidence = any_docs_match(
        [
            "features/ask-yield-ledger/one-pager.md",
            "project/architecture-diagram.md",
            "project/architecture-diagram-v3-notes.md",
            "project/design-strengths.md",
        ],
        claim,
        skip_quoted_references=True,
    )
    code_present = any(
        re.search(r"override.risk|preference.stabilit", read_local(f"starter/{name}"), re.IGNORECASE)
        for name in ("agent.py", "ledger.py", "retrieval.py", "scheduler.py")
    )
    if matched and not code_present:
        return "FAIL", f"docs claim override-risk is shipped, but no matching code exists here. {evidence}"
    if matched and code_present:
        return "PASS", "docs claim shipped, and matching code now exists — claim caught up with reality"
    return "PASS", "docs do not claim override-risk is shipped (correct — it isn't built)"


def check_contradiction_check_not_overclaimed() -> tuple[str, str]:
    """Guards the same incident for the hard slot-value contradiction check:
    ledger.py's SessionState has no retired/contradiction machinery yet.
    Same precise-phrase approach as check_override_risk_not_overclaimed."""
    claim = re.compile(
        r"already built, already optimal\b|\bhard-triggers a state change\b",
        re.IGNORECASE,
    )
    matched, evidence = any_docs_match(
        ["features/ask-yield-ledger/one-pager.md", "project/design-strengths.md"],
        claim,
        skip_quoted_references=True,
    )
    ledger = read_local("starter/ledger.py")
    code_present = bool(re.search(r"\bretired\b|contradiction", ledger, re.IGNORECASE))
    if matched and not code_present:
        return "FAIL", f"docs claim the contradiction check is built, but starter/ledger.py has no such field. {evidence}"
    return "PASS", "no unbacked 'already built' claim about slot-value contradiction found"


def check_fixed_schedule_matches_docs() -> tuple[str, str]:
    """Extracts FIXED_SCHEDULE from this repo and flags if it no longer
    matches what the docs describe as 'the clean six' -- informational: this
    SHOULD eventually fire once issue #21's `budget` addition ships, as a
    reminder to update the docs description alongside it."""
    sched = read_local("starter/scheduler.py")
    m = re.search(r"FIXED_SCHEDULE\s*=\s*\(([^)]*)\)", sched)
    if not m:
        return "FAIL", "could not find FIXED_SCHEDULE in starter/scheduler.py"
    attrs = [a.strip().strip('"\'') for a in m.group(1).split(",") if a.strip()]
    canonical_six = ["material", "feature", "color", "style", "size", "use_case"]
    if attrs == canonical_six:
        return "PASS", f"FIXED_SCHEDULE is still the clean six: {attrs}"
    return "WARN", (
        f"FIXED_SCHEDULE is now {attrs}, no longer the clean six the docs describe — "
        "if this is issue #21's `budget` addition landing, update the docs repo's "
        "standing-findings.md / one-pager.md description to match"
    )


def check_phase_crosswalk() -> tuple[str, str]:
    """Confirms the docs repo's G-key note still describes Phase 2 as
    retrieval and Phase 3 as ask-yield -- guards a re-swap regression.

    Reads the `## Phases` TABLE, not `### Phase N — ` headings: the 1 Sep 2026
    README rewrite replaced those headings with a table and this check failed
    on the next push to main. The intent is unchanged -- Phase 2 must name
    Retrieval, Phase 3 must name Ask-yield.

    The search is deliberately scoped to the section rather than run over the
    whole file. The layer table earlier in the README opens a row with
    `| **2 — adaptive orchestration**`, which an unscoped pattern matches first
    and which would make this check fail for the wrong reason.
    """
    readme = read_local("README.md")
    section = readme.partition("\n## Phases")[2]
    if not section:
        return "FAIL", "could not find the '## Phases' section in README.md"
    p2 = re.search(r"\|\s*\*\*2 — ([^*|]+)\*\*", section)
    p3 = re.search(r"\|\s*\*\*3 — ([^*|]+)\*\*", section)
    if not (p2 and p3):
        return "FAIL", "could not find Phase 2 / Phase 3 rows in README.md's '## Phases' table"
    p2_name, p3_name = p2.group(1).strip(), p3.group(1).strip()
    if "Retrieval" not in p2_name or "Ask-yield" not in p3_name:
        return "FAIL", f"phase names drifted locally: Phase 2={p2_name!r}, Phase 3={p3_name!r}"
    claim = re.compile(
        r"techjam2026-pipeline.{1,3}s `Phase 2.{0,60}Retrieval",
        re.IGNORECASE | re.DOTALL,
    )
    matched, evidence = any_docs_match(["project/architecture-diagram-v3-notes.md"], claim)
    if not matched:
        return "WARN", "docs repo's G-key note doesn't confirm Phase 2 = Retrieval — check for drift or a rewrite"
    return "PASS", f"Phase 2={p2_name!r}, Phase 3={p3_name!r}, docs G-key note agrees"


def check_no_collision_regression() -> tuple[str, str]:
    """Guards against the P1-P5/Phase-1-5 token collision recurring: a bare
    active-looking P-tag (not inside a 'relabelled from'/'was' historical
    reference) anywhere in the docs repo, including plain prose like
    "superseded by the P1-P5 ..." -- not just the mermaid/bracket-tag style
    ("[P2 —", "(P2,") the original check was written against. Caught a real
    miss on the first run: README.md's version table still said "P1-P5" with
    no bracket adjacent to it at all."""
    danger = re.compile(r'["(\[]P[2-5][,\s—-]|\bP1[–-]P5\b')
    safe_context = re.compile(
        r"relabelled|was `P|was P1|before 28 Aug|superseded (28 Aug|early draft)|"
        r"Originally lettered|Kept for history|and phase tags|architecture-level P1",
        re.IGNORECASE,
    )
    files = fetch_docs("README.md")  # not scanned before; caught the miss above
    scan = {
        "README.md": files,
        "project/architecture-diagram.md": fetch_docs("project/architecture-diagram.md"),
        "project/architecture-diagram-v3-notes.md": fetch_docs("project/architecture-diagram-v3-notes.md"),
        "project/one-pager.md": fetch_docs("project/one-pager.md"),
        "project/open-questions.md": fetch_docs("project/open-questions.md"),
    }
    for path, text in scan.items():
        if text.startswith("__FETCH_ERROR_"):
            continue
        for m in danger.finditer(text):
            window = text[max(0, m.start() - 80): m.end() + 80]
            if not safe_context.search(window):
                return "FAIL", f"{path}: possible reintroduced bare P-tag near …{window.strip()}…"
    return "PASS", "no reintroduced bare P1-P5 tags found (README, architecture files, one-pager, open-questions)"


FACTS = [
    ("override-risk-not-overclaimed", check_override_risk_not_overclaimed),
    ("contradiction-check-not-overclaimed", check_contradiction_check_not_overclaimed),
    ("fixed-schedule-matches-docs", check_fixed_schedule_matches_docs),
    ("phase-crosswalk", check_phase_crosswalk),
    ("no-collision-regression", check_no_collision_regression),
]


def main() -> int:
    if not TOKEN:
        print(
            "TECHJAM_CROSS_REPO_TOKEN not set — docs repo is private, fetches will fail. "
            "See README.md 'Cross-repo sync checks' for setup.",
            file=sys.stderr,
        )
    results = []
    worst = "PASS"
    for fact_id, check in FACTS:
        try:
            status, detail = check()
        except Exception as exc:  # a checker bug must not silently pass
            status, detail = "FAIL", f"checker raised {exc!r}"
        results.append((fact_id, status, detail))
        if status == "FAIL":
            worst = "FAIL"
        elif status == "WARN" and worst != "FAIL":
            worst = "WARN"

    lines = ["", "## docs-sync check results", "", "| fact | status | detail |", "|---|---|---|"]
    for fact_id, status, detail in results:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
        lines.append(f"| `{fact_id}` | {icon} {status} | {detail} |")
    report = "\n".join(lines)
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
