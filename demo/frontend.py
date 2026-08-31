"""The chat CLI -- terminal 1. Runs the agent and emits the trace.

    python -m demo.frontend --bracket leaky --sample-id public_0042 --step

Scripted replay only: the conversation is driven by the evaluator's own
simulated customer over a session from ``data/public_set.jsonl``. Nobody types.
That is what makes the demo reproducible and what lets the backend explain a
turn that a scorer would actually have produced.

WHY ``--bracket`` HAS NO DEFAULT
--------------------------------
``public_set.jsonl`` carries no ``intent_card``, so the evaluator builds the
"hidden" customer preferences out of the TARGET PRODUCT'S OWN LISTING and
recites them back turn by turn. Locally that inflates everything. Every other
tool here defaults to ``leaky`` and that is fine for a labelled table; it is not
fine for a screen someone films. So the flag is required, the banner is driven
by what actually ran rather than by the flag, and every line carrying a number
carries its bracket tag.
"""
from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from pathlib import Path

from demo import ansi, driver, pacing, trace, tracer
from demo.pacing import say

TOP_K = 10
MAX_TURNS = 10

# The four cases. One session per scenario type, because the interesting thing
# about this agent is that it behaves DIFFERENTLY across them -- a single long
# session shows one behaviour and hides the other three.
#
# These ids are curated, not arbitrary: each was verified to hit while checking
# demo/driver.py against the vendored evaluator, and between them they span
# ranks 1/10/2/1 at turns 3/2/4/7. A missing id falls back to the first sample
# of its scenario, so a changed public_set.jsonl degrades instead of crashing.
CASES = (
    ("buying", "public_0001",
     "the straightforward path -- the customer states a requirement, and it is found"),
    ("browsing", "public_0006",
     "opens with no constraint at all; every constraint has to be asked for"),
    ("boundary", "public_0035",
     "the customer refuses once -- that ask is burned, but the bucket stays live"),
    ("intent_override", "public_0002",
     "the customer changes their mind; the hit check is OFF until the override lands"),
)


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------

def render_header(width: int, bracket: str, run_id: str, path: Path,
                  catalog: str, rows, degraded: bool, sample: dict, plan: dict) -> None:
    tag = bracket.upper()
    colour = ansi.YELLOW if bracket == trace.LEAKY else ansi.GREEN
    note = trace.BRACKET_NOTE[bracket]

    lines = [ansi.paint("BRACKET: " + tag, colour, ansi.BOLD)]
    for chunk in ansi.wrap(note, width - 6):
        lines.append(ansi.paint(chunk, colour))
    say("\n".join(ansi.box(
        lines, width, "SOL - conversational search - FRONTEND - run " + run_id)))

    say(" sample %s - scenario %s - category %s - difficulty %s" % (
        sample.get("sample_id"), sample.get("scenario_type"),
        sample.get("category_bucket"), sample.get("difficulty_bucket")))
    degraded_text = ansi.paint("True", ansi.RED, ansi.BOLD) if degraded else "False"
    say(" catalog %s (%s rows) - agent.degraded %s" % (catalog, rows, degraded_text))
    say(" trace   %s" % (path,))
    if plan.get("override"):
        say(" %s override lands at turn %s" % (
            ansi.paint("note", ansi.MAGENTA), plan["override"].get("turn")))
    say()


def render_case(width: int, index: int, total: int, sample: dict, note: str) -> None:
    """The banner that opens one case."""
    say()
    say(ansi.paint(ansi.titled_rule(
        "case %d/%d · %s" % (index, total, sample.get("scenario_type")),
        width, "="), ansi.BOLD, ansi.BLUE))
    if note:
        for line in ansi.wrap(note, width - 6, "    "):
            say(ansi.paint(line, ansi.DIM))
    say()


def render_turn(width: int, payload: dict, titles: dict, bracket: str,
                top: int = TOP_K) -> None:
    turn = payload["turn"]
    response = payload["response"]
    ranked = payload["ranked"]

    say(ansi.titled_rule("turn %d" % turn, width))
    _speaker("customer", payload["user_message"], width, ansi.CYAN)
    _speaker("agent", str(response.get("message", "")), width, ansi.BOLD)

    ask = response.get("ask_attribute")
    if ask:
        say(_label("asking") + ansi.paint(str(ask), ansi.YELLOW))

    target = payload["target"]
    top = max(1, int(top))
    shown = ranked[:top]
    label = ("top-%d of %d" % (len(shown), len(ranked)) if len(shown) < len(ranked)
             else "top-%d" % len(ranked))
    say(_label(label).rstrip())

    title_width = max(20, width - 22)
    for i, asin in enumerate(shown, start=1):
        say(_result_row(i, asin, titles, title_width, asin == target))

    # The target always gets a row, even below the cut. Without this exception
    # --top would hide the exact moment the demo exists to show.
    if target in ranked and target not in shown:
        say(ansi.paint("        ...", ansi.DIM))
        say(_result_row(ranked.index(target) + 1, target, titles, title_width, True))

    say(_label("status") + status_line(payload, bracket))
    say()


def _result_row(rank: int, asin: str, titles: dict, title_width: int,
                is_target: bool) -> str:
    marker = ansi.paint("  <- TARGET", ansi.GREEN, ansi.BOLD) if is_target else ""
    return "   %3d  %-12s %s%s" % (
        rank, asin, ansi.truncate(titles.get(asin, ""), title_width), marker)


LABEL_WIDTH = 9


def _label(text: str) -> str:
    """Every left-hand label in the transcript lines up at the same column."""
    return " " + text.ljust(LABEL_WIDTH) + " "


def _speaker(label: str, text: str, width: int, *codes: str) -> None:
    """One labelled, wrapped utterance; the label sits on the first line only."""
    prefix = _label(label)
    for i, line in enumerate(ansi.wrap(text, max(20, width - len(prefix)))):
        say("%s%s" % (prefix if i == 0 else " " * len(prefix),
                        ansi.paint(line, *codes)))


def status_line(payload: dict, bracket: str) -> str:
    tag = "[%s]" % bracket
    turn = "turn %d/%d" % (payload["turn"], MAX_TURNS)
    rank = payload.get("target_rank")

    if payload.get("hit_counted"):
        return "%s - %s - %s" % (
            ansi.paint("HIT at rank %d" % rank, ansi.GREEN, ansi.BOLD), turn, tag)

    if payload.get("hit_suppressed_by_override"):
        # The best moment in the whole system: the target is sitting right
        # there and the evaluator is not looking yet.
        return ("%s\n%s%s - %s - %s" % (
            ansi.paint("target present at rank %d - HIT CHECK IS OFF "
                       "(override has not landed)" % rank, ansi.MAGENTA, ansi.BOLD),
            " " * (LABEL_WIDTH + 2),
            ansi.paint("src/shown.py - these picks are NOT recorded as shown", ansi.DIM),
            turn, tag))

    if rank:
        return "%s - %s - %s" % (
            ansi.paint("target at rank %d" % rank, ansi.YELLOW), turn, tag)
    return "%s - %s - %s" % (
        ansi.paint("miss - target not in top-10", ansi.DIM), turn, tag)


def render_session_close(width: int, result: dict, titles: dict, bracket: str) -> None:
    """One case's verdict."""
    tag = "[%s]" % bracket
    say(ansi.titled_rule("case end", width))
    verdict = (ansi.paint("hit yes", ansi.GREEN, ansi.BOLD) if result["hit"]
               else ansi.paint("hit no", ansi.RED))
    say(" %s - first_hit_turn %s - best_rank %s - rr %.4f - %s" % (
        verdict, result["first_hit_turn"], result["best_rank"],
        result["reciprocal_rank"], tag))
    target = result["plan"]["target"]
    say(" target  %-12s %s" % (
        target, ansi.truncate(titles.get(target, result["plan"]["target_title"]),
                              max(20, width - 24))))
    say()


def render_run_summary(width: int, results: list, metrics: dict, bracket: str,
                       path: Path, size: int, records: int) -> None:
    """The four cases side by side, then the aggregate."""
    say()
    say(ansi.paint(ansi.titled_rule("summary", width, "="), ansi.BOLD))
    widths = [4, 14, 17, 7, 5, 6]
    say(ansi.paint(" " + ansi.columns(
        ["#", "sample", "scenario", "turns", "hit", "rank"], widths), ansi.DIM))
    for index, result in enumerate(results, start=1):
        hit = (ansi.paint("yes", ansi.GREEN, ansi.BOLD) if result["hit"]
               else ansi.paint("no", ansi.DIM))
        say(" " + ansi.columns([
            str(index), result["sample_id"], result["scenario_type"],
            str(result["turns_run"]), hit,
            str(result["best_rank"] or "-")], widths))
    say()
    # Never a bare number: bracket and n travel with every score in this repo.
    say(" score   0.50*%.4f + 0.30*%.4f + 0.20*%.4f = %s  [%s, n=%d]" % (
        metrics["hit_rate_at_10"], metrics["mrr"], metrics["efficiency"],
        ansi.paint("%.6f" % metrics["recommended_technical_score"], ansi.BOLD),
        bracket, metrics["sample_count"]))
    say(" trace   %s (%d bytes, %d records)" % (path, size, records))
    say()


# --------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------

def wants_cases(args) -> bool:
    """Cases mode is the default, unless the caller selected something explicitly.

    So the documented demo command shows the four scenarios, while
    ``--sample-id`` / ``--scenario`` / ``--sessions`` still mean what they
    always did.
    """
    if args.no_cases:
        return False
    if args.cases:
        return True
    explicit = (args.sample_id or args.scenario or args.difficulty
                or args.category or args.sessions is not None)
    return not explicit


def choose_cases(samples: list) -> list:
    """One session per scenario, in CASES order. Returns (sample, note) pairs."""
    by_id = {s.get("sample_id"): s for s in samples}
    chosen = []
    for scenario, sample_id, note in CASES:
        sample = by_id.get(sample_id)
        if sample is None or sample.get("scenario_type") != scenario:
            # The curated id is gone or has been re-typed; take any session of
            # this scenario rather than dropping the case entirely.
            sample = next((s for s in samples
                           if s.get("scenario_type") == scenario), None)
        if sample is not None:
            chosen.append((sample, note))
    if not chosen:
        raise SystemExit(
            "no session of any known scenario type in this dataset -- "
            "expected one of %s" % ([c[0] for c in CASES],))
    return chosen


def choose_samples(samples: list, args) -> list:
    pool = samples
    sessions = args.sessions if args.sessions is not None else 1
    if args.sample_id:
        pool = [s for s in pool if s.get("sample_id") == args.sample_id]
        if not pool:
            raise SystemExit("no sample with sample_id %r" % (args.sample_id,))
        return pool[:sessions]

    if args.scenario:
        pool = [s for s in pool if s.get("scenario_type") == args.scenario]
    if args.difficulty:
        pool = [s for s in pool if s.get("difficulty_bucket") == args.difficulty]
    if args.category:
        pool = [s for s in pool if s.get("category_bucket") == args.category]
    if not pool:
        raise SystemExit("no sample matches those filters")

    rng = random.Random(args.seed)
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:sessions]


def usable_samples(chosen: list, catalog_ids: set, width: int) -> list:
    """Drop sessions whose target is not in this catalog, and say so.

    ``materialize_hidden_fields`` builds the hidden card from the target's own
    listing (local_evaluator.py:208) and raises KeyError if the catalog does not
    hold it. That is exactly what happens when someone points the demo at
    ``tests/fixtures/catalog.jsonl``, and a bare traceback reads like a bug in
    the harness rather than a mismatched pair of files.
    """
    keep, missing = [], []
    for sample in chosen:
        target = str((sample.get("ground_truth") or {}).get("parent_asin", ""))
        (keep if target in catalog_ids else missing).append(sample)

    if missing:
        print(ansi.banner(
            "%d of %d sessions have a target that is not in this catalog"
            % (len(missing), len(chosen)), width, ansi.ON_YELLOW, ansi.BLACK))
        for sample in missing[:5]:
            print("   %s -> %s" % (
                sample.get("sample_id"),
                (sample.get("ground_truth") or {}).get("parent_asin")))
    if not keep:
        print()
        print(ansi.banner("NO USABLE SESSION -- catalog and dataset do not match",
                          width, ansi.ON_RED, ansi.BOLD))
        print("\n   The dataset's ground-truth products are not in the catalog you\n"
              "   pointed at. The simulated customer's hidden card is built from the\n"
              "   target's own listing, so there is nothing to build it from.\n\n"
              "   Use the pair that belong together:\n"
              "     --catalog data/catalog.jsonl --dataset data/public_set.jsonl\n")
        raise SystemExit(1)
    return keep


def build_titles(asins, products: dict) -> dict:
    """One asin->title map per turn. No title is repeated per row on the wire."""
    out = {}
    for asin in asins:
        if asin and asin not in out:
            out[asin] = str((products.get(asin) or {}).get("title") or "")
    return out


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m demo.frontend",
        description="Replay one scripted session against the real agent, "
                    "rendering the conversation and emitting a pipeline trace.")
    parser.add_argument("--bracket", choices=(trace.LEAKY, trace.SCRUBBED),
                        help="REQUIRED. leaky = upper bound, scrubbed = lower bound.")
    parser.add_argument("--sample-id")
    parser.add_argument("--scenario", choices=("buying", "browsing", "boundary",
                                               "intent_override"))
    parser.add_argument("--difficulty")
    parser.add_argument("--category")
    # default=None, not 1: wants_cases() needs to tell "asked for one session"
    # apart from "did not say", so the default can be cases mode.
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--cases", action="store_true",
                        help="one session per scenario type (the default)")
    parser.add_argument("--no-cases", action="store_true",
                        help="a single session, as selected by the other flags")
    parser.add_argument("--top", type=int, default=5,
                        help="recommendations to print per turn; the target is "
                             "always shown even below this cut (default 5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="pause between turns; the line reveal supplies most "
                             "of the pacing now (default 0.5)")
    parser.add_argument("--line-delay", type=float, default=0.045,
                        help="pause between output lines, so a turn unfolds "
                             "instead of landing whole; 0 disables (default 0.045)")
    parser.add_argument("--step", action="store_true",
                        help="wait for Enter before each turn (presenter control)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--trace-pool", type=int, default=25)
    parser.add_argument("--run-dir", default=str(trace.DEFAULT_RUN_DIR))
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args(argv)


BRACKET_HELP = """
--bracket is required. Pick the arm you are willing to be quoted on.

  --bracket leaky      UPPER BOUND. The vendored evaluator builds the simulated
                       customer's hidden card from the TARGET PRODUCT'S OWN
                       LISTING, so the customer recites text that is already
                       indexed. 94.5% of disclosed constraint strings are exact
                       substrings of the target's indexed text. This is the
                       organizer's own behaviour, and it is still an upper bound.

  --bracket scrubbed   LOWER BOUND. The card discloses only atomic attribute
                       values -- a material word, a colour word, a budget number
                       -- never a multi-word span lifted from the listing.

There is no default on purpose: a number on a screen without its bracket is the
one thing CLAUDE.md says never to produce.
"""


def guard_dataset(path: str) -> None:
    """Refuse anything under evaluation-data/ -- test-only, not for demos."""
    try:
        resolved = Path(path).resolve()
    except Exception:
        return
    if any(part == "evaluation-data" for part in resolved.parts):
        raise SystemExit(
            "refusing to read %s\n"
            "evaluation-data/ is test-only: do not open, sample, quote, or tune "
            "against it during development (CLAUDE.md). Use data/public_set.jsonl."
            % (resolved,))


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.bracket:
        print(BRACKET_HELP)
        return 2

    ansi.configure(no_color=args.no_color)
    pacing.configure(args.line_delay)
    width = ansi.terminal_width()
    guard_dataset(args.dataset)

    from evaluator import local_evaluator as vendored
    from evaluator.local_evaluator import catalog_index, load_jsonl
    from scripts.evaluate_src import bracket as bracket_ctx
    from scripts.evaluate_src import index_size, preflight_catalog, git_branch, git_commit

    if not preflight_catalog(args.catalog):
        return 1

    # Captured BEFORE bracket() can patch it, so the restore check in `finally`
    # is a real check rather than a tautology.
    pristine_intent_card = vendored.intent_card

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    if wants_cases(args):
        selected = choose_cases(samples)
    else:
        selected = [(s, "") for s in choose_samples(samples, args)]

    keep = {s.get("sample_id")
            for s in usable_samples([s for s, _ in selected], catalog_ids, width)}
    selected = [(s, n) for s, n in selected if s.get("sample_id") in keep]

    writer = trace.TraceWriter(run_dir=args.run_dir)
    active = tracer.Tracer()
    problems = active.install()
    if problems:
        writer.note("error", trace.PATCH_TARGET_MISSING,
                    "; ".join(str(p) for p in problems))
        writer.close()
        print(ansi.banner("TRACER CANNOT ATTACH TO src/pipeline.py", width,
                          ansi.ON_RED, ansi.BOLD))
        for problem in problems:
            print("  " + str(problem))
        print("\nThe demo observes the pipeline by wrapping its stage functions. "
              "One of them has changed shape, so the render would be wrong rather "
              "than merely missing. Fix demo/tracer.py:PATCH_TARGETS.")
        return 1

    warnings: list = []
    started = time.perf_counter()
    results: list = []
    try:
        agent = tracer.make_tracing_agent(args.catalog, active)
        rows = index_size(agent)
        degraded = bool(getattr(agent, "degraded", False))

        writer.emit(
            trace.RUN_OPEN,
            argv=list(sys.argv[1:]),
            bracket=args.bracket,
            bracket_note=trace.BRACKET_NOTE[args.bracket],
            catalog_path=args.catalog, catalog_rows=len(catalog_ids),
            index_rows=rows, agent_degraded=degraded,
            dataset_path=args.dataset, dataset_rows=len(samples),
            git_commit=git_commit(), git_branch=git_branch(),
            python=platform.python_version(), platform=sys.platform,
            tracer_version=trace.SCHEMA_VERSION,
            patch_targets=[{"module": t.module, "name": t.name,
                            "params": list(t.params), "stage": t.stage, "ok": True}
                           for t in tracer.PATCH_TARGETS],
            patch_targets_ok=True,
            constants=constants(),
            seams=seams(agent),
        )

        if degraded:
            writer.note("error", trace.DEGRADED_AGENT,
                        "agent.degraded is True -- the index did not build")
            warnings.append("agent degraded")
            if not args.allow_degraded:
                from scripts.evaluate_src import print_degraded_alarm
                print_degraded_alarm(args.catalog)
                return 1
        if rows is not None and rows < trace.SMALL_CATALOG_ROWS:
            text = ("SMALL CATALOG (%s rows) -- not representative. Any single-term "
                    "match returns nearly the whole file, so a query-blind ranker "
                    "looks perfect. This is the test_evaluator_smoke trap." % rows)
            writer.note("warn", trace.SMALL_CATALOG, text)
            warnings.append(text)
            print(ansi.banner(text, width, ansi.ON_YELLOW, ansi.BLACK))

        with bracket_ctx(args.bracket):
            for index, (sample, note) in enumerate(selected, start=1):
                results.append(run_one(
                    writer, active, agent, sample, catalog_ids, categories,
                    products, args, width, warnings,
                    case=(index, len(selected), note)))

        metrics = driver.summarise([
            {k: r[k] for k in ("sample_id", "scenario_type", "hit",
                               "first_hit_turn", "best_rank", "reciprocal_rank")}
            for r in results])
        if results:
            render_run_summary(width, results, metrics, args.bracket,
                               writer.path, writer.size_bytes(), writer.seq)
    finally:
        restored = active.restore()
        # evaluator/ is vendored and never edited; bracket() monkeypatches the
        # imported module object. If it were left installed, every later run in
        # this process would silently use the wrong customer.
        patch_left_installed = vendored.intent_card is not pristine_intent_card
        if patch_left_installed:
            writer.note("error", trace.EVALUATOR_PATCH_LEFT_INSTALLED,
                        "local_evaluator.intent_card was not restored")
            warnings.append("evaluator patch left installed")
        writer.emit(
            trace.RUN_CLOSE,
            bracket=args.bracket,
            sessions_run=len(results),
            **({k: metrics[k] for k in
                ("hit_rate_at_10", "mrr", "mttc", "efficiency",
                 "recommended_technical_score", "sample_count")}
               if results else {}),
            score_formula=trace.SCORE_FORMULA,
            wall_seconds=round(time.perf_counter() - started, 3),
            patch_restore_ok=bool(restored),
            tracer_record_errors=active.errors,
            degraded_plan_fired=active.degraded_plan_fired,
            warnings=warnings,
        )
        writer.close()
    return 0


def constants() -> dict:
    """Republish every constant the backend needs, so it never imports src/."""
    from src.pipeline import OVERRIDE_SUPPRESS_MAX_TURN
    from src.askpolicy import ASKABLE
    from src.types import (CARD_CAPACITY, DISCLOSURE_CAP, FIXED_SCHEDULE,
                           HEDGE_ORDER, MAX_QUERY_TERMS, MAX_TURNS as MT,
                           POOL_SIZE, RERANK_WINDOW)
    return {
        "POOL_SIZE": int(POOL_SIZE), "RERANK_WINDOW": int(RERANK_WINDOW),
        "MAX_TURNS": int(MT), "MAX_QUERY_TERMS": int(MAX_QUERY_TERMS),
        "DISCLOSURE_CAP": int(DISCLOSURE_CAP), "CARD_CAPACITY": int(CARD_CAPACITY),
        "OVERRIDE_SUPPRESS_MAX_TURN": int(OVERRIDE_SUPPRESS_MAX_TURN),
        "FIXED_SCHEDULE": list(FIXED_SCHEDULE), "HEDGE_ORDER": list(HEDGE_ORDER),
        "ASKABLE": list(ASKABLE),
    }


def seams(agent) -> dict:
    """Report every optional layer's flag. The demo never sets one."""
    from src import askyield, frames, llm_rerank, rerank, semantic
    reranker = getattr(getattr(agent, "_deps", None), "reranker", None)
    name = str(getattr(reranker, "name", "") or type(reranker).__name__)
    return {
        "reranker_name": name,
        "reranker_inert": "null" in name.lower(),
        "rerank_enabled": bool(getattr(rerank, "RERANK_ENABLED", False)),
        "tier2_enabled": bool(getattr(semantic, "TIER2_ENABLED", False)),
        "askyield_adaptive": bool(getattr(askyield, "ADAPTIVE_ENABLED", False)),
        "tier15_hedge": bool(getattr(frames, "TIER_15_HEDGE", False)),
        "llm_rerank_enabled": bool(getattr(llm_rerank, "LLM_RERANK_ENABLED", False)),
    }


def run_one(writer, active, agent, sample, catalog_ids, categories, products,
            args, width, warnings, case=(1, 1, "")) -> dict:
    """Drive one session, emitting session_open / turn... / session_close."""
    plan = driver.session_plan(sample, categories, products)
    card_source = trace.CARD_SOURCE[args.bracket]
    case_index, case_total, case_note = case

    writer.emit(
        trace.SESSION_OPEN,
        session_id=sample["sample_id"],
        sample_id=sample["sample_id"],
        # So the backend can print the same "case 2/4 - browsing" banner
        # without importing anything or knowing about CASES.
        case_index=case_index, case_total=case_total, case_note=case_note,
        scenario_type=sample.get("scenario_type"),
        category_bucket=sample.get("category_bucket"),
        difficulty_bucket=sample.get("difficulty_bucket"),
        user_profile=sample.get("user_profile"),
        coarse_category=plan["coarse_category"],
        opening_message=plan["opening_message"],
        ground_truth={
            "parent_asin": plan["target"], "title": plan["target_title"],
            "price": plan["target_price"], "visible_to_agent": False,
        },
        hidden_card={
            # The backend's banner reads THIS, never the CLI flag, so a
            # mislabelled bracket is structurally impossible.
            "source": card_source,
            "target_category": plan["card"].get("target_category"),
            "hard_constraints": plan["card"].get("hard_constraints"),
            "soft_preferences": plan["card"].get("soft_preferences"),
            "visible_to_agent": False,
        },
        override=plan["override"],
    )

    titles: dict = {}
    rendered = {"header": False}
    empty_pools = {"streak": 0}

    def on_turn(payload):
        recorder = agent.turns[-1] if agent.turns else None
        record = recorder.finalise(plan["target"], args.trace_pool) if recorder else {}

        asins = list(payload["ranked"])
        asins.append(payload["target"])
        for row in (record.get("retrieval") or {}).get("pool", []):
            asins.append(row.get("parent_asin"))
        turn_titles = build_titles(asins, products)
        titles.update(turn_titles)

        record["titles"] = turn_titles
        record.setdefault("input", {})
        record["input"]["evaluator_template"] = payload["evaluator_template"]
        record["input"]["evaluator_template_line"] = payload["evaluator_template_line"]
        record["outcome"] = {
            "target_parent_asin": payload["target"],
            "target_pool_rank": pool_rank(record, payload["target"]),
            "target_picks_rank": payload["target_rank"],
            "hit_counted": payload["hit_counted"],
            "hit_suppressed_by_override": payload["hit_suppressed_by_override"],
            "override_applied": payload["override_applied"],
            "visible_to_agent": False,
        }

        writer.emit(trace.TURN, session_id=sample["sample_id"],
                    sample_id=sample["sample_id"], turn=payload["turn"], **record)
        emit_turn_notes(writer, record, empty_pools, warnings)

        if not rendered["header"]:
            # The run header prints once, before the first case; each case then
            # gets its own banner.
            if case_index == 1:
                render_header(width, args.bracket, writer.run_id, writer.path,
                              args.catalog, len(catalog_ids),
                              bool(getattr(agent, "degraded", False)), sample, plan)
            render_case(width, case_index, case_total, sample, case_note)
            rendered["header"] = True
        render_turn(width, payload, titles, args.bracket, args.top)
        pace(args)

    result = driver.run_session(agent, sample, catalog_ids, categories, products,
                               on_turn=on_turn, session_id=sample["sample_id"])
    result["titles"] = titles

    writer.emit(
        trace.SESSION_CLOSE, session_id=sample["sample_id"],
        sample_id=result["sample_id"], hit=result["hit"],
        first_hit_turn=result["first_hit_turn"], best_rank=result["best_rank"],
        reciprocal_rank=result["reciprocal_rank"], turns_run=result["turns_run"],
        stop_reason=result["stop_reason"],
    )
    render_session_close(width, result, titles, args.bracket)
    return result


def pool_rank(record: dict, target: str):
    for row in (record.get("retrieval") or {}).get("pool", []):
        if row.get("parent_asin") == target:
            return row.get("rank")
    return None


def emit_turn_notes(writer, record: dict, empty_pools: dict, warnings: list) -> None:
    """Surface the three things that must never be missed by someone looking away."""
    ask = record.get("ask") or {}
    if ask.get("rung_agrees") is False:
        text = ("derived rung %r predicted %r but the policy returned %r -- "
                "demo/askrung.py has drifted from src/askpolicy.py::_select"
                % (ask.get("rung"), ask.get("rung_predicted_attribute"),
                   ask.get("policy_return")))
        writer.note("warn", trace.RUNG_MISMATCH, text)
        warnings.append(text)

    window = record.get("window") or {}
    if window.get("split_consistent") is False:
        text = ("the derived fresh[:RERANK_WINDOW] split disagrees with what "
                "_hydrate and _assemble were handed -- window numbers are suspect")
        writer.note("warn", trace.SPLIT_INCONSISTENT, text)
        warnings.append(text)
    if window.get("gate_guard_rejected") or window.get("rerank_guard_rejected"):
        writer.note("warn", trace.GUARD_REJECTED,
                    "a permutation guard discarded a stage's output")

    if (record.get("wire") or {}).get("degraded_plan_fired"):
        text = ("_degraded_plan fired -- run_turn's outer except caught something "
                "and THIS TRACE IS NOT THE SCORED AGENT")
        writer.note("error", trace.DEGRADED_PLAN_FIRED, text)
        warnings.append(text)

    if (record.get("tracer") or {}).get("record_errors"):
        writer.note("warn", trace.TRACER_RECORD_ERROR,
                    "%d recording steps failed this turn"
                    % record["tracer"]["record_errors"])

    if (record.get("retrieval") or {}).get("pool_size") == 0:
        empty_pools["streak"] += 1
        if empty_pools["streak"] >= 2:
            writer.note("warn", trace.POOL_EMPTY,
                        "the index has rows but MATCH returned nothing for "
                        "%d consecutive turns" % empty_pools["streak"])
    else:
        empty_pools["streak"] = 0


def pace(args) -> None:
    if args.step:
        try:
            input(ansi.paint("   [enter to continue] ", ansi.DIM))
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
    elif args.delay > 0:
        time.sleep(args.delay)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl-C during a paced render is a normal way to stop a demo, not a
        # crash. The finally in main() has already restored the patches.
        print("\ninterrupted.")
        raise SystemExit(130)
