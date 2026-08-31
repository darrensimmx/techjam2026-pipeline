"""The pipeline explainer -- terminal 2. Renders how each turn was produced.

    python -m demo.backend                     # tail the live run
    python -m demo.backend --replay <file>     # re-render a finished one

Reads the JSONL trace the frontend writes and renders ``src/pipeline.py``'s
nineteen stages as eight blocks. It NEVER IMPORTS ``src/``: every constant it
needs is republished in the ``run_open`` record and every enum is a plain
string, so this file keeps working if the pipeline is refactored underneath it
and can render a trace captured on another machine. A test enforces that by
rendering with ``src`` removed from ``sys.path``.

Two conventions carry through every block:

  ``[derived]``  the tracer COMPUTED this rather than observing it. Four values
                 are derived -- the FTS5 MATCH string, the ask-policy rung, the
                 window/rest split, and the overlap report. Two of those have a
                 cross-check, and a failed cross-check prints as a loud MISMATCH
                 rather than being quietly dropped.

  ``[leaky]`` /  which arm produced the numbers. Read from the trace's
  ``[scrubbed]`` ``hidden_card.source`` -- never from a CLI flag -- so the label
                 cannot disagree with what actually ran.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from demo import ansi, trace

BLOCKS = "ABCDEFGH"


class Renderer:
    """Holds run/session context and renders one turn at a time."""

    def __init__(self, width: int, only: str = "", raw: bool = False) -> None:
        self.width = width
        self.only = set(only.upper()) if only else set(BLOCKS)
        self.raw = raw
        self.run: dict = {}
        self.session: dict = {}
        self.constants: dict = {}
        self.bracket = "?"
        self.malformed = 0
        self.tracer_errors = 0
        self.last_seq = -1
        self.empty_pool_streak = 0

    # -- helpers ----------------------------------------------------------

    @property
    def narrow(self) -> bool:
        return self.width < ansi.NARROW_WIDTH

    def tag(self) -> str:
        return "[%s]" % self.bracket

    def title(self, titles: dict, asin: str, room: int) -> str:
        if self.narrow:
            return ""
        return ansi.truncate(str(titles.get(asin, "") or ""), room)

    def head(self, letter: str, name: str, stages: str) -> None:
        left = "%s  %s" % (ansi.paint(letter, ansi.BOLD, ansi.BLUE),
                           ansi.paint(name, ansi.BOLD))
        gap = self.width - ansi.visible_len(left) - len(stages)
        print("\n" + left + " " * max(1, gap) + ansi.paint(stages, ansi.DIM))

    def row(self, text: str) -> None:
        print("   " + text)

    # -- records ----------------------------------------------------------

    def on_run_open(self, record: dict) -> None:
        self.run = record
        self.constants = record.get("constants") or {}
        self.bracket = str(record.get("bracket", "?"))
        rows = record.get("index_rows")
        if isinstance(rows, int) and rows < trace.SMALL_CATALOG_ROWS:
            print(ansi.banner(
                "SMALL CATALOG (%d rows) -- not representative; any single-term "
                "match returns nearly everything" % rows,
                self.width, ansi.ON_YELLOW, ansi.BLACK))
        if record.get("agent_degraded"):
            print(ansi.banner(
                "AGENT DEGRADED -- the index did not build; every turn will be empty",
                self.width, ansi.ON_RED, ansi.BOLD))
        if not record.get("patch_targets_ok", True):
            print(ansi.banner("TRACER PATCH TARGETS DID NOT VERIFY",
                              self.width, ansi.ON_RED, ansi.BOLD))

    def on_session_open(self, record: dict) -> None:
        self.session = record
        card = record.get("hidden_card") or {}
        # The banner is driven by what actually ran, not by anyone's flag.
        source = str(card.get("source", ""))
        if "scrubbed" in source:
            self.bracket = trace.SCRUBBED
        elif "leaky" in source:
            self.bracket = trace.LEAKY
        print()
        print(ansi.titled_rule("session %s  (%s)" % (
            record.get("sample_id"), record.get("scenario_type")), self.width, "="))
        truth = record.get("ground_truth") or {}
        self.row("target   %-12s %s" % (
            truth.get("parent_asin"),
            ansi.truncate(str(truth.get("title") or ""), max(20, self.width - 26))))
        self.row("hidden card [%s]  hard=%s" % (
            self.bracket, (card.get("hard_constraints") or [])[:2]))
        if record.get("override"):
            self.row(ansi.paint("override lands at turn %s -- until then the "
                                "evaluator's hit check is OFF"
                                % (record["override"].get("turn"),), ansi.MAGENTA))

    def on_note(self, record: dict) -> None:
        level = str(record.get("level", "info"))
        codes = {"error": (ansi.ON_RED, ansi.BOLD),
                 "warn": (ansi.ON_YELLOW, ansi.BLACK)}.get(level, (ansi.DIM,))
        print(ansi.banner("%s: %s" % (record.get("code"), record.get("text")),
                          self.width, *codes))

    def on_session_close(self, record: dict) -> None:
        verdict = (ansi.paint("HIT", ansi.GREEN, ansi.BOLD) if record.get("hit")
                   else ansi.paint("MISS", ansi.RED))
        print("\n" + ansi.titled_rule("session end", self.width, "="))
        self.row("%s  first_hit_turn %s  best_rank %s  rr %.4f  turns %s  (%s)  %s" % (
            verdict, record.get("first_hit_turn"), record.get("best_rank"),
            record.get("reciprocal_rank") or 0.0, record.get("turns_run"),
            record.get("stop_reason"), self.tag()))

    def on_run_close(self, record: dict) -> None:
        print("\n" + ansi.titled_rule("run end", self.width, "="))
        if record.get("sample_count"):
            self.row("score  %s  %s n=%s   %s" % (
                ansi.paint("%.6f" % (record.get("recommended_technical_score") or 0.0),
                           ansi.BOLD),
                self.tag(), record.get("sample_count"), record.get("score_formula")))
        self.row("patch_restore_ok %s   tracer_record_errors %s   wall %ss" % (
            record.get("patch_restore_ok"), record.get("tracer_record_errors"),
            record.get("wall_seconds")))
        for warning in record.get("warnings") or []:
            print(ansi.banner("warning: " + str(warning), self.width,
                              ansi.ON_YELLOW, ansi.BLACK))

    # -- the turn ---------------------------------------------------------

    def on_turn(self, record: dict) -> None:
        if self.raw:
            print(json.dumps(record, indent=2)[:20000])
            return

        titles = record.get("titles") or {}
        self.print_turn_header(record)
        if "A" in self.only:
            self.block_a(record)
        if "B" in self.only:
            self.block_b(record)
        if "C" in self.only:
            self.block_c(record, titles)
        if "D" in self.only:
            self.block_d(record)
        if "E" in self.only:
            self.block_e(record)
        if "F" in self.only:
            self.block_f(record, titles)
        if "G" in self.only:
            self.block_g(record)
        if "H" in self.only:
            self.block_h(record, titles)
        print()

    def print_turn_header(self, record: dict) -> None:
        seams = self.run.get("seams") or {}
        colour = ansi.YELLOW if self.bracket == trace.LEAKY else ansi.GREEN
        lines = [
            "%s  index %s rows  seams: rerank=%s%s tier2=%s askyield=%s t1.5=%s" % (
                ansi.paint("BRACKET: " + self.bracket.upper(), colour, ansi.BOLD),
                self.run.get("index_rows"),
                seams.get("reranker_name", "?"),
                "(INERT)" if seams.get("reranker_inert") else "",
                "ON" if seams.get("tier2_enabled") else "OFF",
                "ON" if seams.get("askyield_adaptive") else "OFF",
                "ON" if seams.get("tier15_hedge") else "OFF"),
            "malformed lines %d - tracer errors %d%s" % (
                self.malformed, self.tracer_errors,
                "   " + ansi.paint("NARROW MODE: titles hidden", ansi.DIM)
                if self.narrow else ""),
        ]
        print("\n".join(ansi.box(lines, self.width, "BACKEND - pipeline - %s - turn %s/%s" % (
            record.get("sample_id"), record.get("turn"),
            self.constants.get("MAX_TURNS", 10)))))

    # A ---------------------------------------------------------------

    def block_a(self, record: dict) -> None:
        self.head("A", "INPUT & DECODE", "stages 1-5")
        data = record.get("input") or {}
        decode = record.get("decode") or {}
        tier1 = decode.get("tier1") or {}
        state = record.get("state") or {}

        for i, line in enumerate(ansi.wrap('"%s"' % data.get("user_message", ""),
                                           self.width - 12)):
            self.row("%-9s %s" % ("in" if i == 0 else "", ansi.paint(line, ansi.CYAN)))
        self.row("%-9s %s  %s" % (
            "", ansi.paint("template " + str(data.get("evaluator_template")), ansi.DIM),
            ansi.paint(str(data.get("evaluator_template_line") or ""), ansi.DIM)))
        self.row("1 normalise  %s chars, whitespace collapsed" % data.get("normalised_chars"))
        self.row("2 turn       %s -> clamped %s      top_k %s -> limit %s" % (
            data.get("turn_in"), data.get("turn_clamped"),
            data.get("top_k_in"), data.get("limit")))
        self.row("3 tier1      frame=%s  source=%s  decline=%s  signal=%s" % (
            ansi.paint(str(tier1.get("frame")), ansi.BOLD), tier1.get("source"),
            tier1.get("decline"), tier1.get("scenario_signal")))
        segments = tier1.get("segments") or []
        self.row("             segments(%d) = %s" % (
            len(segments), ansi.truncate(str(segments), self.width - 30)))
        self.row("4 tier2      %s        tier1.5 hedge: %s" % (
            "RAN" if decode.get("tier2_ran") else "NOT RUN (%s)" % decode.get("tier2_reason"),
            "FIRED" if decode.get("tier15_hedge_fired") else "not fired"))
        scenario = ("stays %r" % state.get("scenario_after")
                    if state.get("scenario_before") == state.get("scenario_after")
                    else "%r -> %r" % (state.get("scenario_before"),
                                       state.get("scenario_after")))
        self.row("5 note       frame_counts %s   scenario %s" % (
            state.get("frame_counts"), scenario))

    # B ---------------------------------------------------------------

    def block_b(self, record: dict) -> None:
        self.head("B", "STATE UPDATE", "stages 6-9")
        state = record.get("state") or {}
        guard = state.get("override_guard") or {}
        ledger = state.get("ledger") or {}
        slots = state.get("slots") or {}
        book = state.get("ask_bookkeeping") or {}

        self.row("6 override guard  action=%s %s  suppressed %s->%s  applied %s->%s  shown %s->%s" % (
            ansi.paint(str(guard.get("action")), ansi.BOLD), self.derived(),
            _tf(guard.get("suppressed_before")), _tf(guard.get("suppressed_after")),
            _tf(guard.get("override_applied_before")), _tf(guard.get("override_applied_after")),
            guard.get("shown_before"), guard.get("shown_after")))
        self.row("7 ledger          entries %s -> %s   distinct segments %s" % (
            ledger.get("entries_before"), ledger.get("entries_after"),
            ledger.get("distinct_segment_count")))
        appended = ledger.get("appended")
        if appended:
            for i, line in enumerate(ansi.wrap('+ "%s"' % appended, self.width - 22)):
                self.row("                  " + ansi.paint(line, ansi.GREEN if i == 0 else ansi.DIM))
        else:
            self.row("                  " + ansi.paint(
                "(nothing appended -- content-free frame)", ansi.DIM))

        filled = slots.get("filled") or []
        if filled:
            for item in filled:
                self.row('8 slots           fill %s="%s"%s' % (
                    item.get("attribute"),
                    ansi.truncate(str(item.get("value")), 46),
                    "  (replaced %r)" % item["replaced"] if item.get("replaced") else ""))
        else:
            self.row("8 slots           no change")
        if slots.get("cleared_by_override"):
            self.row("                  " + ansi.paint(
                "CLEARED by override: %s" % slots["cleared_by_override"], ansi.MAGENTA))
        self.row("                  after %s" % ansi.truncate(
            str(slots.get("after")), self.width - 26))

        before = book.get("before") or {}
        after = book.get("after") or {}
        self.row("9 ask bookkeep    %s  target=%r segments=%s decline=%s %s" % (
            ansi.paint(str(book.get("branch")), ansi.BOLD), book.get("target"),
            book.get("segment_count"), book.get("decline"), self.derived()))
        newly = sorted(set(after.get("retired") or ()) - set(before.get("retired") or ()))
        if newly:
            self.row("                  " + ansi.paint(
                "RETIRED %s -- never asked again" % newly, ansi.RED))
        self.row("                  disclosed_count %s -> %s   yield %s" % (
            before.get("disclosed_count"), after.get("disclosed_count"),
            after.get("yield_counts")))

    # C ---------------------------------------------------------------

    def block_c(self, record: dict, titles: dict) -> None:
        self.head("C", "QUERY & POOL", "stages 10-12")
        retrieval = record.get("retrieval") or {}
        partition = record.get("partition") or {}

        self.row("10 query      source=%s   %s chars   %s" % (
            ansi.paint(str(retrieval.get("query_source", "")).upper(), ansi.BOLD),
            retrieval.get("query_chars"), self.derived()))
        for i, line in enumerate(ansi.wrap(str(retrieval.get("query", "")),
                                           self.width - 18)):
            self.row("   %-10s %s" % ("" if i else "", ansi.paint(line, ansi.DIM)))

        expression = retrieval.get("match_expression")
        if expression:
            self.row("   FTS5 MATCH " + self.derived())
            for line in ansi.wrap(expression, self.width - 18)[:4]:
                self.row("   " + ansi.paint(line, ansi.BLUE))
            self.row("   %s of <=%s terms, %s" % (
                retrieval.get("match_terms"), retrieval.get("match_term_cap"),
                ansi.paint("CAPPED", ansi.YELLOW) if retrieval.get("match_capped")
                else "not capped"))

        pool = retrieval.get("pool") or []
        size = retrieval.get("pool_size")
        if not size:
            print(ansi.banner(
                "POOL EMPTY -- the query matched nothing%s" % (
                    ": " + str(retrieval.get("search_skipped_reason"))
                    if retrieval.get("search_skipped_reason") else ""),
                self.width, ansi.ON_RED, ansi.BOLD))
        self.row("11 pool       index.search(query, depth %s) -> %s      %s" % (
            retrieval.get("pool_depth_requested"), size,
            ansi.paint(str(retrieval.get("score_note", "")), ansi.DIM)))
        shown = pool[:12]
        self.table(shown, titles, "bm25")
        if len(shown) < (size or 0):
            self.row("   %s" % ansi.paint(
                "showing %d of %d serialised, out of a pool of %d "
                "(--trace-pool; * = kept past the cap because it is a pick or the target)"
                % (len(shown), len(pool), size), ansi.DIM))

        ok = partition.get("is_true_partition")
        self.row("12 partition  fresh %s / seen %s      true partition: %s" % (
            partition.get("fresh_count"), partition.get("seen_count"),
            ansi.paint("yes", ansi.GREEN) if ok else ansi.paint("NO", ansi.RED, ansi.BOLD)))
        self.row("              " + ansi.paint(
            "seen = shown earlier this session; reordered, never dropped", ansi.DIM))

    def table(self, rows: list, titles: dict, score_label: str) -> None:
        if not rows:
            return
        room = max(18, self.width - 42)
        self.row(ansi.paint("    #   parent_asin    %-9s %s" % (
            score_label, "" if self.narrow else "title"), ansi.DIM))
        for row in rows:
            asin = row.get("parent_asin", "")
            marker = ""
            if row.get("is_target"):
                marker = ansi.paint("  <- TARGET", ansi.GREEN, ansi.BOLD)
            beyond = ansi.paint(" *", ansi.YELLOW) if row.get("beyond_trace_cap") else ""
            self.row("  %3s   %-12s %9.3f %s%s%s" % (
                row.get("rank"), asin, row.get("score") or 0.0,
                self.title(titles, asin, room), marker, beyond))

    # D ---------------------------------------------------------------

    def block_d(self, record: dict) -> None:
        self.head("D", "WINDOW  order-only, permutation-checked", "stages 13-15")
        window = record.get("window") or {}

        consistent = window.get("split_consistent")
        self.row("split      fresh[:%s] -> window %s  |  fresh[%s:] -> rest %s   %s" % (
            window.get("rerank_window"), window.get("window_size"),
            window.get("rerank_window"), window.get("rest_size"), self.derived()))
        if consistent:
            self.row("           cross-check vs _hydrate arg / _assemble arg: %s"
                     % ansi.paint("OK", ansi.GREEN))
        else:
            print(ansi.banner(
                "SPLIT MISMATCH -- the derived slice disagrees with what the "
                "pipeline was handed; every window number below is suspect",
                self.width, ansi.ON_RED, ansi.BOLD))

        self.row("13 hydrate   %s/%s texts filled        order %s" % (
            window.get("hydrated_count"), window.get("window_size"),
            _changed(window.get("hydrate_changed_order"))))
        self.row("14 rerank    %s  %s -> order %s%s" % (
            window.get("rerank_name"),
            ansi.paint("DECLARED INERT", ansi.DIM) if window.get("rerank_declared_inert")
            else ansi.paint("ACTIVE", ansi.YELLOW),
            _changed(window.get("rerank_changed_order")),
            ansi.paint("   GUARD REJECTED IT", ansi.RED, ansi.BOLD)
            if window.get("rerank_guard_rejected") else ""))
        self.row("15 gate      overlap.gate over %s ledger segments -> order %s, %s moved%s" % (
            len(window.get("gate_segments") or []),
            _changed(window.get("gate_changed_order")),
            window.get("gate_positions_moved"),
            ansi.paint("   GUARD REJECTED IT", ansi.RED, ansi.BOLD)
            if window.get("gate_guard_rejected") else ""))
        for mover in (window.get("gate_movers") or [])[:4]:
            direction = ansi.GREEN if mover["delta"] > 0 else ansi.DIM
            self.row("           %-12s #%-3s -> #%-3s %s" % (
                mover["parent_asin"], mover["from"], mover["to"],
                ansi.paint("(up %d)" % mover["delta"] if mover["delta"] > 0
                           else "(down %d)" % -mover["delta"], direction)))

    # E ---------------------------------------------------------------

    def block_e(self, record: dict) -> None:
        report = record.get("overlap_report")
        self.head("E", "INSTRUMENT  overlap.measure() -- nothing filters on it", "observation")
        if not report:
            self.row(ansi.paint("not measured this turn", ansi.DIM))
            return
        self.row("segments %s   matched %s   rate %.3f   top_overlap %s   %s" % (
            report.get("segments"), report.get("matched"), report.get("rate") or 0.0,
            report.get("top_overlap"), self.derived()))
        self.row(ansi.paint("measured on the post-gate window via %s"
                            % report.get("function"), ansi.DIM))
        if self.bracket == trace.LEAKY and (report.get("rate") or 0) >= 0.9:
            # The leak shown as a measurement rather than a disclaimer.
            self.row(ansi.paint(
                "! rate %.3f under a LEAKY card: every disclosed string is "
                "literally in the pool text." % report["rate"],
                ansi.YELLOW, ansi.BOLD))

    # F ---------------------------------------------------------------

    def block_f(self, record: dict, titles: dict) -> None:
        self.head("F", "PICKS", "stages 16-17")
        picks = record.get("picks") or {}
        asins = picks.get("parent_asins") or []
        provenance = picks.get("provenance") or []
        scores = picks.get("scores") or []
        target = (record.get("outcome") or {}).get("target_parent_asin")

        self.row(ansi.paint(
            "assemble  window + rest-of-fresh + seen -> first %s unique. "
            "Never short, never duplicated." % picks.get("limit"), ansi.DIM))
        room = max(18, self.width - 52)
        self.row(ansi.paint("   rank  parent_asin    from        bm25      %s"
                            % ("" if self.narrow else "title"), ansi.DIM))
        for i, asin in enumerate(asins):
            score = scores[i] if i < len(scores) else None
            marker = ansi.paint("  <- TARGET", ansi.GREEN, ansi.BOLD) if asin == target else ""
            self.row("   %3d   %-12s %-10s %9s %s%s" % (
                i + 1, asin, provenance[i] if i < len(provenance) else "?",
                "%.3f" % score if isinstance(score, float) else "-",
                self.title(titles, asin, room), marker))
        self.row("provenance  window %s / rest %s / seen %s" % (
            picks.get("from_window"), picks.get("from_rest"), picks.get("from_seen")))
        self.row("17 record   shown %s -> %s   recorded %s   suppressed=%s%s" % (
            picks.get("shown_before"), picks.get("shown_after"), picks.get("recorded"),
            picks.get("record_suppressed"),
            ansi.paint("  <- nothing recorded: override not yet landed", ansi.MAGENTA)
            if picks.get("record_suppressed") else ""))

    # G ---------------------------------------------------------------

    def block_g(self, record: dict) -> None:
        self.head("G", "ASK", "stages 18-19")
        ask = record.get("ask") or {}
        before = ask.get("state_before") or {}

        self.row("before   turn=%s  asked=%s  retired=%s  burned=%r" % (
            before.get("turn"), before.get("asked"),
            before.get("retired"), before.get("burned")))
        self.row("         yield=%s  last_ask=%r  disclosed=%s" % (
            before.get("yield_counts"), before.get("last_ask"),
            before.get("disclosed_count")))
        self.row("rung     %s   %s" % (
            ansi.paint(str(ask.get("rung")), ansi.BOLD), self.derived()))
        for line in ansi.wrap(str(ask.get("rung_reason", "")), self.width - 14):
            self.row("         " + ansi.paint(line, ansi.DIM))
        self.row("policy   askyield.next_attribute -> %r   (adaptive %s)" % (
            ask.get("policy_return"), "ON" if ask.get("adaptive_enabled") else "OFF"))

        if ask.get("rung_agrees"):
            self.row("check    derived %r == policy %r   %s" % (
                ask.get("rung_predicted_attribute"), ask.get("policy_return"),
                ansi.paint("OK", ansi.GREEN)))
        else:
            print(ansi.banner("RUNG MISMATCH -- the label is a DERIVATION and it is "
                              "wrong here", self.width, ansi.ON_RED, ansi.BOLD))
            self.row("         derived %r != policy %r. The agent used %r. "
                     "Trust the policy row." % (
                         ask.get("rung_predicted_attribute"), ask.get("policy_return"),
                         ask.get("final")))
            self.row(ansi.paint("         demo/askrung.py has drifted from "
                                "src/askpolicy.py::_select -- file it.", ansi.DIM))
        if ask.get("fallback_fired"):
            self.row(ansi.paint(
                "fallback _fallback_attribute FIRED -- _valid_ask rejected the "
                "policy's choice", ansi.YELLOW, ansi.BOLD))
        self.row("19 msg   %s" % ansi.truncate(str(ask.get("message", "")), self.width - 12))

    # H ---------------------------------------------------------------

    def block_h(self, record: dict, titles: dict) -> None:
        self.head("H", "GROUND TRUTH  (never visible to the agent)", "evaluator side")
        outcome = record.get("outcome") or {}
        timing = record.get("timing") or {}
        wire = record.get("wire") or {}
        target = outcome.get("target_parent_asin")

        self.row("target      %-12s %s" % (
            target, self.title(titles, target, max(18, self.width - 28))))
        self.row("pool rank   %s of %s        picks rank  %s of %s" % (
            outcome.get("target_pool_rank") or "-",
            (record.get("retrieval") or {}).get("pool_size"),
            outcome.get("target_picks_rank") or "-",
            wire.get("recommendation_count")))
        if outcome.get("hit_counted"):
            self.row("hit         %s -> the session ends here   %s" % (
                ansi.paint("COUNTED", ansi.GREEN, ansi.BOLD), self.tag()))
        elif outcome.get("hit_suppressed_by_override"):
            self.row("hit         %s" % ansi.paint(
                "SUPPRESSED -- target is at rank %s but override has not landed, "
                "so the evaluator is not looking" % outcome.get("target_picks_rank"),
                ansi.MAGENTA, ansi.BOLD))
        else:
            self.row("hit         %s   %s" % (ansi.paint("no", ansi.DIM), self.tag()))
        if wire.get("degraded_plan_fired"):
            print(ansi.banner(
                "_degraded_plan FIRED -- run_turn's outer except caught something. "
                "THIS TRACE IS NOT THE SCORED AGENT.",
                self.width, ansi.ON_RED, ansi.BOLD))
        self.row(ansi.paint(
            "timing      turn %sms  search %s  hydrate %s  rerank %s  gate %s   "
            "(recording excluded)" % (
                timing.get("turn_ms"), timing.get("search"), timing.get("hydrate"),
                timing.get("rerank"), timing.get("gate")), ansi.DIM))

    def derived(self) -> str:
        return ansi.paint("[derived]", ansi.DIM)


def _tf(value) -> str:
    return "T" if value else "F"


def _changed(value) -> str:
    return (ansi.paint("CHANGED", ansi.YELLOW) if value
            else ansi.paint("UNCHANGED", ansi.DIM))


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m demo.backend",
        description="Render how each turn's output was produced, from a trace "
                    "written by demo.frontend.")
    parser.add_argument("--run", help="explicit trace file")
    parser.add_argument("--replay", help="re-render a finished run and exit")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--step", action="store_true",
                        help="pause after each turn (safe -- the file buffers)")
    parser.add_argument("--only", default="",
                        help="render a subset of blocks, e.g. --only A,C,F,G")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--run-dir", default=str(trace.DEFAULT_RUN_DIR))
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="dump each turn record as JSON instead of rendering")
    parser.add_argument("--wait", type=float, default=0.0,
                        help="give up after N seconds with no new records (0 = never)")
    return parser.parse_args(argv)


def dispatch(renderer: Renderer, record: dict, args) -> None:
    kind = record.get("type")
    seq = record.get("seq")
    if isinstance(seq, int):
        renderer.last_seq = seq
    if kind == trace.RUN_OPEN:
        renderer.on_run_open(record)
    elif kind == trace.SESSION_OPEN:
        renderer.on_session_open(record)
    elif kind == trace.TURN:
        renderer.tracer_errors += (record.get("tracer") or {}).get("record_errors", 0)
        renderer.on_turn(record)
        if args.step:
            try:
                input(ansi.paint("   [enter for the next turn] ", ansi.DIM))
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(0)
        elif args.replay and args.speed > 0:
            time.sleep(min(2.0, 0.6 / args.speed))
    elif kind == trace.SESSION_CLOSE:
        renderer.on_session_close(record)
    elif kind == trace.RUN_CLOSE:
        renderer.on_run_close(record)
    elif kind == trace.NOTE:
        renderer.on_note(record)


def main(argv=None) -> int:
    args = parse_args(argv)
    ansi.configure(no_color=args.no_color)
    width = args.width or ansi.terminal_width()

    path = trace.discover_run(args.run_dir, args.replay or args.run)
    if path is None and args.replay:
        print("no such trace: %s" % args.replay)
        return 1

    renderer = Renderer(width, args.only.replace(",", ""), args.raw)

    if path is None:
        print(ansi.paint(
            "waiting for a run in %s ...\n"
            "start the conversation with:  python -m demo.frontend --bracket leaky"
            % args.run_dir, ansi.DIM))
        deadline = time.monotonic() + (args.wait or 0)
        while path is None:
            if args.wait and time.monotonic() > deadline:
                print("gave up waiting.")
                return 1
            time.sleep(0.2)
            path = trace.discover_run(args.run_dir, None)

    reader = trace.TraceReader(path)
    print(ansi.paint("reading %s" % path, ansi.DIM))

    try:
        if args.replay:
            for record in reader.read_all():
                renderer.malformed = reader.malformed
                dispatch(renderer, record, args)
        else:
            for record in reader.follow(idle_timeout=args.wait):
                renderer.malformed = reader.malformed
                dispatch(renderer, record, args)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130
    if reader.malformed:
        print(ansi.banner("%d malformed line(s) were skipped" % reader.malformed,
                          width, ansi.ON_YELLOW, ansi.BLACK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
