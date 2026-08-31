"""Observe ``src/pipeline.py``'s nineteen stages without editing ``src/``.

WHY MONKEYPATCH AND NOT A SEAM
------------------------------
``_run_turn`` keeps every intermediate in a local and returns only a
``TurnPlan``, so there is nothing to read from the outside. The alternative --
threading a tracer through ``Deps`` and emitting at each stage -- would add
nineteen call sites to the file whose docstring reads "Every step below runs on
every turn and is FORBIDDEN TO RAISE", for a feature that never runs on the
graded path. The four seams that already exist (``rerank``, ``semantic``,
``llm_rerank``, ``askyield``) are *capability* seams, each a submission-level
decision with a disclosure attached; an observation seam is not one of those.

Patching works because the ``_xxx`` helpers are called as bare globals and
resolve through the module dict at call time. It reaches every stage, including
the three that look lossy, because each stage RECEIVES the previous stage's
output as an argument:

  - the pre-Tier-2 decode IS ``_tier1``'s return (pipeline.py:126);
  - the inline split at :137 is reconstructed from ``_partition``'s return and
    the live ``RERANK_WINDOW``, then cross-checked against ``_hydrate``'s
    argument and ``_assemble``'s -- disagreement sets ``split_consistent``;
  - the three orderings rebound onto ``window`` at :138-140 are each visible
    from two independent points (a return, and the next stage's argument).

THE ONE RULE THAT MATTERS
-------------------------
If a recorder raises inside a stage, ``run_turn``'s outer except (:105-109)
swallows it into ``_degraded_plan()`` and the demo silently shows a DIFFERENT
agent than the one being scored. So: the original is called first and its
result returned unconditionally, and every byte of recording sits in its own
try/except that counts the failure instead of propagating it. The one thing
that must happen before the original call -- snapshotting mutable state so a
before/after diff is possible -- is itself wrapped, for the same reason.

``_degraded_plan`` is patched as a canary so that if this ever does go wrong,
the trace says so out loud instead of quietly rendering fiction.
"""
from __future__ import annotations

import importlib
import inspect
import time
from contextlib import contextmanager

from demo import askrung

# --------------------------------------------------------------------------
# What we patch.
# --------------------------------------------------------------------------


class Target:
    """One patch site, with the signature we expect to find there.

    ``params`` is checked by NAME AND ORDER, not merely for existence:
    ``_assemble(window, rest, seen, limit)`` reordered to
    ``(seen, window, rest, limit)`` would sail through an existence check while
    silently inverting every provenance label in the render.
    """

    __slots__ = ("module", "name", "params", "stage", "note", "handler")

    def __init__(self, module: str, name: str, params: tuple, stage: str,
                 note: str = "", handler: str = "") -> None:
        self.module = module
        self.name = name
        self.params = params
        self.stage = stage
        self.note = note
        # Handlers are looked up by this name, not derived from `name`, because
        # src.overlap.gate and src.pipeline._gate would otherwise collide.
        self.handler = handler or name.lstrip("_")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Target(%s.%s)" % (self.module, self.name)


PATCH_TARGETS = (
    Target("src.pipeline", "_normalise", ("user_message",), "1"),
    Target("src.pipeline", "_clamp_turn", ("turn", "session"), "2"),
    Target("src.pipeline", "clamp_top_k", ("top_k",), "2b"),
    Target("src.pipeline", "_tier1", ("text",), "3"),
    Target("src.pipeline", "_tier2", ("decode", "text", "deps"), "4"),
    Target("src.pipeline", "_note", ("session", "decode"), "5"),
    Target("src.pipeline", "_override_guard", ("session", "decode", "turn_number"), "6"),
    Target("src.pipeline", "_append_to_ledger", ("session", "decode"), "7"),
    Target("src.pipeline", "_fill_slots", ("session", "decode"), "8"),
    Target("src.pipeline", "_ask_bookkeeping", ("session", "decode", "turn_number"), "9"),
    Target("src.pipeline", "_query_for", ("session", "text"), "10"),
    Target("src.pipeline", "_search", ("deps", "query", "limit"), "11"),
    Target("src.pipeline", "_partition", ("session", "pool"), "12"),
    Target("src.pipeline", "_hydrate", ("deps", "window"), "13"),
    Target("src.pipeline", "_rerank", ("deps", "query", "window"), "14"),
    Target("src.pipeline", "_gate", ("session", "decode", "window"), "15"),
    Target("src.pipeline", "_assemble", ("window", "rest", "seen", "limit"), "16"),
    Target("src.pipeline", "_record", ("session", "picks"), "17"),
    Target("src.pipeline", "_choose_attribute", ("session", "turn_number"), "18"),
    Target("src.askyield", "next_attribute", ("state",), "18a"),
    Target("src.pipeline", "_fallback_attribute", ("session", "turn_number"), "18b"),
    Target("src.pipeline", "_message_for", ("attribute",), "19"),
    # Canaries. None of these should ever fire; each is silent if it does.
    Target("src.pipeline", "_degraded_plan", ("session",), "canary",
           "run_turn's outer except fired -- the trace is not the scored agent"),
    Target("src.overlap", "gate", ("candidates", "segments"), "canary",
           "raw gate output, to detect the permutation guard rejecting stage 15",
           handler="overlap_gate"),
    Target("src.rerank", "safe_rerank", ("reranker", "query", "candidates"), "canary",
           "raw rerank output, to detect the permutation guard rejecting stage 14"),
)


class Problem:
    __slots__ = ("target", "detail")

    def __init__(self, target: Target, detail: str) -> None:
        self.target = target
        self.detail = detail

    def __str__(self) -> str:
        return "%s.%s: %s" % (self.target.module, self.target.name, self.detail)


def verify_targets() -> list:
    """Check every patch site resolves with the signature we expect.

    This is what converts refactor brittleness from a silent failure -- a demo
    that renders confident nonsense -- into a loud one. Run at startup and in
    CI.
    """
    problems: list = []
    for target in PATCH_TARGETS:
        try:
            module = importlib.import_module(target.module)
        except Exception as exc:
            problems.append(Problem(target, "module import failed: %r" % (exc,)))
            continue
        function = getattr(module, target.name, None)
        if function is None:
            problems.append(Problem(target, "attribute is missing"))
            continue
        if not callable(function):
            problems.append(Problem(target, "attribute is not callable"))
            continue
        try:
            found = tuple(inspect.signature(function).parameters)
        except Exception as exc:
            problems.append(Problem(target, "signature unreadable: %r" % (exc,)))
            continue
        if found != target.params:
            problems.append(Problem(
                target, "signature changed: expected %s, found %s"
                % (list(target.params), list(found))))
    return problems


# --------------------------------------------------------------------------
# Null-safe snapshots.
#
# _salvage_session (src/session.py:79-111) builds collaborators one at a time,
# so ledger / slots / asks / shown may each legitimately be None. Every helper
# below returns None rather than raising, and the backend renders that as a dash.
# --------------------------------------------------------------------------

def _attr(obj: object, name: str, default: object = None) -> object:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def snap_ledger(session: object) -> "dict | None":
    ledger = _attr(session, "ledger")
    if ledger is None:
        return None
    try:
        entries = tuple(ledger.entries)
        segments = tuple(ledger.segments)
        return {
            "entries": list(entries),
            "segments": list(segments),
            "entry_count": len(entries),
            "segment_count": len(segments),
            "distinct_segment_count": int(ledger.distinct_segment_count()),
            "query": str(ledger.query),
        }
    except Exception:
        return None


def snap_slots(session: object) -> "dict | None":
    slots = _attr(session, "slots")
    if slots is None:
        return None
    try:
        return dict(slots.as_dict())
    except Exception:
        return None


ASK_FIELDS = (
    "asked", "retired", "yield_counts", "yield_order", "burned",
    "burned_reasked", "last_ask", "turn", "disclosed_count",
)


def snap_askstate(asks: object) -> "dict | None":
    """Deep-copy the nine AskState fields.

    Must be taken BEFORE the call that mutates them. The recorded args tuple is
    no substitute: it holds a live reference, so reading it afterwards shows
    post-mutation values.
    """
    if asks is None:
        return None
    try:
        return {
            "asked": list(_attr(asks, "asked") or []),
            "retired": sorted(str(x) for x in (_attr(asks, "retired") or ())),
            "yield_counts": dict(_attr(asks, "yield_counts") or {}),
            "yield_order": list(_attr(asks, "yield_order") or []),
            "burned": _attr(asks, "burned"),
            "burned_reasked": bool(_attr(asks, "burned_reasked", False)),
            "last_ask": _attr(asks, "last_ask"),
            "turn": int(_attr(asks, "turn", 0) or 0),
            "disclosed_count": int(_attr(asks, "disclosed_count", 0) or 0),
        }
    except Exception:
        return None


def snap_shown(session: object) -> "dict | None":
    shown = _attr(session, "shown")
    if shown is None:
        return None
    try:
        return {"count": len(shown), "suppressed": bool(shown.suppressed)}
    except Exception:
        return None


def decode_dict(decode: object) -> "dict | None":
    if decode is None:
        return None
    try:
        return {
            "frame": str(_attr(decode, "frame", "")),
            "payload": str(_attr(decode, "payload", "")),
            "segments": [str(s) for s in (_attr(decode, "segments") or ())],
            "decline": str(_attr(decode, "decline", "none")),
            "scenario_signal": str(_attr(decode, "scenario_signal", "unknown")),
            "attribute": _attr(decode, "attribute"),
            "source": str(_attr(decode, "source", "")),
        }
    except Exception:
        return None


def cand_ids(items: object) -> list:
    """parent_asins only. Materialised NOW -- ``window`` is rebound three times."""
    out: list = []
    try:
        for item in items or ():
            asin = _attr(item, "parent_asin", "")
            out.append(str(asin) if asin else str(item))
    except Exception:
        pass
    return out


def cand_rows(items: object) -> list:
    out: list = []
    try:
        for position, item in enumerate(items or (), start=1):
            out.append({
                "rank": position,
                "parent_asin": str(_attr(item, "parent_asin", "") or ""),
                "rowid": int(_attr(item, "rowid", 0) or 0),
                "score": float(_attr(item, "score", 0.0) or 0.0),
                "hydrated": bool(str(_attr(item, "text", "") or "")),
            })
    except Exception:
        pass
    return out


def _moved(before: list, after: list) -> int:
    """How many items changed position between two orderings of the same set."""
    try:
        index = {asin: i for i, asin in enumerate(before)}
        return sum(1 for i, asin in enumerate(after) if index.get(asin, i) != i)
    except Exception:
        return 0


def _movers(before: list, after: list, limit: int = 5) -> list:
    try:
        index = {asin: i for i, asin in enumerate(before)}
        moves = []
        for i, asin in enumerate(after):
            was = index.get(asin)
            if was is not None and was != i:
                moves.append({"parent_asin": asin, "from": was + 1, "to": i + 1,
                              "delta": was - i})
        moves.sort(key=lambda m: -abs(m["delta"]))
        return moves[:limit]
    except Exception:
        return []


# --------------------------------------------------------------------------
# The per-turn recorder.
# --------------------------------------------------------------------------

class TurnRecorder:
    """Accumulates one turn's observations. One instance per ``respond()``."""

    def __init__(self, turn: int, top_k: int) -> None:
        self.turn = turn
        self.top_k = top_k
        self.errors = 0
        self.started = time.perf_counter()

        self.input: dict = {"turn_in": turn, "top_k_in": top_k}
        self.decode: dict = {}
        self.state: dict = {}
        self.retrieval: dict = {}
        self.partition: dict = {}
        self.window: dict = {}
        self.picks: dict = {}
        self.ask: dict = {}
        self.timing: dict = {}
        self.wire: dict = {}
        self.flags: dict = {"degraded_plan_fired": False}

        # Full-fidelity holds, trimmed only at finalise() once the target and
        # the final ten are known.
        self._pool_rows: list = []
        self._window_groups: dict = {}
        self._index = None
        self._raw_gate: list = []
        self._raw_rerank: list = []
        # The live post-gate Candidate objects, held only until finalise() runs
        # overlap.measure() over them. Candidates are frozen and the turn is
        # over by then, so holding them cannot perturb anything.
        self._gated_window: list = []

    # -- guarded plumbing -------------------------------------------------

    def guard(self, function, *args) -> None:
        """Run a recording step; count and swallow anything it throws."""
        try:
            function(*args)
        except Exception:
            self.errors += 1

    def time(self, key: str, seconds: float) -> None:
        try:
            self.timing[key] = round(seconds * 1000.0, 3)
        except Exception:
            self.errors += 1

    # -- stage handlers ---------------------------------------------------

    def on_normalise(self, args, result) -> None:
        self.input["user_message"] = str(args[0]) if args else ""
        self.input["normalised"] = str(result)
        self.input["normalised_chars"] = len(str(result))

    def on_clamp_turn(self, args, result) -> None:
        self.input["turn_clamped"] = int(result)

    def on_clamp_top_k(self, args, result) -> None:
        self.input["limit"] = int(result)

    def on_tier1(self, args, result) -> None:
        payload = decode_dict(result)
        self.decode["tier1"] = payload
        # frames.py:207-209 returns the hedge as unknown/refusal with an empty
        # payload; a plain unknown keeps the original text and decline "none".
        # So this is observed, not inferred.
        self.decode["tier15_hedge_fired"] = bool(
            payload and payload["frame"] == "unknown"
            and payload["decline"] == "refusal" and not payload["payload"])

    def on_tier2(self, args, result) -> None:
        incoming = decode_dict(args[0]) if args else None
        outgoing = decode_dict(result)
        ran = bool(incoming and incoming["frame"] == "unknown")
        self.decode["tier2_ran"] = ran
        self.decode["tier2_changed"] = bool(
            ran and outgoing and incoming and outgoing["frame"] != incoming["frame"])
        self.decode["tier2_reason"] = (
            "tier1 frame is unknown -- semantic fallback consulted" if ran
            else "tier1 frame != unknown, fallback skipped")
        self.decode["final"] = outgoing

    def on_note(self, args, result, before) -> None:
        session = args[0] if args else None
        self.state["scenario_before"] = before
        self.state["scenario_after"] = _attr(session, "scenario")
        try:
            self.state["frame_counts"] = dict(_attr(session, "frame_counts") or {})
        except Exception:
            self.state["frame_counts"] = {}

    def on_override_guard(self, args, result, before) -> None:
        session = args[0] if args else None
        decode = decode_dict(args[1]) if len(args) > 1 else None
        turn_number = int(args[2]) if len(args) > 2 else self.turn
        after = {"shown": snap_shown(session),
                 "applied": bool(_attr(session, "override_applied", False))}

        frame = (decode or {}).get("frame", "")
        signal = (decode or {}).get("scenario_signal", "")
        before_shown = before.get("shown") or {}
        after_shown = after.get("shown") or {}

        # The three branches of pipeline.py:232-263, labelled from the observed
        # state transition rather than from re-running the conditions.
        if frame == "override":
            action = "restore_all + release (override landed)"
        elif not before.get("applied") and after_shown.get("suppressed") and not before_shown.get("suppressed"):
            action = "suppress (override expected)"
        elif before_shown.get("suppressed") and not after_shown.get("suppressed"):
            action = "release (false positive healed)"
        else:
            action = "none"

        self.state["override_guard"] = {
            "action": action, "action_derived": True,
            "frame": frame, "signal": signal, "turn": turn_number,
            "suppressed_before": bool(before_shown.get("suppressed")),
            "suppressed_after": bool(after_shown.get("suppressed")),
            "override_applied_before": bool(before.get("applied")),
            "override_applied_after": bool(after.get("applied")),
            "shown_before": int(before_shown.get("count", 0)),
            "shown_after": int(after_shown.get("count", 0)),
        }

    def on_append_to_ledger(self, args, result, before) -> None:
        session = args[0] if args else None
        after = snap_ledger(session)
        appended = None
        if before and after:
            extra = after["entries"][len(before["entries"]):]
            appended = extra[0] if extra else None
        self.state["ledger"] = {
            "entries_before": (before or {}).get("entry_count"),
            "entries_after": (after or {}).get("entry_count"),
            "appended": appended,
            "segments_before": (before or {}).get("segment_count"),
            "segments_after": (after or {}).get("segment_count"),
            "segments": (after or {}).get("segments", []),
            "distinct_segment_count": (after or {}).get("distinct_segment_count"),
        }

    def on_fill_slots(self, args, result, before) -> None:
        session = args[0] if args else None
        after = snap_slots(session)
        filled: list = []
        cleared: list = []
        if before is not None and after is not None:
            for key, value in after.items():
                if before.get(key) != value:
                    filled.append({"attribute": key, "value": value,
                                   "replaced": before.get(key)})
            cleared = [k for k in before if k not in after]
        self.state["slots"] = {
            "before": before, "after": after,
            "filled": filled,
            "cleared_by_override": cleared or None,
        }

    def on_ask_bookkeeping(self, args, result, before) -> None:
        session = args[0] if args else None
        decode = decode_dict(args[1]) if len(args) > 1 else None
        after = snap_askstate(_attr(session, "asks"))
        frame = (decode or {}).get("frame", "")
        decline = (decode or {}).get("decline", "none")

        # The four-way branch at pipeline.py:347-360.
        if frame == "override":
            branch = "burn(last_ask)"
        elif frame == "refusal" or decline == "refusal":
            branch = "burn(target)"
        elif frame == "exhaustion" or decline == "exhaustion":
            branch = "record_reply(exhaustion) -> retires the slot"
        elif frame == "disclosure":
            branch = "record_reply(disclosure)"
        else:
            branch = "none"

        self.state["ask_bookkeeping"] = {
            "branch": branch, "branch_derived": True,
            "target": (decode or {}).get("attribute") or (before or {}).get("last_ask"),
            "segment_count": len((decode or {}).get("segments", [])),
            "decline": decline,
            "before": before, "after": after,
        }

    def on_query_for(self, args, result, ledger_query) -> None:
        query = str(result)
        source = "ledger" if ledger_query and query == ledger_query else "raw message"
        self.retrieval["query"] = query
        self.retrieval["query_source"] = source
        self.retrieval["query_source_derived"] = True
        self.retrieval["query_chars"] = len(query)

    def on_search(self, args, result) -> None:
        deps = args[0] if args else None
        query = str(args[1]) if len(args) > 1 else ""
        limit = int(args[2]) if len(args) > 2 else 0
        index = _attr(deps, "index")
        self._index = index
        self._pool_rows = cand_rows(result)

        # _search asks the index for max(POOL_SIZE, limit), not for limit
        # (pipeline.py:393). Reporting the top_k here would misdescribe the
        # search by a factor of thirty.
        try:
            from src.types import POOL_SIZE
            depth = max(int(POOL_SIZE), limit)
        except Exception:
            depth = limit
        self.retrieval["top_k_limit"] = limit
        self.retrieval["pool_depth_requested"] = depth
        self.retrieval["pool_size"] = len(self._pool_rows)
        self.retrieval["score_note"] = "raw SQLite bm25(); MORE NEGATIVE = BETTER"
        self.retrieval["search_skipped_reason"] = (
            None if index is not None else "deps.index is None (degraded agent)")

        # The literal FTS5 MATCH expression. _match_expression is private but
        # pure and total (retrieval.py:305-326); it is the only way to show what
        # actually went to SQLite, so it is derived and labelled as such.
        expression = None
        terms = None
        if index is not None:
            try:
                expression = str(index._match_expression(query))
                terms = expression.count('"') // 2
            except Exception:
                expression = None
        self.retrieval["match_expression"] = expression
        self.retrieval["match_expression_derived"] = True
        self.retrieval["match_expression_source"] = "Bm25Index._match_expression(query)"
        self.retrieval["match_terms"] = terms
        try:
            from src.types import MAX_QUERY_TERMS
            self.retrieval["match_term_cap"] = int(MAX_QUERY_TERMS)
            self.retrieval["match_capped"] = bool(terms and terms >= int(MAX_QUERY_TERMS))
        except Exception:
            self.retrieval["match_term_cap"] = None
            self.retrieval["match_capped"] = False

    def on_partition(self, args, result) -> None:
        pool = args[1] if len(args) > 1 else ()
        fresh, seen = (result if isinstance(result, tuple) and len(result) == 2
                       else (result, []))
        fresh_ids = cand_ids(fresh)
        seen_ids = cand_ids(seen)
        self._window_groups["fresh"] = fresh_ids
        self.partition = {
            "fresh_count": len(fresh_ids),
            "seen_count": len(seen_ids),
            "pool_count": len(cand_ids(pool)),
            "is_true_partition": len(fresh_ids) + len(seen_ids) == len(cand_ids(pool)),
            "seen_ids": seen_ids[:25],
            "fresh_top": fresh_ids[:10],
        }

    def on_hydrate(self, args, result) -> None:
        incoming = cand_ids(args[1]) if len(args) > 1 else []
        outgoing = cand_rows(result)
        self._window_groups["pre_hydrate"] = incoming
        self._window_groups["post_hydrate"] = [r["parent_asin"] for r in outgoing]
        self.window["window_size_observed"] = len(incoming)
        self.window["hydrated_count"] = sum(1 for r in outgoing if r["hydrated"])
        self.window["hydrate_changed_order"] = (
            incoming != self._window_groups["post_hydrate"])

    def on_rerank(self, args, result) -> None:
        deps = args[0] if args else None
        incoming = cand_ids(args[2]) if len(args) > 2 else []
        outgoing = cand_ids(result)
        self._window_groups["post_rerank"] = outgoing
        reranker = _attr(deps, "reranker")
        self.window["rerank_name"] = str(_attr(reranker, "name", "") or type(reranker).__name__)
        self.window["rerank_declared_inert"] = "null" in self.window["rerank_name"].lower()
        self.window["rerank_changed_order"] = incoming != outgoing
        # If the raw reranker moved things but the stage did not, the
        # permutation guard (_same_multiset_or_original) threw the result away.
        self.window["rerank_guard_rejected"] = bool(
            self._raw_rerank and self._raw_rerank != outgoing)

    def on_gate(self, args, result, segments) -> None:
        incoming = cand_ids(args[2]) if len(args) > 2 else []
        outgoing = cand_ids(result)
        self._window_groups["post_gate"] = outgoing
        self.window["gate_segments"] = list(segments or [])
        self.window["gate_segments_source"] = "ledger.segments"
        self.window["gate_changed_order"] = incoming != outgoing
        self.window["gate_positions_moved"] = _moved(incoming, outgoing)
        self.window["gate_movers"] = _movers(incoming, outgoing)
        self.window["gate_guard_rejected"] = bool(
            self._raw_gate and self._raw_gate != outgoing)

    def on_assemble(self, args, result) -> None:
        window = cand_ids(args[0]) if args else []
        rest = cand_ids(args[1]) if len(args) > 1 else []
        seen = cand_ids(args[2]) if len(args) > 2 else []
        limit = int(args[3]) if len(args) > 3 else self.top_k
        picks = [str(p) for p in (result or ())]

        self._window_groups["rest"] = rest
        self._window_groups["seen_at_assemble"] = seen
        self.window["rest_size_observed"] = len(rest)

        # Provenance: which of the three groups each pick came from. This is
        # where _assemble's arguments earn their keep -- picks is list[str] and
        # has dropped every score by the time it returns.
        position = {}
        for group_name, group in (("window", window), ("rest", rest), ("seen", seen)):
            for i, asin in enumerate(group):
                position.setdefault(asin, "%s#%d" % (group_name, i))

        self.picks = {
            "limit": limit,
            "parent_asins": picks,
            "provenance": [position.get(a, "?") for a in picks],
            "from_window": sum(1 for a in picks if position.get(a, "").startswith("window")),
            "from_rest": sum(1 for a in picks if position.get(a, "").startswith("rest")),
            "from_seen": sum(1 for a in picks if position.get(a, "").startswith("seen")),
        }

    def on_record(self, args, result, before) -> None:
        session = args[0] if args else None
        after = snap_shown(session)
        self.picks["record_suppressed"] = bool((before or {}).get("suppressed"))
        self.picks["shown_before"] = int((before or {}).get("count", 0))
        self.picks["shown_after"] = int((after or {}).get("count", 0))
        self.picks["recorded"] = self.picks["shown_after"] - self.picks["shown_before"]

    def on_next_attribute(self, args, result, snapshot) -> None:
        """Stage 18a -- the only place the pre-selection AskState is readable."""
        self.ask["state_before"] = snapshot
        self.ask["policy_return"] = result if isinstance(result, str) else None
        try:
            from src import askyield
            self.ask["adaptive_enabled"] = bool(getattr(askyield, "ADAPTIVE_ENABLED", False))
        except Exception:
            self.ask["adaptive_enabled"] = None

    def on_fallback_attribute(self, args, result) -> None:
        self.ask["fallback_fired"] = True
        self.ask["fallback_return"] = str(result)

    def on_choose_attribute(self, args, result) -> None:
        self.ask["final"] = str(result)

    def on_message_for(self, args, result) -> None:
        self.ask["message"] = str(result)

    # -- finalise ---------------------------------------------------------

    def finalise(self, target_asin: str = "", trace_pool: int = 25) -> dict:
        """Assemble the ``turn`` record's blocks. Called once, after respond()."""
        self.ask.setdefault("fallback_fired", False)
        self.window.setdefault("rerank_guard_rejected", False)
        self.window.setdefault("gate_guard_rejected", False)

        self._finalise_split()
        self._finalise_rung()
        pool = self._finalise_pool(target_asin, trace_pool)
        self._finalise_scores()

        self.timing["turn_ms"] = round((time.perf_counter() - self.started) * 1000.0, 3)

        record = {
            "input": self.input,
            "decode": self.decode,
            "state": self.state,
            "retrieval": dict(self.retrieval, pool=pool),
            "partition": self.partition,
            "window": self.window,
            "overlap_report": self._overlap_report(),
            "picks": self.picks,
            "ask": self.ask,
            "wire": self.wire,
            "timing": self.timing,
            "tracer": {
                "record_errors": self.errors,
                "pool_truncated": len(self._pool_rows) > len(pool),
            },
        }
        return record

    def _finalise_split(self) -> None:
        """Reconstruct the inline ``fresh[:50] / fresh[50:]`` split at :137.

        It is the one boundary with no call to wrap, so it is derived -- and
        then checked against two independent observations (``_hydrate``'s
        argument and ``_assemble``'s). ``split_consistent`` going False means a
        refactor moved the slice and every window number below is suspect.
        """
        try:
            from src.pipeline import RERANK_WINDOW
            width = int(RERANK_WINDOW)
        except Exception:
            width = None
        fresh = self._window_groups.get("fresh", [])
        derived_window = fresh[:width] if width is not None else []
        derived_rest = fresh[width:] if width is not None else []

        self.window["rerank_window"] = width
        self.window["window_size"] = len(derived_window)
        self.window["rest_size"] = len(derived_rest)
        self.window["window_size_derived"] = True

        observed_window = self.window.get("window_size_observed")
        observed_rest = self.window.get("rest_size_observed")
        self.window["split_consistent"] = bool(
            observed_window is None or observed_rest is None
            or (observed_window == len(derived_window)
                and observed_rest == len(derived_rest)))

        for key in ("pre_hydrate", "post_hydrate", "post_rerank", "post_gate"):
            self.window[key] = self._window_groups.get(key, [])[:15]

    def _finalise_rung(self) -> None:
        """Label which rung of the ask ladder fired, and cross-check it."""
        snapshot = self.ask.get("state_before")
        policy = self.ask.get("policy_return")
        if self.ask.get("fallback_fired"):
            self.ask["rung"] = askrung.PIPELINE_FALLBACK
            self.ask["rung_reason"] = (
                "askyield returned %r, which _valid_ask rejected; "
                "_fallback_attribute chose instead" % (policy,))
            self.ask["rung_predicted_attribute"] = self.ask.get("fallback_return")
            self.ask["rung_agrees"] = True
            self.ask["rung_derived"] = True
            return

        rung, reason, predicted = askrung.derive(snapshot)
        self.ask["rung"] = rung
        self.ask["rung_reason"] = reason
        self.ask["rung_predicted_attribute"] = predicted
        self.ask["rung_derived"] = True
        self.ask["rung_agrees"] = bool(predicted is not None and predicted == policy)

    def _finalise_pool(self, target_asin: str, trace_pool: int) -> list:
        """Trim the pool for the wire, but never drop a row the render needs.

        The must-include set is what stops a pick that came from ``rest`` (rank
        60, say) rendering with no BM25 score at all.
        """
        must = set(self.picks.get("parent_asins", []))
        if target_asin:
            must.add(target_asin)

        out: list = []
        for row in self._pool_rows:
            keep = row["rank"] <= max(0, trace_pool) or row["parent_asin"] in must
            if not keep:
                continue
            entry = dict(row)
            if target_asin and row["parent_asin"] == target_asin:
                entry["is_target"] = True
            if row["rank"] > max(0, trace_pool):
                entry["beyond_trace_cap"] = True
            out.append(entry)
        return out

    def _finalise_scores(self) -> None:
        scores = {r["parent_asin"]: r["score"] for r in self._pool_rows}
        self.picks["scores"] = [scores.get(a) for a in self.picks.get("parent_asins", [])]

    def _overlap_report(self) -> "dict | None":
        """Run src/overlap.py:138 measure() -- the repo's own unwired instrument.

        Observation only: nothing in the pipeline filters on it, and calling it
        here cannot affect the turn, which has already finished.
        """
        try:
            from src import overlap
            candidates = self._gated_window
            segments = self.window.get("gate_segments") or []
            if not candidates:
                return None
            report = overlap.measure(candidates, segments)
            return {
                "derived": True,
                "measured_on": "post_gate_window",
                "function": "src.overlap.measure",
                "segments": int(report.segments),
                "matched": int(report.matched),
                "rate": float(report.rate),
                "top_overlap": int(report.top_overlap),
            }
        except Exception:
            self.errors += 1
            return None


# --------------------------------------------------------------------------
# Install / restore.
# --------------------------------------------------------------------------

class Tracer:
    """Owns the patches and the currently-recording turn."""

    def __init__(self) -> None:
        self.recorder: "TurnRecorder | None" = None
        self.errors = 0
        self.degraded_plan_fired = False
        self._saved: list = []
        self._installed = False

    # -- turn boundary ----------------------------------------------------

    def begin(self, turn: int, top_k: int) -> TurnRecorder:
        self.recorder = TurnRecorder(turn, top_k)
        return self.recorder

    def end(self) -> "TurnRecorder | None":
        recorder, self.recorder = self.recorder, None
        return recorder

    # -- patching ---------------------------------------------------------

    def install(self) -> list:
        problems = verify_targets()
        if problems:
            return problems
        for target in PATCH_TARGETS:
            module = importlib.import_module(target.module)
            original = getattr(module, target.name)
            self._saved.append((module, target.name, original))
            setattr(module, target.name, self._wrap(target, original))
        self._installed = True
        return []

    def restore(self) -> bool:
        ok = True
        while self._saved:
            module, name, original = self._saved.pop()
            try:
                setattr(module, name, original)
                if getattr(module, name) is not original:
                    ok = False
            except Exception:
                ok = False
        self._installed = False
        return ok

    def _wrap(self, target: Target, original):
        handler = getattr(self, "_h_" + target.handler, None)

        def wrapper(*args, **kwargs):
            recorder = self.recorder
            if recorder is None or handler is None:
                return original(*args, **kwargs)

            # The one thing that must precede the original call: a snapshot of
            # state the call is about to mutate. Guarded, so it can only cost a
            # counted error, never the turn.
            before = None
            try:
                before = handler(recorder, args, None, None, pre=True)
            except Exception:
                recorder.errors += 1

            start = time.perf_counter()
            result = original(*args, **kwargs)      # nothing may precede this
            elapsed = time.perf_counter() - start   # timing excludes recording

            try:
                handler(recorder, args, result, before, pre=False)
                # Keyed by handler, not stage: the three canaries all carry
                # stage "canary" and would otherwise overwrite each other.
                recorder.time(target.handler, elapsed)
            except Exception:
                recorder.errors += 1
            return result

        wrapper.__name__ = getattr(original, "__name__", target.name)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        wrapper._demo_original = original
        return wrapper

    # -- handlers: (recorder, args, result, before, pre) -------------------
    # `pre=True` returns the before-snapshot; `pre=False` records.

    def _h_normalise(self, rec, args, result, before, pre):
        if not pre:
            rec.on_normalise(args, result)

    def _h_clamp_turn(self, rec, args, result, before, pre):
        if not pre:
            rec.on_clamp_turn(args, result)

    def _h_clamp_top_k(self, rec, args, result, before, pre):
        if not pre:
            rec.on_clamp_top_k(args, result)

    def _h_tier1(self, rec, args, result, before, pre):
        if not pre:
            rec.on_tier1(args, result)

    def _h_tier2(self, rec, args, result, before, pre):
        if not pre:
            rec.on_tier2(args, result)

    def _h_note(self, rec, args, result, before, pre):
        if pre:
            return _attr(args[0] if args else None, "scenario")
        rec.on_note(args, result, before)

    def _h_override_guard(self, rec, args, result, before, pre):
        session = args[0] if args else None
        if pre:
            return {"shown": snap_shown(session),
                    "applied": bool(_attr(session, "override_applied", False))}
        rec.on_override_guard(args, result, before or {})

    def _h_append_to_ledger(self, rec, args, result, before, pre):
        if pre:
            return snap_ledger(args[0] if args else None)
        rec.on_append_to_ledger(args, result, before)

    def _h_fill_slots(self, rec, args, result, before, pre):
        if pre:
            return snap_slots(args[0] if args else None)
        rec.on_fill_slots(args, result, before)

    def _h_ask_bookkeeping(self, rec, args, result, before, pre):
        if pre:
            return snap_askstate(_attr(args[0] if args else None, "asks"))
        rec.on_ask_bookkeeping(args, result, before)

    def _h_query_for(self, rec, args, result, before, pre):
        if pre:
            snapshot = snap_ledger(args[0] if args else None)
            return (snapshot or {}).get("query")
        rec.on_query_for(args, result, before)

    def _h_search(self, rec, args, result, before, pre):
        if not pre:
            rec.on_search(args, result)

    def _h_partition(self, rec, args, result, before, pre):
        if not pre:
            rec.on_partition(args, result)

    def _h_hydrate(self, rec, args, result, before, pre):
        if not pre:
            rec.on_hydrate(args, result)

    def _h_rerank(self, rec, args, result, before, pre):
        if not pre:
            rec.on_rerank(args, result)

    def _h_gate(self, rec, args, result, before, pre):
        if pre:
            session = args[0] if args else None
            snapshot = snap_ledger(session)
            segments = (snapshot or {}).get("segments") or []
            if not segments and len(args) > 1:
                segments = (decode_dict(args[1]) or {}).get("segments") or []
            return segments
        rec.on_gate(args, result, before)
        # Held for overlap.measure() at finalise: these carry the hydrated text
        # the instrument reads, in the order the gate actually produced.
        try:
            rec._gated_window = list(result or ())
        except Exception:
            rec.errors += 1

    def _h_assemble(self, rec, args, result, before, pre):
        if not pre:
            rec.on_assemble(args, result)

    def _h_record(self, rec, args, result, before, pre):
        if pre:
            return snap_shown(args[0] if args else None)
        rec.on_record(args, result, before)

    def _h_choose_attribute(self, rec, args, result, before, pre):
        if not pre:
            rec.on_choose_attribute(args, result)

    def _h_next_attribute(self, rec, args, result, before, pre):
        if pre:
            return snap_askstate(args[0] if args else None)
        rec.on_next_attribute(args, result, before)

    def _h_fallback_attribute(self, rec, args, result, before, pre):
        if not pre:
            rec.on_fallback_attribute(args, result)

    def _h_message_for(self, rec, args, result, before, pre):
        if not pre:
            rec.on_message_for(args, result)

    # -- canaries ---------------------------------------------------------

    def _h_degraded_plan(self, rec, args, result, before, pre):
        if not pre:
            rec.flags["degraded_plan_fired"] = True
            self.degraded_plan_fired = True

    def _h_overlap_gate(self, rec, args, result, before, pre):
        """Raw ``src.overlap.gate`` output, before pipeline's permutation guard."""
        if not pre:
            rec._raw_gate = cand_ids(result)

    def _h_safe_rerank(self, rec, args, result, before, pre):
        if not pre:
            rec._raw_rerank = cand_ids(result)


def make_tracing_agent(catalog_path: str, tracer: Tracer):
    """An ``Agent`` that opens and closes a recorder around each ``respond()``.

    A subclass rather than a patch on ``src.pipeline.run_turn``, because
    ``src/agent.py:40`` binds ``run_turn`` at import -- patching the pipeline's
    copy would never fire. Precedent: ``tests/test_src_end_to_end.py:139``.

    ``respond()`` stays a never-raise wrapper: the bookkeeping around the
    ``super()`` call is guarded, so a tracer bug cannot turn into a zeroed turn.
    """
    from agent import Agent

    class TracingAgent(Agent):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.turns: list = []

        def respond(self, session_id, user_message, turn, top_k):
            recorder = None
            try:
                recorder = tracer.begin(int(turn), int(top_k))
            except Exception:
                tracer.errors += 1
            response = super().respond(session_id, user_message, turn, top_k)
            try:
                tracer.end()
                if recorder is not None:
                    recorder.wire = {
                        "message": str(response.get("message", "")),
                        "ask_attribute": response.get("ask_attribute"),
                        "recommendation_count": len(response.get("recommendations", []) or []),
                        "usage": response.get("usage") or {},
                        "degraded_plan_fired": bool(recorder.flags.get("degraded_plan_fired")),
                    }
                    self.turns.append(recorder)
            except Exception:
                tracer.errors += 1
            return response

    return TracingAgent(catalog_path)


@contextmanager
def installed(tracer: "Tracer | None" = None):
    """Install the patches for the duration of the block, always restoring."""
    tracer = tracer or Tracer()
    problems = tracer.install()
    if problems:
        tracer.restore()
        raise RuntimeError(
            "demo tracer cannot attach to src/pipeline.py:\n  "
            + "\n  ".join(str(p) for p in problems))
    try:
        yield tracer
    finally:
        tracer.restore()
