"""The per-turn orchestration -- stateless.  [WS-E OWNS]

One turn, repeated up to ten times. The ledger persists across the loop;
everything else is rebuilt from it each turn. The override branch does NOT skip
the ledger append or the ask -- it only clears a conflicting slot and restores
the shown-set before rejoining the same path.

Every step below runs on every turn and is FORBIDDEN TO RAISE. The evaluator
swallows any exception into an empty response: no crash, no traceback, just a
quietly worse score. So every collaborator call sits in its own try/except with
a local fallback, and the whole turn sits in one more.

Two invariants worth stating outright, because both are silent when broken:

  - STEP 16 NEVER RETURNS SHORT. picks = window + rest-of-fresh + seen, cut to
    top_k. The shown set reorders; it never removes. A filter here is the one
    change that turns a full ten into an empty list on a drained pool.
  - STEP 16 NEVER RETURNS A DUPLICATE. normalize_recommendations() drops a
    repeated parent_asin without a word, so a duplicate costs a slot out of the
    ten with no error anywhere. The assembler dedupes as it builds.

Slots are scheduling-only. They are filled every turn and never reach the query:
the query IS the ledger. That separation is the safety property -- a parsing bug
can corrupt what we ask next, but it cannot corrupt what we search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src import askyield, frames, overlap, rerank, semantic, slots
from src.contract import clamp_top_k
from src.retrieval import Bm25Index
from src.session import Session
from src.shown import parent_asin_of
from src.types import (
    ALLOWED_ATTRIBUTES,
    CONTENT_FREE_FRAMES,
    FIXED_SCHEDULE,
    FORBIDDEN_ASK,
    MAX_TURNS,
    POOL_SIZE,
    RERANK_WINDOW,
    Candidate,
    Decode,
    Reranker,
    SemanticDecoder,
    TurnPlan,
)

ASK_TEMPLATES: dict[str, str] = {
    "material": "Do you have a material preference?",
    "feature": "Is there a specific feature that matters most to you?",
    "color": "Do you have a color preference?",
    "style": "What style are you looking for?",
    "size": "Do you have a size in mind?",
    "use_case": "What will you mainly use this for?",
    "budget": "Roughly what budget did you have in mind?",
    "brand": "Is there a brand you prefer?",
    "category": "What kind of item are you after, specifically?",
}
CLOSING_MESSAGE = "Here are the closest matches I found so far."

# behavior_for() draws the override turn from [3, 4], never later, so a session
# still suppressed after turn 3 was a false positive and heals itself here.
OVERRIDE_SUPPRESS_MAX_TURN: int = 3


@dataclass(frozen=True)
class Deps:
    """Everything built once at Agent construction and shared across sessions."""

    index: Bm25Index | None = None
    reranker: Reranker | None = None
    semantic: SemanticDecoder | None = None
    llm_reranker: Reranker | None = None


def run_turn(session: Session, user_message: object, turn: object,
             top_k: int, deps: Deps) -> TurnPlan:
    """One turn. See the docstring above for the ordering contract.

    WS-E implements, in this order:
      1  normalise the message
      2  clamp the turn onto [1, MAX_TURNS]
      3  Tier 1 decode
      4  Tier 2 only on `unknown`, and it never overrides Tier 1
      5  record the scenario signal and the frame count
      6  OVERRIDE GUARD: suppress / restore / release the shown set
      7  ledger append (verbatim) + record segments
      8  slots: fill, and on the override frame run the DIFF and clear
      9  ask bookkeeping: burn, retire, refresh disclosed_count
     10  query = ledger.query or the raw message
     11  pool = index.search(query, POOL_SIZE)
     12  fresh, seen = shown.partition(pool)
     13  hydrate the rerank window
     14  safe_rerank  (order-only, permutation-checked)
     15  overlap.gate (order-only, stable -- composes with 14)
     16  LLMRR: escalate to the LLM reranker ONLY when overlap.gate found
         zero literal overlap ("vague") -- report.md's design of record.
         Skipped whenever deps.llm_reranker is None, which is every turn
         until a model is chosen. Order-only, safe_rerank-guarded like 14.
     17  picks = (window + rest of fresh + seen)[:top_k]   <- never short
     18  shown.record(picks)
     19  attribute = askyield.next_attribute(...)  <- never None, never "other"
     20  return the TurnPlan
    """
    try:
        return _run_turn(session, user_message, turn, top_k, deps)
    except Exception:
        # Unreachable by design -- every step below is guarded individually.
        # Kept because "unreachable" and "cannot happen" are different claims,
        # and the cost of being wrong is the whole turn.
        return _degraded_plan(session)


# --------------------------------------------------------------------------
# The turn.
# --------------------------------------------------------------------------

def _run_turn(session: Session, user_message: object, turn: object,
              top_k: int, deps: Deps) -> TurnPlan:
    if not isinstance(deps, Deps):
        deps = Deps()

    text = _normalise(user_message)                              # 1
    turn_number = _clamp_turn(turn, session)                     # 2
    _set(session, "turn", turn_number)
    limit = clamp_top_k(top_k)

    decode = _tier1(text)                                        # 3
    decode = _tier2(decode, text, deps)                          # 4
    _note(session, decode)                                       # 5
    _override_guard(session, decode, turn_number)                # 6
    _append_to_ledger(session, decode)                           # 7
    _fill_slots(session, decode)                                 # 8
    _ask_bookkeeping(session, decode, turn_number)               # 9

    query = _query_for(session, text)                            # 10
    pool = _search(deps, query, limit)                           # 11
    fresh, seen = _partition(session, pool)                      # 12
    window, rest = fresh[:RERANK_WINDOW], fresh[RERANK_WINDOW:]
    window = _hydrate(deps, window)                              # 13
    window = _rerank(deps, query, window)                        # 14
    window = _gate(session, decode, window)                      # 15
    window, prompt_tokens, completion_tokens = \
        _llm_escalate(deps, query, session, decode, window)      # 16
    picks = _assemble(window, rest, seen, limit)                 # 17
    _record(session, picks)                                      # 18

    attribute = _choose_attribute(session, turn_number)          # 19
    return TurnPlan(                                             # 20
        message=_message_for(attribute),
        ask_attribute=attribute,
        parent_asins=tuple(picks),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# --------------------------------------------------------------------------
# 1-2  input coercion.
# --------------------------------------------------------------------------

def _normalise(user_message: object) -> str:
    try:
        text = frames.normalise(user_message)
    except Exception:
        text = user_message if isinstance(user_message, str) else ""
    return text if isinstance(text, str) else ""


def _clamp_turn(turn: object, session: object) -> int:
    """Onto [1, MAX_TURNS]. A non-numeric turn falls back to the session's own
    counter, so a harness that passes garbage still walks forward."""
    value: int | None = None
    try:
        if not isinstance(turn, bool):
            value = int(turn)  # int, float, Decimal, numeric str all land here
    except Exception:
        value = None
    if value is None:
        try:
            previous = getattr(session, "turn", 0)
            value = (previous if isinstance(previous, int) else 0) + 1
        except Exception:
            value = 1
    return max(1, min(MAX_TURNS, value))


# --------------------------------------------------------------------------
# 3-5  intent.
# --------------------------------------------------------------------------

def _tier1(text: str) -> Decode:
    try:
        decoded = frames.decode(text)
    except Exception:
        decoded = None
    if isinstance(decoded, Decode):
        return decoded
    return Decode(frame="unknown", payload=text, source="none")


def _tier2(decode: Decode, text: str, deps: Deps) -> Decode:
    """Fires ONLY on a Tier 1 `unknown`, and never overrides Tier 1.

    On None -- the abstention -- the Tier 1 decode is kept unchanged, which is
    what makes a mediocre fallback safe to have at all.
    """
    if decode.frame != "unknown":
        return decode
    try:
        alternative = semantic.safe_decode(getattr(deps, "semantic", None), text)
    except Exception:
        return decode
    if isinstance(alternative, Decode) and alternative.frame != "unknown":
        return alternative
    return decode


def _note(session: Session, decode: Decode) -> None:
    try:
        session.note_frame(decode.frame)
    except Exception:
        pass
    try:
        signal = decode.scenario_signal
        # Never downgrade a known scenario back to "unknown": a mid-session
        # frame that carries no signal is silence, not a retraction.
        if isinstance(signal, str) and signal and signal != "unknown":
            _set(session, "scenario", signal)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 6  the override guard.
# --------------------------------------------------------------------------

def _override_guard(session: Session, decode: Decode, turn_number: int) -> None:
    """Suppress from turn 1; restore and release when the override lands.

    Suppressing from turn 1 rather than only restoring at the override turn is
    worth a whole turn: under restore-only, turn 1 shows candidates 1-10 and
    marks them shown, turn 2 shows 11-20 -- and neither is scored -- so the
    first scored turn tests our second-best list. Frame 2 is structurally
    distinguishable from frames 1 and 3 at turn 1, so we can just not record.
    """
    registry = getattr(session, "shown", None)
    if registry is None:
        return
    frame = decode.frame if isinstance(decode.frame, str) else "unknown"
    signal = decode.scenario_signal if isinstance(decode.scenario_signal, str) else "unknown"

    if frame == "override":
        # The override landed. Everything shown while the hit check was off is
        # unproven, so it all goes back in play, and recording resumes.
        _set(session, "override_applied", True)
        _call(registry.restore_all)
        _call(registry.release)
        return

    applied = getattr(session, "override_applied", False) is True
    if not applied and turn_number <= OVERRIDE_SUPPRESS_MAX_TURN:
        if frame == "override_open" or signal == "intent_override":
            _call(registry.suppress)

    if turn_number > OVERRIDE_SUPPRESS_MAX_TURN and _truthy(registry, "suppressed"):
        # Nothing arrived by turn 4, so turn-1 detection was a false positive.
        # Recording resumes; the shown set is empty, so nothing is lost.
        _call(registry.release)


# --------------------------------------------------------------------------
# 7  the ledger. Verbatim, append-only, and it IS the query.
# --------------------------------------------------------------------------

def _append_to_ledger(session: Session, decode: Decode) -> None:
    ledger = getattr(session, "ledger", None)
    if ledger is None:
        return
    frame = decode.frame if isinstance(decode.frame, str) else "unknown"
    payload = decode.payload if isinstance(decode.payload, str) else ""
    # The Decode contract already sets payload="" for the three content-free
    # frames; this is the second lock on the same door, using the frozen
    # CONTENT_FREE_FRAMES rather than a private copy of the list.
    if payload.strip() and frame not in CONTENT_FREE_FRAMES:
        _call(ledger.append, payload)
    segments = _segments_of(decode)
    if segments:
        _call(ledger.record_segments, segments)


def _segments_of(decode: Decode) -> tuple[str, ...]:
    try:
        segments = decode.segments
        if not segments:
            return ()
        return tuple(item for item in segments if isinstance(item, str) and item.strip())
    except Exception:
        return ()


# --------------------------------------------------------------------------
# 8  slots. Scheduling only -- these never reach the query.
# --------------------------------------------------------------------------

def _fill_slots(session: Session, decode: Decode) -> None:
    state = getattr(session, "slots", None)
    if state is None:
        return
    segments = _segments_of(decode)
    if decode.frame == "override":
        # DIFF and clear BEFORE filling: apply_override classifies the new value
        # and clears the slot it contradicts, so filling first would leave it
        # comparing the new value against itself and finding no conflict.
        new_value = segments[0] if segments else (decode.payload or "")
        if new_value:
            _call(slots.apply_override, state, new_value)
    for segment in segments:
        attribute = _classify(segment)
        if attribute:
            _call(state.fill, attribute, segment)


def _classify(segment: str) -> str:
    try:
        attribute = slots.classify_local(segment)
    except Exception:
        return ""
    return attribute if isinstance(attribute, str) and attribute else ""


# --------------------------------------------------------------------------
# 9  ask bookkeeping.
# --------------------------------------------------------------------------

def _ask_bookkeeping(session: Session, decode: Decode, turn_number: int) -> None:
    asks = getattr(session, "asks", None)
    if asks is None:
        return
    _set(asks, "turn", turn_number)

    frame = decode.frame if isinstance(decode.frame, str) else "unknown"
    last_ask = getattr(asks, "last_ask", None)
    last_ask = last_ask if isinstance(last_ask, str) else None
    attribute = decode.attribute if isinstance(decode.attribute, str) else None
    target = attribute or last_ask
    decline = decode.decline if isinstance(decode.decline, str) else "none"

    # The decline is read alongside the frame, not underneath it. Tier 1.5
    # returns frame="unknown" carrying decline="refusal" for a decline it could
    # not match to a template -- half the point of that hedge is burning the ask
    # so it gets re-asked, and a frame-only branch here would drop that half.
    if frame == "override":
        # customer_reply() is never called on the override turn, so the ask we
        # sent last turn was never read. Burn it so it can be re-asked.
        _call(asks.burn, last_ask)
    elif frame == "refusal" or decline == "refusal":
        # A refusal proves nothing about the bucket -- the customer declined to
        # look, not to have a preference. Burn the ask, keep the attribute live.
        _call(asks.burn, target)
    elif frame == "exhaustion" or decline == "exhaustion":
        # Provably drained: 0 segments plus the `exhaustion` decline is what
        # record_reply() retires on. Never ask it again this session.
        _call(asks.record_reply, target, 0, "exhaustion")
    elif frame == "disclosure":
        _call(asks.record_reply, target, len(_segments_of(decode)), decline)

    try:
        ledger = getattr(session, "ledger", None)
        if ledger is not None:
            count = ledger.distinct_segment_count()
            if isinstance(count, int):
                _set(asks, "disclosed_count", count)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 10-12  the query and the pool.
# --------------------------------------------------------------------------

def _query_for(session: Session, text: str) -> str:
    """The query IS the ledger. The raw message is the turn-1 fallback only --
    a browsing session discloses nothing on its first turn, so the ledger is
    legitimately empty and the opener is all we have."""
    try:
        query = session.ledger.query
    except Exception:
        query = ""
    if isinstance(query, str) and query.strip():
        return query
    return text


def _search(deps: Deps, query: str, limit: int) -> list[Candidate]:
    index = getattr(deps, "index", None)
    if index is None or not query.strip():
        return []
    depth = max(POOL_SIZE, limit)
    try:
        pool = index.search(query, depth)
    except Exception:
        return []
    return list(pool) if isinstance(pool, (list, tuple)) else []


def _partition(session: Session, pool: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    registry = getattr(session, "shown", None)
    if registry is None:
        return list(pool), []
    try:
        fresh, seen = registry.partition(pool)
    except Exception:
        return list(pool), []
    if not isinstance(fresh, list) or not isinstance(seen, list):
        return list(pool), []
    if len(fresh) + len(seen) != len(pool):
        # Not a partition. Something dropped a candidate, and a dropped
        # candidate is a retrieval change wearing an ordering change's clothes.
        return list(pool), []
    return fresh, seen


# --------------------------------------------------------------------------
# 13-15  the window. All three are order-only.
# --------------------------------------------------------------------------

def _hydrate(deps: Deps, window: list[Candidate]) -> list[Candidate]:
    index = getattr(deps, "index", None)
    if index is None or not window:
        return window
    try:
        hydrated = index.hydrate(window)
    except Exception:
        return window
    return _same_multiset_or_original(window, hydrated)


def _rerank(deps: Deps, query: str, window: list[Candidate]) -> list[Candidate]:
    reranker = getattr(deps, "reranker", None)
    if reranker is None or not window:
        return window
    try:
        reranked = rerank.safe_rerank(reranker, query, window)
    except Exception:
        return window
    return _same_multiset_or_original(window, reranked)


def _segments_for_turn(session: Session, decode: Decode) -> tuple[str, ...]:
    """The disclosed-segment list overlap.gate() and _llm_escalate() both key
    on -- factored out so the two stay in lockstep by construction rather than
    by two copies of the same fallback order agreeing."""
    segments: tuple[str, ...] = ()
    try:
        recorded = session.ledger.segments
        if recorded:
            segments = tuple(item for item in recorded if isinstance(item, str))
    except Exception:
        segments = ()
    if not segments:
        segments = _segments_of(decode)
    return segments


def _gate(session: Session, decode: Decode, window: list[Candidate]) -> list[Candidate]:
    if not window:
        return window
    segments = _segments_for_turn(session, decode)
    try:
        gated = overlap.gate(window, segments)
    except Exception:
        return window
    return _same_multiset_or_original(window, gated)


def _llm_escalate(deps: Deps, query: str, session: Session, decode: Decode,
                   window: list[Candidate]) -> tuple[list[Candidate], int, int]:
    """LLMRR: hand the shortlist to a language model ONLY when overlap.gate
    found zero literal overlap ("vague" -- keyword matching has gone blind).
    Any literal overlap at all means BM25 + the cross-encoder are already
    doing the right thing, so the LLM is skipped -- report.md's design of
    record, section 7. INERT (skipped) whenever deps.llm_reranker is None,
    which is every turn until a model is chosen and enabled (docs/todo.md
    item 3). Order-only and safe_rerank-guarded, identically to the
    cross-encoder at step 14 -- a broken or hallucinating model costs BM25's
    order and nothing else.
    """
    if not window:
        return window, 0, 0
    llm = getattr(deps, "llm_reranker", None)
    if llm is None:
        return window, 0, 0
    segments = _segments_for_turn(session, decode)
    if not segments:
        return window, 0, 0
    try:
        report = overlap.measure(window, segments)
    except Exception:
        return window, 0, 0
    if report.segments == 0 or report.rate > 0.0:
        return window, 0, 0
    try:
        reranked = rerank.safe_rerank(llm, query, window)
    except Exception:
        return window, 0, 0
    usage_fn = getattr(llm, "usage", None)
    prompt_tokens, completion_tokens = 0, 0
    if callable(usage_fn):
        try:
            counted = usage_fn()
            prompt_tokens = int(counted[0])
            completion_tokens = int(counted[1])
        except Exception:
            prompt_tokens, completion_tokens = 0, 0
    return _same_multiset_or_original(window, reranked), prompt_tokens, completion_tokens


def _same_multiset_or_original(original: list[Candidate], produced: object) -> list[Candidate]:
    """An order-only stage must hand back a permutation of its input.

    Checked here rather than trusted, because the failure is invisible: a stage
    that quietly drops one candidate has performed retrieval, and the only
    symptom is a slightly worse score.
    """
    try:
        if not isinstance(produced, (list, tuple)):
            return original
        produced = list(produced)
        if len(produced) != len(original):
            return original
        if sorted(parent_asin_of(item) for item in produced) != \
           sorted(parent_asin_of(item) for item in original):
            return original
        return produced
    except Exception:
        return original


# --------------------------------------------------------------------------
# 16-17  the picks.
# --------------------------------------------------------------------------

def _assemble(window: Sequence[Candidate], rest: Sequence[Candidate],
              seen: Sequence[Candidate], limit: int) -> list[str]:
    """window + rest-of-fresh + seen, cut to `limit`. Never short, never a dup.

    `seen` is the backfill and the reason this is a partition and not a filter:
    when the fresh pool is smaller than limit, the proven-wrong products come
    back to fill the list out. Re-showing one costs exactly nothing.
    """
    picks: list[str] = []
    chosen: set[str] = set()
    if limit <= 0:
        return picks
    for group in (window, rest, seen):
        for item in group:
            if len(picks) >= limit:
                return picks
            parent_asin = parent_asin_of(item)
            if not parent_asin or parent_asin in chosen:
                continue
            chosen.add(parent_asin)
            picks.append(parent_asin)
    return picks


def _record(session: Session, picks: Sequence[str]) -> None:
    registry = getattr(session, "shown", None)
    if registry is None:
        return
    _call(registry.record, picks)


# --------------------------------------------------------------------------
# 18-19  the ask, and the reply.
# --------------------------------------------------------------------------

def _choose_attribute(session: Session, turn_number: int) -> str:
    """Never None, never "other".

    Anything outside ALLOWED_ATTRIBUTES is silently rewritten to `other` by the
    evaluator, which switches on an exploit we have permanently declined. So the
    return value is validated here, not assumed.
    """
    attribute: object = None
    asks = getattr(session, "asks", None)
    if asks is not None:
        try:
            attribute = askyield.next_attribute(asks)
        except Exception:
            attribute = None
    if not _valid_ask(attribute):
        attribute = _fallback_attribute(session, turn_number)
    if asks is not None:
        _call(asks.mark_asked, attribute)
        _set(asks, "last_ask", attribute)
    return str(attribute)


def _valid_ask(attribute: object) -> bool:
    return (isinstance(attribute, str)
            and attribute in ALLOWED_ATTRIBUTES
            and attribute not in FORBIDDEN_ASK)


def _fallback_attribute(session: object, turn_number: int) -> str:
    """The fixed schedule, minus anything retired. Used only when the ask policy
    hands back something unusable -- but it must still be a sensible ask, since
    a wasted ask is a wasted turn."""
    asked: tuple = ()
    retired: tuple = ()
    asks = getattr(session, "asks", None)
    try:
        asked = tuple(getattr(asks, "asked", ()) or ())
    except Exception:
        asked = ()
    try:
        retired = tuple(getattr(asks, "retired", ()) or ())
    except Exception:
        retired = ()
    for attribute in FIXED_SCHEDULE:
        if attribute not in asked and attribute not in retired:
            return attribute
    for attribute in FIXED_SCHEDULE:
        if attribute not in retired:
            return attribute
    index = (turn_number - 1) % len(FIXED_SCHEDULE) if isinstance(turn_number, int) else 0
    return FIXED_SCHEDULE[max(0, index)]


def _message_for(attribute: str) -> str:
    """Customer-facing prose. The simulator reads ask_attribute, never this, so
    it costs nothing -- which is not a reason for it to read like a stub."""
    question = ASK_TEMPLATES.get(attribute, "")
    return f"{CLOSING_MESSAGE} {question}".strip() if question else CLOSING_MESSAGE


def _degraded_plan(session: object) -> TurnPlan:
    """A valid turn with no recommendations. The floor, not a target."""
    try:
        attribute = _fallback_attribute(session, 1)
    except Exception:
        attribute = FIXED_SCHEDULE[0]
    return TurnPlan(message=CLOSING_MESSAGE, ask_attribute=attribute, parent_asins=())


# --------------------------------------------------------------------------
# Guarded plumbing. Every collaborator here belongs to another workstream.
# --------------------------------------------------------------------------

def _call(function: object, *args: object) -> None:
    try:
        function(*args)
    except Exception:
        pass


def _set(target: object, name: str, value: object) -> None:
    try:
        setattr(target, name, value)
    except Exception:
        pass


def _truthy(target: object, name: str) -> bool:
    try:
        return bool(getattr(target, name, False))
    except Exception:
        return False
