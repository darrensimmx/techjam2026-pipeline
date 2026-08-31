"""Label which rung of the ask ladder fired, honestly.

The rung is the one thing in the backend render that cannot be observed: the
pipeline sees an attribute come back from ``askyield.next_attribute`` and never
learns why. But it can be RE-DERIVED faithfully, and the reason is a design
property the code states outright at ``src/askpolicy.py:107-112``:

    Consuming the burn happens HERE [in mark_asked], not in next_attribute(),
    so that selection stays free of side effects.

``_select`` (askpolicy.py:211-268) only reads: ``_str_set``, ``_int_or``, and
duck-typed calls to ``pending_reask`` / ``overflow_candidates``. Re-running its
predicates against a snapshot taken before the real call therefore reproduces
the real decision -- it is a re-derivation, not a guess.

Two things keep it honest:

  - We reconstruct a REAL ``AskState`` and call its REAL methods, and import the
    REAL constants. Only the ~12 lines of branch ORDERING below are restated,
    so drift has one small surface.
  - The result is cross-checked. ``rung_agrees`` compares the attribute this
    module predicts against the attribute the policy actually returned. On
    disagreement the backend prints a MISMATCH block saying the label is wrong
    and the policy row is the truth, and a CI test fails the build. A derived
    label that can be silently wrong would be worse than no label at all.
"""
from __future__ import annotations

FIXED_SCHEDULE_RUNG = "1-fixed-schedule"
PENDING_REASK = "2i-pending-reask"
OVERFLOW = "2ii-overflow"
HEDGE = "2iii-hedge"
FIRST_LIVE = "2v-first-live"
LAST_RESORT = "2v-last-resort"
DEFAULT = "D-default"
PIPELINE_FALLBACK = "F-pipeline-fallback"
UNKNOWN = "?-unknown"

RUNG_TITLES = {
    FIXED_SCHEDULE_RUNG: "FIXED SCHEDULE owns turns 1-7",
    PENDING_REASK: "PENDING RE-ASK -- a burned ask outranks everything",
    OVERFLOW: "OVERFLOW -- a bucket that yielded a full pair may be truncated",
    HEDGE: "HEDGE -- brand/category, priced honestly at zero",
    FIRST_LIVE: "FIRST LIVE -- first attribute not yet retired",
    LAST_RESORT: "LAST RESORT -- everything retired, we still may not send null",
    DEFAULT: "DEFAULT -- _select returned something invalid",
    PIPELINE_FALLBACK: "PIPELINE FALLBACK -- _valid_ask rejected the policy's choice",
    UNKNOWN: "UNKNOWN -- no snapshot available",
}


def _rebuild(snapshot: dict):
    """Turn a snapshot dict back into a real AskState.

    Reusing the real class means ``pending_reask()`` and
    ``overflow_candidates()`` below are the shipped implementations, not copies.
    """
    from src.askpolicy import AskState

    return AskState(
        asked=list(snapshot.get("asked") or []),
        retired=set(snapshot.get("retired") or ()),
        yield_counts=dict(snapshot.get("yield_counts") or {}),
        yield_order=list(snapshot.get("yield_order") or []),
        burned=snapshot.get("burned"),
        burned_reasked=bool(snapshot.get("burned_reasked", False)),
        last_ask=snapshot.get("last_ask"),
        turn=int(snapshot.get("turn", 0) or 0),
        disclosed_count=int(snapshot.get("disclosed_count", 0) or 0),
    )


def derive(snapshot: "dict | None") -> tuple:
    """Return ``(rung_id, reason, predicted_attribute)`` for a pre-call snapshot.

    Mirrors the ladder in ``src/askpolicy.py::_select`` (:211-268) in order.
    Never raises: an unusable snapshot yields ``UNKNOWN`` rather than a wrong
    confident label.
    """
    if not isinstance(snapshot, dict):
        return UNKNOWN, "no AskState snapshot was captured for this turn", None

    try:
        from src.askpolicy import ASKABLE, _is_valid_ask
        from src.types import CARD_CAPACITY, FIXED_SCHEDULE, HEDGE_ORDER

        state = _rebuild(snapshot)
        asked = [a for a in state.asked if isinstance(a, str)]
        retired = set(str(r) for r in state.retired)
        turn = state.turn

        # 1. The fixed schedule owns turns 1-7 (askpolicy.py:220-226).
        if 1 <= turn <= len(FIXED_SCHEDULE):
            scheduled = FIXED_SCHEDULE[turn - 1]
            if scheduled not in retired and scheduled not in asked:
                return (FIXED_SCHEDULE_RUNG,
                        'turn %d in [1,%d]; FIXED_SCHEDULE[%d]="%s"; not retired; not asked'
                        % (turn, len(FIXED_SCHEDULE), turn - 1, scheduled),
                        scheduled)
            why = "retired" if scheduled in retired else "already asked"
            prefix = ('turn %d: FIXED_SCHEDULE[%d]="%s" is %s, so the free-turn '
                      "ladder runs -- " % (turn, turn - 1, scheduled, why))
        else:
            prefix = "turn %d is past the fixed schedule -- " % (turn,)

        # 2.i  A burned ask outranks everything (askpolicy.py:234-236).
        pending = state.pending_reask()
        if _is_valid_ask(pending):
            return (PENDING_REASK,
                    prefix + 'burned ask "%s" was never read, re-ask it' % (pending,),
                    pending)

        # 2.ii Overflow, while the card may still hold something
        #      (askpolicy.py:243-251).
        if state.disclosed_count < CARD_CAPACITY:
            overflow = [n for n in state.overflow_candidates() if n not in retired]
            choice = next((n for n in overflow if _is_valid_ask(n)), None)
            if choice is not None:
                return (OVERFLOW,
                        prefix + 'disclosed %d < CARD_CAPACITY %d; "%s" yielded a '
                        "full pair and may have been truncated"
                        % (state.disclosed_count, CARD_CAPACITY, choice),
                        choice)
            overflow_note = ("disclosed %d < %d but no untruncated candidate; "
                             % (state.disclosed_count, CARD_CAPACITY))
        else:
            overflow_note = ("disclosed %d >= CARD_CAPACITY %d, card is full; "
                             % (state.disclosed_count, CARD_CAPACITY))

        # 2.iii The hedge (askpolicy.py:259-261).
        hedge = next((n for n in HEDGE_ORDER if n not in asked and _is_valid_ask(n)), None)
        if hedge is not None:
            return (HEDGE,
                    prefix + overflow_note + 'hedge "%s" not yet asked '
                    "(worth zero under this simulator, asked anyway)" % (hedge,),
                    hedge)

        # 2.v  First live, then unconditional (askpolicy.py:265-268).
        live = next((n for n in ASKABLE if n not in retired and _is_valid_ask(n)), None)
        if live is not None:
            return (FIRST_LIVE,
                    prefix + overflow_note + 'hedge exhausted; first non-retired '
                    'ASKABLE is "%s"' % (live,),
                    live)

        last = next((n for n in ASKABLE if _is_valid_ask(n)), None)
        if last is not None:
            return (LAST_RESORT,
                    prefix + "every ASKABLE attribute is retired; sending "
                    '"%s" anyway because null is never allowed' % (last,),
                    last)

        return DEFAULT, prefix + "_select found nothing valid", None
    except Exception as exc:
        return UNKNOWN, "rung derivation failed: %r" % (exc,), None


def title(rung: str) -> str:
    return RUNG_TITLES.get(rung, rung)
