"""Layer 2 seam -- adaptive ask ordering.  [WS-C OWNS]

Swaps in behind the same next_attribute(state) signature the fixed schedule
uses. A swap-in, not a parallel system. The whole question-ordering band is
worth about 0.004, so this is judged on the design, not the delta -- and if it
ever regresses we fall straight back to the fixed schedule.

The body is out of scope for this build. This module is the single swap point:
nothing else changes when Layer 2 lands.

The guard is the whole contract of the seam, and it is the same shape every
optional layer in this system uses: its own try/except, a local fallback, and
never on the critical path. Three things can go wrong with an adaptive ordering
and all three land here rather than in respond():

  * it raises            -> caught, fixed schedule
  * it returns None      -> rejected, fixed schedule  (a null ask is never sent)
  * it returns a bad str -> rejected, fixed schedule

That last one is the dangerous one. Any value outside ALLOWED_ATTRIBUTES is
silently rewritten to `other` by the evaluator (local_evaluator.py:172-173), and
`other` bypasses the constraint filter entirely -- the declined exploit switches
itself on from a typo. So the validation is an allowlist test, not a blocklist.
"""
from __future__ import annotations

from src.askpolicy import AskState
from src.askpolicy import next_attribute as _fixed
from src.types import ALLOWED_ATTRIBUTES, FIXED_SCHEDULE, FORBIDDEN_ASK

ADAPTIVE_ENABLED: bool = False


def _adaptive(state: AskState) -> str | None:
    """Returns None today. Layer 2's body goes here and nowhere else."""
    return None


def _acceptable(attribute: object) -> bool:
    return (
        isinstance(attribute, str)
        and attribute in ALLOWED_ATTRIBUTES
        and attribute not in FORBIDDEN_ASK
    )


def next_attribute(state: AskState) -> str:
    """The ONLY entry point the pipeline calls. Never None, never raises."""
    if ADAPTIVE_ENABLED:
        try:
            choice = _adaptive(state)
        except Exception:
            choice = None
        if _acceptable(choice):
            return choice  # type: ignore[return-value]
    try:
        fallback = _fixed(state)
    except Exception:
        fallback = None
    return fallback if _acceptable(fallback) else FIXED_SCHEDULE[0]
