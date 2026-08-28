"""Layer 1: fixed six-attribute ask order.

The clean six (material, feature, color, style, size, use_case), not "other" —
see the planning repo's standing findings for why "other" is a rejected
short-circuit despite scoring marginally higher. Ask-yield (Phase 2) replaces
this with a dynamic order behind the same `next_attribute` interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for typing — keeps the scheduler runtime-decoupled
    from starter.ledger import SessionState

FIXED_SCHEDULE = ("material", "feature", "color", "style", "size", "use_case")


def next_attribute(state: SessionState) -> str | None:
    """The next attribute not yet asked this session, or None once exhausted.

    An attribute counts as asked the moment it's asked about, regardless of
    whether the customer answers it or declines (boundary case) — either way
    it must never be asked again.

    Takes the whole `state`, not just the asked list, so Phase 2 (ask-yield)
    can swap its body in behind this exact signature. Layer 2 needs two fields
    this layer ignores — `state.retired` (attributes that yielded nothing and
    must never be re-asked) and `state.yield_seen` (observed yield per
    attribute) — neither of which exists on SessionState yet. Add them there
    when building Phase 2; do not widen this signature again.
    """
    asked_set = set(state.asked_attributes)
    for attribute in FIXED_SCHEDULE:
        if attribute not in asked_set:
            return attribute
    return None
