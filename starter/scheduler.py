"""Layer 1: fixed six-attribute ask order.

The clean six (material, feature, color, style, size, use_case), not "other" —
see the planning repo's standing findings for why "other" is a rejected
short-circuit despite scoring marginally higher. Ask-yield (Phase 2) replaces
this with a dynamic order behind the same `next_attribute` interface.
"""
from __future__ import annotations

FIXED_SCHEDULE = ("material", "feature", "color", "style", "size", "use_case")


def next_attribute(asked: list[str]) -> str | None:
    """The next attribute not yet asked this session, or None once exhausted.

    An attribute counts as asked the moment it's asked about, regardless of
    whether the customer answers it or declines (boundary case) — either way
    it must never be asked again.
    """
    asked_set = set(asked)
    for attribute in FIXED_SCHEDULE:
        if attribute not in asked_set:
            return attribute
    return None
