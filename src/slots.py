"""Typed slot state and the contradiction check.  [WS-C OWNS]

Slot state is used ONLY to decide what to ask next. It must never touch
retrieval. That separation is the safety property: a parsing bug can corrupt
scheduling but can never corrupt search -- which is the concrete mechanism
behind structured slot parsing measuring +0.000000.

tests/test_src_layering.py asserts this module is not imported by retrieval or
overlap, so the property is checkable rather than aspirational. Nothing here
imports src.retrieval or src.overlap, and nothing here returns anything that
belongs in a search query -- the only exports are an attribute LABEL and a
boolean.

Never raises. Every entry point takes `object` in spirit if not in annotation:
a wrong type degrades to the inert answer ("feature", False, None) rather than
propagating into respond(), where the evaluator would zero the turn in silence.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# classify_local: a line-for-line mirror of the evaluator's
# classify_constraint()  (evaluator/local_evaluator.py:137-151).
#
# It is copied rather than imported because importing evaluator.local_evaluator
# would pull `from starter.agent import Agent` into the graded path. The copy is
# pinned by tests/test_src_slots.py, which imports the real function and asserts
# equality on a table of strings -- so a drift in the vendored harness fails a
# test instead of silently desynchronising the schedule.
#
# Two properties of the original that are easy to "improve" by accident and
# must not be:
#   - the MATERIALS test is a plain substring test, not a word-boundary one, so
#     "silk" fires inside "silky" and "wool" inside "woolen";
#   - the branch order is load-bearing. "budget around $40 for a black belt"
#     is `budget`, not `color`, because budget is tested first.
# --------------------------------------------------------------------------

MATERIALS: tuple[str, ...] = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
)
_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")
_COLOR_WORDS: tuple[str, ...] = ("color", "black", "white", "blue", "red", "pink", "green")
_SIZE_WORDS: tuple[str, ...] = ("size", "sizing", "width", "wide", "narrow")
_STYLE_WORDS: tuple[str, ...] = ("department", "style", "fit", "sleeve", "neck")
_USE_CASE_WORDS: tuple[str, ...] = ("hiking", "running", "gym", "winter", "outdoor", "work")

# The seven labels classify_constraint() can return, and the only seven this
# function may return. `other` is not among them and never will be.
LABELS: frozenset[str] = frozenset(
    ("budget", "material", "color", "size", "style", "use_case", "feature")
)
_FALLBACK_LABEL = "feature"


def classify_local(value: str) -> str:
    """Mirror of the evaluator's classify_constraint(). Returns one of seven."""
    if not isinstance(value, str):
        return _FALLBACK_LABEL
    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in _COLOR_WORDS):
        return "color"
    if any(word in lowered for word in _SIZE_WORDS):
        return "size"
    if any(word in lowered for word in _STYLE_WORDS):
        return "style"
    if any(word in lowered for word in _USE_CASE_WORDS):
        return "use_case"
    return _FALLBACK_LABEL


def _norm(value: object) -> str:
    """Casefolded, stripped. The whole of the DIFF's normalisation."""
    return value.strip().casefold() if isinstance(value, str) else ""


def _diff(left: object, right: object) -> bool:
    """The contradiction test. Two non-empty values that are not the same
    string, casefolded and stripped, contradict. That is the entire rule --
    no model, no threshold, no synonym table, nothing to tune."""
    first, second = _norm(left), _norm(right)
    if not first or not second:
        return False
    return first != second


class SlotState:
    def __init__(self) -> None:
        self._slots: dict[str, str] = {}

    def fill(self, attribute: str, value: str) -> bool:
        """Set a slot. Returns True iff this CONTRADICTS an existing value.

        The contradiction check (DIFF) is a plain casefolded string comparison.
        No model, no threshold, nothing to tune.
        """
        if not isinstance(attribute, str) or not isinstance(value, str):
            return False
        name = attribute.strip()
        text = value.strip()
        if not name or not text:
            return False
        previous = self._slots.get(name)
        # The newest disclosure wins the slot either way: the customer's most
        # recent word is the one the next ask should be planned against. The
        # return value reports the collision; it does not veto the write.
        self._slots[name] = text
        return _diff(previous, text)

    def get(self, attribute: str) -> str | None:
        if not isinstance(attribute, str):
            return None
        return self._slots.get(attribute.strip())

    def clear(self, attribute: str) -> None:
        """Clear ONE slot. The ledger is never touched by this."""
        if isinstance(attribute, str):
            self._slots.pop(attribute.strip(), None)

    def filled(self) -> tuple[str, ...]:
        return tuple(self._slots)

    def as_dict(self) -> dict[str, str]:
        return dict(self._slots)


def apply_override(slots: SlotState, new_value: str) -> str | None:
    """G5: classify -> DIFF -> clear the conflicting slot. Returns its name.

    Worth at most 0.0078 -- it changes WHEN a hit is recorded, never what rank
    is returned. Built for coverage and honesty, not for score.

    It clears the SLOT and only the slot. The ledger has no clear() at all --
    the override's old_value and new_value are both manufactured from the same
    target listing, so the abandoned preference still describes the target and
    erasing it would cost retrieval a real term.
    """
    if not isinstance(slots, SlotState) or not isinstance(new_value, str):
        return None
    text = new_value.strip()
    if not text:
        return None
    attribute = classify_local(text)
    existing = slots.get(attribute)
    if existing is None or not _diff(existing, text):
        return None
    slots.clear(attribute)
    return attribute
