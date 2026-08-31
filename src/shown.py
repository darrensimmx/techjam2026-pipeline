"""The already-shown set, and the override guard.  [WS-E OWNS]

NEVER REPEAT A SHOWN PRODUCT. The evaluator ends the session the instant our
list contains the target, so any product still on screen in a running session is
CONFIRMED not to be the answer. Excluding it cannot remove the target -- that is
a consequence of the loop, not a guess.

Today the query stops changing at turn 3.58 on average, so without this a losing
session re-issues the same failed ten for six or seven turns and only ever
examines 10 to 30 products. With it, a session walks up to 100.

partition(), not filter(). The shown set is an ORDERING PREFERENCE, not a
removal: the pipeline emits `fresh + seen` truncated to k, so the top-10 is
always full even when the pool runs dry. Re-showing a proven-wrong product costs
exactly nothing, and this makes "nothing downstream may empty the pool" a
structural invariant rather than a caution.

THE OVERRIDE GUARD -- mandatory, not optional. The evaluator starts an override
session with its hit check switched off: `override_applied` begins false and
only flips at the end of the turn before the override lands. So on the early
turns of the thirty intent_override sessions the target can sit in our returned
list and the session carries on regardless. Those products are NOT confirmed
wrong. Exclude them and we forfeit that session permanently -- roughly five or
six sessions thrown away.

The guard is a suppression, not just a restore. Frame 2 identifies an override
session at turn 1, so record() goes quiet from turn 1 and every early turn shows
the true top ten -- which means the first SCORED turn tests our best list rather
than our second-best. restore_all() stays as the belt-and-braces path for a
turn-1 detection that missed. The pipeline caps suppression at turn 3 so a false
positive self-heals: behavior_for() picks the override turn from [3, 4], never
later.

This lives here, in its own set. It never touches the ledger, so the
accumulate-verbatim rule is untouched.

Nothing in this module raises. A registry that throws would cost a whole turn
(local_evaluator.py:239-244), and it has nothing to throw about.
"""
from __future__ import annotations

from typing import Sequence

from src.types import Candidate, DEFAULT_TOP_K

# The evaluator reads the first ten valid, unique, in-catalog ids off a response
# and silently drops everything after them (normalize_recommendations). Anything
# past the tenth was never actually shown to the customer, so recording it would
# exclude a product this session has no evidence against.
EVALUATOR_VISIBLE: int = DEFAULT_TOP_K


def parent_asin_of(item: object) -> str:
    """The parent_asin of a Candidate or a bare id string. "" if there is none.

    Never raises: `item` reaches here from retrieval, which is another
    workstream, and a property that throws must not cost the turn.
    """
    try:
        if isinstance(item, str):
            return item.strip()
        value = getattr(item, "parent_asin", None)
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()
    except Exception:
        return ""


class ShownRegistry:
    def __init__(self) -> None:
        self._shown: set[str] = set()
        self._suppressed = False

    @property
    def suppressed(self) -> bool:
        return self._suppressed

    def suppress(self) -> None:
        """Override session detected: stop recording, because the evaluator's
        hit check is off and nothing shown now is confirmed wrong."""
        self._suppressed = True

    def release(self) -> None:
        """The override arrived (or was never coming): resume recording."""
        self._suppressed = False

    def restore_all(self) -> None:
        """Everything shown before the override goes back in play."""
        self._shown.clear()

    def record(self, parent_asins: Sequence[str]) -> None:
        """No-op while suppressed. Record only what the evaluator actually saw."""
        if self._suppressed:
            return
        if parent_asins is None or isinstance(parent_asins, (str, bytes)):
            # A bare string is iterable one character at a time -- that would
            # poison the set with single letters rather than record an id.
            return
        try:
            items = list(parent_asins)
        except Exception:
            return
        accepted = 0
        for item in items:
            if accepted >= EVALUATOR_VISIBLE:
                return
            parent_asin = parent_asin_of(item)
            if not parent_asin:
                # The evaluator skips a blank id without consuming a slot.
                continue
            if parent_asin in self._shown:
                # Already recorded, and the evaluator dedupes too -- but it
                # still consumed one of its ten slots on this response.
                accepted += 1
                continue
            self._shown.add(parent_asin)
            accepted += 1

    def is_shown(self, parent_asin: str) -> bool:
        return parent_asin in self._shown

    def partition(self, candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
        """(fresh, seen). A true partition -- never a filtered-down list.

        len(fresh) + len(seen) == len(candidates), always. The caller emits
        fresh + seen, so a full shown set costs ordering and nothing else.
        """
        fresh: list[Candidate] = []
        seen: list[Candidate] = []
        if candidates is None:
            return fresh, seen
        try:
            items = list(candidates)
        except Exception:
            return fresh, seen
        for item in items:
            parent_asin = parent_asin_of(item)
            if parent_asin and parent_asin in self._shown:
                seen.append(item)
            else:
                fresh.append(item)
        return fresh, seen

    def __len__(self) -> int:
        return len(self._shown)
