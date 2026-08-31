"""The constraint ledger -- append-only, and it IS the query.  [WS-B OWNS]

The most load-bearing thing in the system and deliberately the dumbest. Every
disclosed reply goes in verbatim; the concatenation of those raw strings is
what gets searched. Not an input to a query builder -- the query itself.

There is deliberately NO clear/remove/replace/pop. "Never erase, not even on
intent override" is enforced by the ABSENCE of an API, not by a comment: the
override's old_value and new_value are both generated from the same target
listing, so the abandoned preference still describes the target.

Concretely (local_evaluator.py:79-86): old_value is soft_preferences[-1] and
new_value is hard_constraints[0] of the SAME intent_card, which
materialize_hidden_fields() built out of the target product's own listing. And
old_value is never added to the evaluator's `disclosed` set, so it is not even
spent -- it can come back later in a disclosure. Erasing it on the override
turn would throw away terms that still describe the answer.

Two things that look like bugs and are not:

  * append() takes the reply VERBATIM, not a parsed slot value. Structured slot
    parsing measures +0.000000 here; the raw string is what moves the number.
  * `query` is recomputed on every access. A cached query is a query that can
    go stale, and a stale query is invisible -- it returns plausible results
    for a conversation that has moved on.

Never raises: a wrong type is a no-op, because respond() dropping a turn on a
TypeError scores exactly zero with no traceback (local_evaluator.py:239-244).
"""
from __future__ import annotations

from typing import Sequence


class ConstraintLedger:
    def __init__(self) -> None:
        self._entries: list[str] = []
        self._segments: list[str] = []

    def append(self, payload: str) -> None:
        """Append one reply verbatim. No-op on "" or whitespace-only.

        The no-op is how the three content-free frames (refusal, null_nudge,
        exhaustion) drop out of the query without anyone writing a filter:
        frames.decode() hands back payload="" for each of them, and "" does not
        survive this guard.

        No dedupe. A repeated reply is a repeated term, and repetition is
        signal to BM25, not noise to be cleaned up.
        """
        if not isinstance(payload, str) or not payload.strip():
            return
        self._entries.append(payload)

    def record_segments(self, segments: Sequence[str]) -> None:
        """Record the disclosed constraint strings. Dedupes, preserves order.

        Segments are the parsed view -- what the slots and the overlap
        instrument consume. They never feed the query; `_entries` does that.
        Dedupe is global across the session, because intent_card() can put the
        same cleaned string in both hard_constraints and soft_preferences
        (local_evaluator.py:69-70) and it can therefore be disclosed twice.
        """
        if isinstance(segments, str):          # a caller passing one bare string
            segments = (segments,)
        try:
            items = list(segments)
        except Exception:                      # not iterable at all
            return
        known = set(self._segments)
        for item in items:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in known:
                known.add(text)
                self._segments.append(text)

    @property
    def query(self) -> str:
        """Rebuilt from scratch every call, so a bug cannot leave a stale query."""
        return " ".join(self._entries).strip()

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def segments(self) -> tuple[str, ...]:
        return tuple(self._segments)

    def distinct_segment_count(self) -> int:
        """Distinct, not raw: one reply can legitimately carry the same string
        twice, and a raw count would read 'may be truncated' on a drained bucket."""
        return len(set(self._segments))

    def __len__(self) -> int:
        return len(self._entries)
