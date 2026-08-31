"""The verbatim-overlap gate.  [WS-D OWNS]

"Is the customer literally quoting the product listing?" A substring check over
text already sitting in memory. Free, deterministic, no model -- build it first.

It is also the INSTRUMENT: computing verbatim overlap per session is how the
94.5% claim gets measured, and it is the signal an LLM escalation would read to
decide whether keyword matching has gone blind.

Two hard constraints:
  - gate() is ORDER-ONLY. len(gate(x, s)) == len(x), always. It never filters.
  - gate() is a STABLE sort, so it composes with the reranker instead of
    overwriting it. Final key: (overlap desc, incoming rank asc).
"""
from __future__ import annotations

import re
from typing import Sequence

from src.types import Candidate, OverlapReport

_WHITESPACE = re.compile(r"\s+")

# A one-character variant matches essentially every listing, which would turn
# the gate into noise and the instrument into a flat 100%. Two is the floor.
MIN_VARIANT_LEN = 2


def normalise(text: str) -> str:
    """Casefold and collapse whitespace for substring comparison."""
    if not isinstance(text, str) or not text:
        return ""
    try:
        return _WHITESPACE.sub(" ", text).strip().casefold()
    except Exception:
        return ""


def variants(segment: str) -> tuple[str, ...]:
    '''Comparable forms of one disclosed segment.

    The evaluator manufactures segments like "color: brown" and
    "budget around $29.99" which never appear verbatim in a listing -- but the
    post-colon tail does. Raw, after-first-colon, and $-stripped.

    intent_card() (local_evaluator.py:52) is the source of the shapes: it emits
    _flatten_values() output ("key: value" out of `details`), a bare material
    token, "color: <c>", and "budget around $<price>". The listing itself is
    indexed as "key value" pairs, so the raw "color: brown" is absent from it
    while the tail "brown" is present -- which is the entire reason this
    function is not the identity.

    Returned forms are already normalised, deduped, and ordered
    raw-first-then-derived, so the caller can substring-test them directly.
    '''
    raw = normalise(segment)
    if not raw:
        return ()
    candidates = [raw]
    _, separator, tail = raw.partition(":")
    tail = tail.strip() if separator else ""
    if tail:
        candidates.append(tail)
    for form in tuple(candidates):
        if "$" in form:
            candidates.append(form.replace("$", "").strip())

    out: list[str] = []
    seen: set[str] = set()
    for form in candidates:
        if len(form) >= MIN_VARIANT_LEN and form not in seen:
            seen.add(form)
            out.append(form)
    return tuple(out)


def _prepare(segments: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Variants for each usable segment, computed once per turn rather than
    once per candidate -- gate() is 300 candidates wide."""
    if not segments:
        return ()
    prepared: list[tuple[str, ...]] = []
    try:
        iterator = list(segments)
    except Exception:
        return ()
    for segment in iterator:
        forms = variants(segment) if isinstance(segment, str) else ()
        if forms:
            prepared.append(forms)
    return tuple(prepared)


def _count(prepared: tuple[tuple[str, ...], ...], candidate_text: str) -> int:
    """Segments (not variants) present in this text. One hit per segment."""
    haystack = normalise(candidate_text)
    if not haystack or not prepared:
        return 0
    return sum(1 for forms in prepared if any(form in haystack for form in forms))


def overlap_count(segments: Sequence[str], candidate_text: str) -> int:
    """How many segments appear literally in this candidate's text.

    An unhydrated candidate (text == "") scores 0 rather than raising -- the
    two-phase split means most of the pool legitimately has no text yet.
    """
    try:
        return _count(_prepare(segments), candidate_text)
    except Exception:
        return 0


def gate(candidates: Sequence[Candidate], segments: Sequence[str]) -> list[Candidate]:
    """Stable sort by overlap, descending. NEVER drops an element.

    The explicit index tiebreak makes the stability structural rather than an
    inherited property of sorted(): equal-overlap candidates keep the order the
    reranker gave them, which is the only reason the two compose instead of the
    second silently erasing the first.
    """
    try:
        items = list(candidates)
    except Exception:
        return []
    try:
        prepared = _prepare(segments)
        if not prepared or not items:
            return items
        scored = [(-_count(prepared, getattr(item, "text", "")), index, item)
                  for index, item in enumerate(items)]
        scored.sort(key=lambda entry: (entry[0], entry[1]))
        return [entry[2] for entry in scored]
    except Exception:
        return items


def measure(candidates: Sequence[Candidate], segments: Sequence[str]) -> OverlapReport:
    """The readout. Observation only -- nothing downstream reads this to filter.

    `matched` counts a segment once if it appears in ANY candidate's text, so
    `rate` is "how much of what the customer literally said is present anywhere
    in the pool" -- the pool-level form of the 94.5% claim. `top_overlap` is the
    same count for the highest-ranked candidate alone, which is the number that
    moves when retrieval goes blind.
    """
    try:
        items = list(candidates)
        prepared = _prepare(segments)
        total = len(prepared)
        if total == 0:
            return OverlapReport()
        texts = [normalise(getattr(item, "text", "")) for item in items]
        matched = sum(
            1 for forms in prepared
            if any(form in haystack for haystack in texts for form in forms)
        )
        top = _count(prepared, getattr(items[0], "text", "")) if items else 0
        return OverlapReport(
            segments=total,
            matched=matched,
            rate=round(matched / total, 6),
            top_overlap=top,
        )
    except Exception:
        return OverlapReport()
