"""Cross-encoder rerank seam.  [WS-D OWNS]

Re-sorts the top RERANK_WINDOW we have NOT yet shown, so the window follows the
never-repeat frontier instead of going dark around turn 6. Worth +0.047 with the
CI excluding zero, bundled and offline -- no network at any point.

INERT until a checkpoint is chosen. Two axes are still open: which checkpoint,
and the ~1.2 s/turn cost against a per-turn timeout the organizers have never
published. See docs/todo.md.

If it fails to load or throws, we keep BM25's order and nothing breaks. The
worst outcome is exactly the behaviour without it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence as _RuntimeSequence
from typing import Sequence

from src.types import Candidate, Reranker


class NullReranker:
    """The identity pass-through. Ships today."""

    name = "null"

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        return list(candidates)


def _load_checkpoint(timeout_s: float) -> Reranker | None:
    """The single seam a real cross-encoder plugs into.

    Deliberately returns None. When a checkpoint is chosen this is the only
    function that changes: it loads the bundled model, wraps it behind the
    Reranker protocol, and returns it -- or returns None and we ship BM25's
    order, which is what we ship today anyway.
    """
    return None


def load_reranker(enabled: bool = False, timeout_s: float = 1.2) -> Reranker:
    """Returns NullReranker unless a real checkpoint is chosen AND loads."""
    if not enabled:
        return NullReranker()
    try:
        loaded = _load_checkpoint(float(timeout_s))
    except Exception:
        return NullReranker()
    if loaded is None or not callable(getattr(loaded, "rerank", None)):
        return NullReranker()
    return loaded


def _fingerprint(candidates: Sequence[Candidate]) -> Counter:
    """The multiset of parent_asins. A sentinel object for anything that has no
    parent_asin, so a malformed item can never accidentally compare equal."""
    counts: Counter = Counter()
    for candidate in candidates:
        parent_asin = getattr(candidate, "parent_asin", None)
        counts[parent_asin if isinstance(parent_asin, str) else object()] += 1
    return counts


def safe_rerank(reranker: Reranker, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
    """Run a reranker, returning the input unchanged on ANY problem.

    The permutation check is the load-bearing half: the result must be the same
    multiset of parent_asins as the input. A broken reranker must not be able to
    drop a candidate or invent one -- that would be a retrieval change wearing
    an ordering change's clothes.

    try/except alone is not enough. A reranker that returns nine of ten
    candidates has not thrown; it has silently shortened the pool, and the pool
    is the recall floor. So the result is DISCARDED unless it is a genuine
    permutation of the input.
    """
    try:
        original = list(candidates)
    except Exception:
        return []
    if not original or reranker is None:
        return original
    rerank = getattr(reranker, "rerank", None)
    if not callable(rerank):
        return original

    try:
        result = rerank(query if isinstance(query, str) else "", original)
    except Exception:
        return original
    if isinstance(result, (str, bytes)) or not isinstance(result, _RuntimeSequence):
        return original
    try:
        reordered = list(result)
    except Exception:
        return original

    if len(reordered) != len(original):
        return original
    if _fingerprint(reordered) != _fingerprint(original):
        return original
    return reordered
