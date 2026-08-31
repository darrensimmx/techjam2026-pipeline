"""Cross-encoder rerank seam.  [WS-D OWNS]

LIVE as of 1 Sep 2026: cross-encoder/ms-marco-MiniLM-L-6-v2.

Re-sorts the top RERANK_WINDOW we have NOT yet shown, so the window follows the
never-repeat frontier instead of going dark around turn 6. Worth +0.047 with the
CI excluding zero (bakeoff/results-part4.json, top-10/top-20 windows), bundled
and offline -- no network at any point once the checkpoint is vendored.

CHECKPOINT CHOSEN -- against a field, not unopposed. Four arms over 32 sessions,
each reranking BM25's top-50 (harness: bakeoff/part4_checkpoint_comparison.py,
cherry-picked from PR #21; the run itself was never archived to a results JSON):

    model          hit rate   tech score   time
    minilm-l6      84.38%     0.7229       343 s   <- chosen
    tinybert-l2    78.12%     0.6930       110 s
    mminilm-l12    68.75%     0.5832       852 s
    minilm-l112    65.62%     0.5668       804 s

Note what this is NOT: a speed/quality trade. The two slowest arms are also the
two worst, and the winner is second-cheapest -- so there is no bigger-is-better
frontier here to buy latency on. That closes the "which checkpoint -- never
compared" axis of docs/todo.md item 4 (axis 1), which this docstring previously
recorded as open because MiniLM-L6 was the only checkpoint ever measured.

Two caveats travel with the table and must not be dropped when it is quoted:
the harness carries SIX arms, and `distilroberta` and `zerank-1-small` are
absent from these four rows; and it is 32 sessions on one seed with no
confidence interval, unlike bakeoff/part4_rerank.py, which bootstraps. It
settles WHICH checkpoint, not HOW MUCH the rerank is worth.

Item 4's other axes stay open. The ~1.2 s/turn cost against a per-turn timeout
the organizers have never published (axis 2) is a Feasibility disclosure, not a
reason to hold this back -- and see _load_checkpoint below: that budget is
documented but not enforced anywhere in this module.

If it fails to load or throws, we keep BM25's order and nothing breaks. The
worst outcome is exactly the behaviour without it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence as _RuntimeSequence
from pathlib import Path
from typing import Sequence

from src.optional_deps import try_import
from src.types import Candidate, Reranker

# Where the vendored checkpoint lives. Gitignored, distributed as a release
# asset / fetched locally -- same convention as data/catalog.jsonl and
# data/models/potion-base-8m/. See docs/windows-dev-setup.md.
CE_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "ms-marco-MiniLM-L-6-v2"
CE_CHECKPOINT_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class NullReranker:
    """The identity pass-through. Ships when the checkpoint is unavailable."""

    name = "null"

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        return list(candidates)


class _CrossEncoderReranker:
    """Wraps a loaded sentence_transformers.CrossEncoder behind the Reranker
    protocol. Pointwise scoring, one (query, doc) pair at a time -- no
    cross-candidate attention, so this never degrades with pool size the way
    a listwise model would (report.md section 3)."""

    name = CE_CHECKPOINT_NAME

    def __init__(self, model: object) -> None:
        self._model = model

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        items = list(candidates)
        if not items:
            return items
        pairs = [(query, getattr(item, "text", "") or "") for item in items]
        scores = self._model.predict(pairs)  # type: ignore[attr-defined]
        order = sorted(range(len(items)), key=lambda i: float(scores[i]), reverse=True)
        return [items[i] for i in order]


def _load_checkpoint() -> Reranker | None:
    """The single seam a real cross-encoder plugs into.

    Raising here is fine and expected when the checkpoint isn't vendored on
    this machine: load_reranker() below wraps this call and falls back to
    NullReranker, exactly today's behaviour without it.

    NO TIMEOUT IS ENFORCED, here or on predict(). This function used to accept a
    `timeout_s` and ignore it, which read as a budget being applied when none
    was; the parameter is gone rather than left to imply a guarantee. The ~1.2 s
    per reranked turn is a MEASURED cost and a disclosure (README.md, and
    docs/todo.md item 4 axis 2 -- the organizers have never published a per-turn
    limit to enforce against). If a limit is ever published, enforcing it is a
    real change: sentence_transformers.predict() is a blocking call with no
    cancellation, so a wall-clock bound needs a worker, not a parameter.
    """
    sentence_transformers = try_import("sentence_transformers")
    if sentence_transformers is None:
        return None
    model = sentence_transformers.CrossEncoder(str(CE_MODEL_PATH))
    return _CrossEncoderReranker(model)


def load_reranker(enabled: bool = True) -> Reranker:
    """Returns NullReranker unless a real checkpoint is chosen AND loads.

    The `timeout_s=1.2` parameter this used to take was passed straight into
    `_load_checkpoint`, which ignored it; see that function for why no timeout
    is enforced and what enforcing one would actually take.
    """
    if not enabled:
        return NullReranker()
    try:
        loaded = _load_checkpoint()
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
