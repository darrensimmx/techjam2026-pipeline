"""The graded Agent.  [WS-A OWNS]

THE RULE THAT GOVERNS THIS FILE. The evaluator swallows exceptions into a silent
zero, and a schema-invalid dict is zeroed just as silently:

  - respond() must never raise. It returns empty_response() on any exception.
    One throw costs one turn; throwing every turn makes TechnicalScore exactly
    0.00000.
  - __init__ and reset() are NOT wrapped by the evaluator
    (local_evaluator.py:306 and :228). A raise in either kills the ENTIRE RUN,
    not one session. Both are guarded here; keep them guarded.
  - Every outgoing payload passes through contract.validated().

Signatures are frozen. In particular `catalog_path` stays the FIRST POSITIONAL
parameter and keeps its default: the evaluator constructs `Agent(args.catalog)`
positionally at local_evaluator.py:306, and the submission harness may construct
`Agent()` with no argument at all. Both must work.

Construction is built out of three independently guarded pieces rather than one
big try, because they fail for different reasons and degrade differently:

  - the BM25 index      -- a missing/unreadable catalog is the single most
                           likely real-world failure. It sets _degraded and the
                           agent keeps answering (and keeps asking) without
                           retrieval, which is strictly better than no run.
  - the reranker        -- optional Layer 3. Null on failure; BM25's order
                           survives untouched.
  - the semantic decoder-- optional Tier 2. Null on failure; Tier 1 is unchanged.

An optional layer must never be able to take construction down, so each one owns
its own except clause: a single shared try would let a broken reranker cost us
the index that was already built.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.contract import clamp_top_k, empty_response, response_from_plan
from src.pipeline import Deps, run_turn
from src.rerank import load_reranker
from src.retrieval import Bm25Index
from src.semantic import load_semantic_decoder
from src.session import Session, new_session


def _guard(build: Callable[[], Any]) -> Any:
    """Run `build`, returning None on ANY failure. Never raises."""
    try:
        return build()
    except Exception:
        return None


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        # Assigned before anything that can fail, so every attribute this class
        # reads later exists even if the rest of __init__ degrades to nothing.
        self._sessions: dict[str, Session] = {}
        self._deps: Deps | None = None
        self._degraded: bool = True

        index = _guard(lambda: Bm25Index(catalog_path))
        reranker = _guard(load_reranker)
        semantic = _guard(load_semantic_decoder)

        deps = _guard(lambda: Deps(index=index, reranker=reranker, semantic=semantic))
        if deps is None:
            # Deps itself refused the parts. Fall back to an all-null Deps so the
            # pipeline still runs (ask-only); if even that fails, leave None and
            # let respond()'s except clause hold the line.
            index = None
            deps = _guard(Deps)
        self._deps = deps
        self._degraded = _is_degraded(index)

    @property
    def degraded(self) -> bool:
        """True when the index failed to build. A degraded agent still returns
        schema-valid responses with a real ask_attribute -- it just cannot
        retrieve. This is the silent-zero case, so it is worth being able to
        assert on."""
        return self._degraded

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Replace -- never merge -- the session for this id. Returns None on
        failure rather than raising: a raise here kills all 200 sessions.

        The pop is not redundant. If new_session() fails we must not be left
        holding the PREVIOUS shopper's session under this id -- session ids are
        random UUIDs with no identity behind them, so carrying a shown-set over
        would be mixing up two different people. Dropping it makes respond()
        lazily build a clean one instead.
        """
        try:
            key = str(session_id)
            self._sessions.pop(key, None)
            self._sessions[key] = new_session(session_id, user_profile)
        except Exception:
            pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            key = str(session_id)
            session = self._sessions.get(key)
            if session is None:
                # Defensive, not per contract: reset is always called first by
                # the official harness, but respond() must never raise regardless.
                session = new_session(session_id, {})
                self._sessions[key] = session
            plan = run_turn(session, user_message, turn, clamp_top_k(top_k), self._deps)
            return response_from_plan(plan)
        except Exception:
            return empty_response()


def _is_degraded(index: Any) -> bool:
    """Degraded when there is no index, or the index it built holds nothing.

    An empty index is the same outcome as no index -- zero recommendations every
    turn -- and it is what a present-but-empty or all-malformed catalog produces.
    Treating the two alike keeps `degraded` a statement about the agent's ability
    to retrieve rather than about which line of __init__ failed.
    """
    if index is None:
        return True
    try:
        return bool(index.is_empty())
    except Exception:
        return True
