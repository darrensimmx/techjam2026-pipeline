"""The graded Agent: Phase 1 foundational pipeline.

Fixed six-attribute schedule + accumulated-constraint BM25 retrieval. Offline,
local, no network calls, no LLM. respond() never raises — see
`decisions/standing-findings.md` -> "Don't score zero" in the planning repo
for why: the official evaluator swallows any exception into a silent zero.

P1 (offline safety) additionally guards __init__ and reset(), which the
evaluator does NOT wrap (local_evaluator.py:306 and :228 respectively) — a
raise in either crashes the entire run rather than one session — and validates
every outgoing payload against the turn_response contract, because a
schema-invalid dict is zeroed just as silently as an exception.
"""
from __future__ import annotations

from pathlib import Path

from starter.ledger import SessionState
from starter.retrieval import Bm25Index
from starter.scheduler import next_attribute

_ASK_TEMPLATES = {
    "material": "Do you have a material preference?",
    "feature": "Is there a specific feature that matters most to you?",
    "color": "Do you have a color preference?",
    "style": "What style are you looking for?",
    "size": "Do you have a size in mind?",
    "use_case": "What will you mainly use this for?",
}
_CLOSING_MESSAGE = "Here are the closest matches I found so far."

# turn_response contract, docs/agent_api_contract.json.
_ALLOWED_ATTRIBUTES = frozenset(
    ("category", "material", "color", "size", "style", "brand",
     "budget", "feature", "use_case", "other")
)
_MAX_RECOMMENDATIONS = 100
_DEFAULT_TOP_K = 10


def _empty_response() -> dict:
    """A fresh schema-valid response — built per call, not copied from a module
    constant, whose nested `usage` dict a shallow copy would leave shared."""
    return {
        "message": "",
        "ask_attribute": None,
        "recommendations": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def _limit(top_k: object) -> int:
    """Clamp top_k into [0, maxItems]. A negative or non-integer LIMIT makes
    SQLite return the whole catalog (47,602 rows when probed) or raise — the
    first silently violates maxItems: 100, the second silently zeroes a turn."""
    if isinstance(top_k, int) and not isinstance(top_k, bool):
        return max(0, min(top_k, _MAX_RECOMMENDATIONS))
    return _DEFAULT_TOP_K


def _validated(payload: object) -> dict:
    """Coerce any payload into a value that validates against turn_response.
    Every field falls back to its schema-valid empty form rather than passing
    an invalid value through to the evaluator's isinstance check."""
    response = _empty_response()
    if not isinstance(payload, dict):
        return response
    message = payload.get("message")
    if isinstance(message, str):
        response["message"] = message
    attribute = payload.get("ask_attribute")
    if isinstance(attribute, str) and attribute in _ALLOWED_ATTRIBUTES:
        response["ask_attribute"] = attribute
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list):
        clean: list[dict] = []
        for item in recommendations:
            parent_asin = item.get("parent_asin") if isinstance(item, dict) else None
            if isinstance(parent_asin, str) and parent_asin:
                clean.append({"parent_asin": parent_asin})
                if len(clean) >= _MAX_RECOMMENDATIONS:
                    break
        response["recommendations"] = clean
    usage = payload.get("usage")
    if isinstance(usage, dict) and all(
        isinstance(usage.get(field), int) and usage[field] >= 0
        for field in ("prompt_tokens", "completion_tokens")
    ):
        response["usage"] = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }
    return response


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._sessions: dict[str, SessionState] = {}
        # Unwrapped by the evaluator (local_evaluator.py:306): a raise here ends
        # the run before session 1. Degrade to a null index instead.
        try:
            self._index: Bm25Index | None = Bm25Index(catalog_path)
        except Exception:
            self._index = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Unwrapped by the evaluator (local_evaluator.py:228): a raise here ends
        # the whole run, not one session. str() also survives an unhashable id.
        try:
            self._sessions[str(session_id)] = SessionState()
        except Exception:
            pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return _validated(self._respond(session_id, user_message, top_k))
        except Exception:
            return _empty_response()

    def _respond(self, session_id: str, user_message: str, top_k: int) -> dict:
        # Defensive, not per contract: reset is always called first per the
        # official harness, but respond() must never raise regardless.
        state = self._sessions.setdefault(str(session_id), SessionState())
        message_text = user_message if isinstance(user_message, str) else ""
        state.record_message(message_text)

        query = state.disclosed_constraints or message_text
        matches = self._index.search(query, _limit(top_k)) if self._index else []
        recommendations = [{"parent_asin": parent_asin} for parent_asin in matches]

        attribute = next_attribute(state.asked_attributes)
        if attribute is not None:
            state.mark_asked(attribute)

        return {
            "message": _ASK_TEMPLATES.get(attribute, _CLOSING_MESSAGE),
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
