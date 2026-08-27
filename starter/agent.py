"""The graded Agent: Phase 1 foundational pipeline.

Fixed six-attribute schedule + accumulated-constraint BM25 retrieval. Offline,
local, no network calls, no LLM. respond() never raises — see
`decisions/standing-findings.md` -> "Don't score zero" in the planning repo
for why: the official evaluator swallows any exception into a silent zero.
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
_EMPTY_RESPONSE = {
    "message": "",
    "ask_attribute": None,
    "recommendations": [],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._index = Bm25Index(catalog_path)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState()

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, top_k)
        except Exception:
            return dict(_EMPTY_RESPONSE)

    def _respond(self, session_id: str, user_message: str, top_k: int) -> dict:
        # Defensive, not per contract: reset is always called first per the
        # official harness, but respond() must never raise regardless.
        state = self._sessions.setdefault(session_id, SessionState())
        state.record_message(user_message)

        query = state.disclosed_constraints or user_message or ""
        matches = self._index.search(query, top_k)
        recommendations = [{"parent_asin": parent_asin} for parent_asin in matches]

        attribute = next_attribute(state.asked_attributes)
        if attribute is not None:
            state.mark_asked(attribute)
            message = _ASK_TEMPLATES[attribute]
        else:
            message = _CLOSING_MESSAGE

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
