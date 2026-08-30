"""Prove the phrase arm through the real evaluator, not the replay rig.

Every other arm in this bake-off only re-orders candidates, so `simulate.py`'s
validation covers them: the query trajectory is retrieval-independent, and that
was checked to six decimals on both ledgers.

The phrase arm is different. It changes the *query expression*, which is the one
thing the replay rig holds fixed, and it is the arm being recommended. So it gets
its own end-to-end run: a real `Agent` subclass, the stock `evaluate()`, no
replay. If this does not reproduce `followup_phrase.py`'s 0.751298, the
recommendation is built on an artifact of the rig.

The agent below is deliberately close to what a real implementation would be --
the ledger keeps the disclosed constraint strings alongside the raw text, and the
query is `phrase OR phrase OR ... OR unigram OR unigram`. That doubles as a
sketch of the change `starter/` would need.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.bm25_scores import ScoringIndex  # noqa: E402
from bakeoff.followup_phrase import _fts_phrase, constraints_from  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import _validated, _empty_response, _limit, _ASK_TEMPLATES, _CLOSING_MESSAGE  # noqa: E402
from starter.ledger import SessionState  # noqa: E402
from starter.retrieval import _terms  # noqa: E402
from starter.scheduler import next_attribute  # noqa: E402


class PhraseState(SessionState):
    """The shipped ledger, plus the decoded constraint strings it already sees."""

    def __init__(self) -> None:
        super().__init__()
        self.constraints: list[str] = []

    def record_message(self, message: str) -> None:
        before = self.disclosed_constraints
        super().record_message(message)
        if self.disclosed_constraints != before:
            self.constraints.extend(constraints_from(message.strip()))


class PhraseAgent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._sessions: dict[str, PhraseState] = {}
        try:
            self._index: ScoringIndex | None = ScoringIndex(catalog_path)
        except Exception:
            self._index = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        try:
            self._sessions[str(session_id)] = PhraseState()
        except Exception:
            pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return _validated(self._respond(session_id, user_message, top_k))
        except Exception:
            return _empty_response()

    def _respond(self, session_id: str, user_message: str, top_k: int) -> dict:
        state = self._sessions.setdefault(str(session_id), PhraseState())
        message_text = user_message if isinstance(user_message, str) else ""
        state.record_message(message_text)

        text = state.disclosed_constraints or message_text
        phrases = [p for p in (_fts_phrase(c) for c in state.constraints) if p]
        unigrams = [f'"{t}"' for t in list(dict.fromkeys(_terms(text)))[:40]]
        expression = " OR ".join(phrases + unigrams)

        matches: list[str] = []
        if self._index is not None and expression:
            try:
                matches = [a for a, _ in self._index.search_expression(expression, _limit(top_k))]
            except Exception:
                matches = []

        attribute = next_attribute(state)
        if attribute is not None:
            state.mark_asked(attribute)
        return {
            "message": _ASK_TEMPLATES.get(attribute, _CLOSING_MESSAGE),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": a} for a in matches],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def main() -> None:
    catalog = ROOT / "data" / "catalog.jsonl"
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, cats, products = catalog_index(catalog)
    t0 = time.time()
    result = evaluate(PhraseAgent(catalog), samples, ids, cats, products)
    expected = 0.751298
    got = result["recommended_technical_score"]
    print(f"end-to-end phrase agent, real evaluate(): {got}")
    print(f"  hit@10 {result['hit_rate_at_10']}  mrr {result['mrr']}  "
          f"mttc {result['mttc']}  eff {result['efficiency']}")
    print(f"  replay rig said: {expected}")
    print(f"  {'MATCH' if abs(got - expected) < 1e-9 else 'MISMATCH'}  ({time.time() - t0:.0f}s)")
    raise SystemExit(0 if abs(got - expected) < 1e-9 else 1)


if __name__ == "__main__":
    main()
