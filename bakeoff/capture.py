"""Capture the full 10-turn query trajectory for all 200 public sessions.

Why this is exact, and not an approximation of the evaluator
------------------------------------------------------------
In `evaluator/local_evaluator.py:238-268` the next customer message depends on
exactly three things: `response["ask_attribute"]`, the `disclosed` set, and
`boundary_used`. It does NOT depend on `response["recommendations"]`. The only
thing recommendations control is `if override_applied and target in ranked:
break` (:252-255) -- i.e. when the session *stops*.

So the query trajectory is independent of retrieval quality. Run the shipped
ask policy once with an agent that returns zero recommendations, and every
session runs the full 10 turns and emits the same queries any retrieval arm
would have seen. Every arm can then be scored offline by replaying those
queries. That turns "one full evaluator run per arm" into "one run, then pure
re-ranking", which is what makes a weight sweep affordable.

This is validated, not assumed: `simulate.py` replays R1 and must reproduce the
real `evaluate()` TechnicalScore to 6 decimals. If it does not, the assumption
above is wrong and every downstream number is void.

The vendored evaluator is imported, never modified (submission_rules.md:51).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.ledger import SessionState  # noqa: E402
from starter.retrieval import Bm25Index  # noqa: E402
from starter.scheduler import next_attribute  # noqa: E402

CANDIDATE_DEPTH = 100  # the contract's maxItems (agent_api_contract.json)

# The ledger as it stood at 94e8916 ("P1 offline safety") -- the commit that
# produced the 0.722818 the planning repo still quotes as our offline score.
# 3bc061f then widened the content-free filter, and HEAD scores 0.692586.
# The bake-off's fixed conditions pin "every disclosed string concatenated into
# the query, every turn", which is closer to the legacy behaviour, and the gap
# between the two ledgers is an order of magnitude larger than the effects
# under test -- so both are captured and every ceiling is reported against both.
_LEGACY_DECLINE_RE = re.compile(
    r"no preference|don'?t have a preference|do not have a preference|use your judgment",
    re.IGNORECASE,
)


class LegacySessionState(SessionState):
    def record_message(self, message: str) -> None:
        if not message or _LEGACY_DECLINE_RE.search(message):
            return
        cleaned = re.sub(r"\s+", " ", message).strip()
        if cleaned:
            self.disclosed_constraints = f"{self.disclosed_constraints} {cleaned}".strip()


LEDGERS = {"current": SessionState, "legacy": LegacySessionState}


class RecordingAgent:
    """The shipped ask policy and a chosen ledger, with retrieval removed.

    Returns no recommendations so no session can terminate early -- every
    session plays out all 10 turns and discloses everything the schedule can
    extract. Records the query the shipped agent *would* have issued.
    """

    def __init__(self, ledger_cls=SessionState) -> None:
        self._ledger_cls = ledger_cls
        self._sessions: dict[str, SessionState] = {}
        self.order: list[str] = []
        self.log: dict[str, list[str]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[str(session_id)] = self._ledger_cls()
        self.order.append(str(session_id))
        self.log[str(session_id)] = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.setdefault(str(session_id), self._ledger_cls())
        state.record_message(user_message if isinstance(user_message, str) else "")
        # starter/agent.py:128 -- identical query construction.
        query = state.disclosed_constraints or (user_message or "")
        self.log.setdefault(str(session_id), []).append(query)
        attribute = next_attribute(state)
        if attribute is not None:
            state.mark_asked(attribute)
        return {
            "message": "",
            "ask_attribute": attribute,
            "recommendations": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", choices=sorted(LEDGERS), default="current")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    catalog_path = ROOT / "data" / "catalog.jsonl"
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    catalog_ids, categories, products = catalog_index(catalog_path)

    agent = RecordingAgent(LEDGERS[args.ledger])
    t0 = time.time()
    evaluate(agent, samples, catalog_ids, categories, products)
    print(f"trajectory pass: {time.time() - t0:.1f}s", file=sys.stderr)

    assert len(agent.order) == len(samples), (len(agent.order), len(samples))

    t0 = time.time()
    index = Bm25Index(catalog_path)
    print(f"bm25 index build: {time.time() - t0:.1f}s", file=sys.stderr)

    records = []
    t0 = time.time()
    for position, (sample, session_id) in enumerate(zip(samples, agent.order)):
        queries = agent.log[session_id]
        assert len(queries) == 10, (sample["sample_id"], len(queries))
        _card, behavior = materialize_hidden_fields(sample, products)
        # local_evaluator.py:234 + :259 -- for intent_override the hit check is
        # suppressed until the override turn; every other scenario scores from 1.
        if sample["scenario_type"] == "intent_override":
            scoreable_from = int((behavior.get("override") or {}).get("turn", 3))
        else:
            scoreable_from = 1
        records.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "category_bucket": sample.get("category_bucket"),
            "target": str(sample["ground_truth"]["parent_asin"]),
            "scoreable_from": scoreable_from,
            "queries": queries,
            "bm25": [index.search(query, CANDIDATE_DEPTH) for query in queries],
        })
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(samples)} retrieved", file=sys.stderr)
    print(f"bm25 retrieval ({len(samples) * 10} queries): {time.time() - t0:.1f}s", file=sys.stderr)

    out = Path(args.out) if args.out else ROOT / "bakeoff" / f"trajectories-{args.ledger}.json"
    out.write_text(json.dumps(records), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
