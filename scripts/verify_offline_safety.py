"""P1 acceptance check: drive the real evaluator over the full public set with
every reset()/respond() call instrumented, and report all five criteria.

Run it under scripts/no-network.sb (see scripts/verify_offline_safety.sh) so
criteria 4 and 5 are measured with the process's networking actually revoked.

The instrumentation only observes -- it re-raises anything it catches after
recording it, so a guard that is missing shows up as a failure rather than
being masked by this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
import starter.agent as agent_module  # noqa: E402
from starter.agent import Agent  # noqa: E402

# The organizer's published weak-BM25 starter baseline, from the competition kit
# (TechJam2026/techjam-conversational-search): HitRate@10 0.125, MRR 0.068034,
# MTTC 9.81. Restated here as a TechnicalScore so criterion 5 can compare like
# with like, using the evaluator's own formula (local_evaluator.py:279-280):
#
#   efficiency = (11.0 - 9.81) / 10.0                        = 0.119
#   score      = 0.50*0.125 + 0.30*0.068034 + 0.20*0.119     = 0.10671
#
# Derived, not measured locally — there is no committed baseline artifact in this
# repo, and Phase 0 (reproduce the baseline ourselves) is still open. Recompute
# from the three published numbers if the organizer ever revises them.
BASELINE_TECHNICAL_SCORE = 0.10671

# Read the contract rather than restate it: a transcription drift here would
# make criterion 3 self-confirming against the wrong schema.
_CONTRACT = json.loads(
    (Path(__file__).resolve().parent.parent / "docs" / "agent_api_contract.json").read_text(encoding="utf-8")
)
_TURN_RESPONSE = _CONTRACT["turn_response"]
ALLOWED_ATTRIBUTES = frozenset(
    value for value in _TURN_RESPONSE["properties"]["ask_attribute"]["enum"] if value is not None
)
MAX_RECOMMENDATIONS = _TURN_RESPONSE["properties"]["recommendations"]["maxItems"]
REQUIRED_FIELDS = tuple(_TURN_RESPONSE["required"])
REC_PROPERTIES = frozenset(_TURN_RESPONSE["properties"]["recommendations"]["items"]["properties"])


def schema_error(response: object) -> str | None:
    """Why `response` fails docs/agent_api_contract.json turn_response, or None."""
    if not isinstance(response, dict):
        return f"not a dict: {type(response).__name__}"
    missing = [field for field in REQUIRED_FIELDS if field not in response]
    if missing:
        return f"missing required field(s): {missing}"
    if not isinstance(response.get("message"), str):
        return f"message is {type(response.get('message')).__name__}, not string"
    attribute = response.get("ask_attribute")
    if attribute is not None and (not isinstance(attribute, str) or attribute not in ALLOWED_ATTRIBUTES):
        return f"ask_attribute {attribute!r} outside enum"
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        return f"recommendations is {type(recommendations).__name__}, not array"
    if len(recommendations) > MAX_RECOMMENDATIONS:
        return f"recommendations has {len(recommendations)} items, maxItems is {MAX_RECOMMENDATIONS}"
    for item in recommendations:
        if not isinstance(item, dict):
            return f"recommendation item is {type(item).__name__}, not object"
        if set(item) - REC_PROPERTIES:
            return f"recommendation has additional properties: {sorted(set(item) - REC_PROPERTIES)}"
        if not isinstance(item.get("parent_asin"), str) or not item["parent_asin"]:
            return f"parent_asin {item.get('parent_asin')!r} is not a non-empty string"
    usage = response.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            return f"usage is {type(usage).__name__}, not object"
        for field in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage.get(field), int) or usage[field] < 0:
                return f"usage.{field} = {usage.get(field)!r} is not a non-negative integer"
    return None


class InstrumentedAgent(Agent):
    """Counts calls and records any raise or schema violation, then re-raises."""

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.resets = 0
        self.responds = 0
        self.reset_failures: list[str] = []
        self.respond_failures: list[str] = []
        self.schema_failures: list[str] = []
        # Criterion 4 needs a signal that a *silently dead* agent fails. An empty
        # recommendation list is schema-valid and raises nothing, so neither of
        # the lists above catches it -- see the criterion 4 note in main().
        self.empty_recommendation_turns: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.resets += 1
        try:
            super().reset(session_id, user_profile)
        except Exception as error:
            self.reset_failures.append(f"reset #{self.resets}: {type(error).__name__}: {error}")
            raise

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.responds += 1
        try:
            response = super().respond(session_id, user_message, turn, top_k)
        except Exception as error:
            self.respond_failures.append(f"respond #{self.responds}: {type(error).__name__}: {error}")
            raise
        error_text = schema_error(response)
        if error_text:
            self.schema_failures.append(f"respond #{self.responds}: {error_text}")
        recommendations = response.get("recommendations") if isinstance(response, dict) else None
        if not isinstance(recommendations, list) or not recommendations:
            self.empty_recommendation_turns.append(
                f"respond #{self.responds}: session {session_id} turn {turn} returned 0 recommendations"
            )
        return response


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    # The guard in starter/agent.py must enforce the contract's own values.
    assert agent_module._ALLOWED_ATTRIBUTES == ALLOWED_ATTRIBUTES, (
        f"agent enum {sorted(agent_module._ALLOWED_ATTRIBUTES)} != contract enum {sorted(ALLOWED_ATTRIBUTES)}"
    )
    assert agent_module._MAX_RECOMMENDATIONS == MAX_RECOMMENDATIONS, (
        f"agent maxItems {agent_module._MAX_RECOMMENDATIONS} != contract maxItems {MAX_RECOMMENDATIONS}"
    )
    print(f"contract cross-check: enum ({len(ALLOWED_ATTRIBUTES)} values) and "
          f"maxItems ({MAX_RECOMMENDATIONS}) match starter/agent.py")

    catalog = str(root / "data" / "catalog.jsonl")
    samples = load_jsonl(root / "data" / "public_set.jsonl")
    catalog_ids, categories, products = catalog_index(catalog)

    agent = InstrumentedAgent(catalog)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    score = result["recommended_technical_score"]

    criteria = [
        ("1. reset() never raised, all profile shapes in the set",
         not agent.reset_failures, f"{agent.resets} calls, {len(agent.reset_failures)} raised"),
        ("2. respond() never raised, every turn of every session",
         not agent.respond_failures, f"{agent.responds} calls, {len(agent.respond_failures)} raised"),
        ("3. every respond() value validates against turn_response",
         not agent.schema_failures, f"{agent.responds} responses, {len(agent.schema_failures)} invalid"),
        # `sample_count == len(samples)` on its own is structurally always true:
        # evaluate() appends exactly one session row per sample unconditionally
        # (local_evaluator.py:269), so the only way it can differ is an exception
        # escaping -- which would have killed this script before the check ran. A
        # fully dead agent (null index, zero recommendations, score 0.0) passed
        # that check. The turn-level clause is what a dead agent actually fails:
        # the organizer's rule is that a session ends when the target appears in
        # the scored Top 10 or after turn 10, so an agent that never puts
        # anything in the Top 10 is not "completing" sessions in any useful
        # sense. Phase 5 depends on this criterion, so it has to bite.
        ("4. all sessions ran AND every turn returned a non-empty Top-10",
         result["sample_count"] == len(samples)
         and agent.responds > 0
         and not agent.empty_recommendation_turns,
         f"{result['sample_count']}/{len(samples)} sessions scored, "
         f"{agent.responds - len(agent.empty_recommendation_turns)}/{agent.responds} "
         f"turns returned recommendations"),
        (f"5. score >= {BASELINE_TECHNICAL_SCORE} baseline and not 0.00000",
         score >= BASELINE_TECHNICAL_SCORE and round(score, 5) != 0.0,
         f"recommended_technical_score = {score}"),
    ]

    print("\n=== P1 acceptance criteria ===")
    for label, passed, detail in criteria:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}\n         {detail}")
    for failures in (agent.reset_failures, agent.respond_failures, agent.schema_failures,
                     agent.empty_recommendation_turns):
        for failure in failures[:10]:
            print(f"    ! {failure}")

    print("\n=== aggregate ===")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))
    (root / "results_offline.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return 0 if all(passed for _, passed, _ in criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
