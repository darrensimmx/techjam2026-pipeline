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

BASELINE_TECHNICAL_SCORE = 0.10671  # docs/baseline_results.json, weak_bm25

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
        ("4. full 200-session run completed without crashing",
         result["sample_count"] == len(samples), f"{result['sample_count']}/{len(samples)} sessions scored"),
        (f"5. score >= {BASELINE_TECHNICAL_SCORE} baseline and not 0.00000",
         score >= BASELINE_TECHNICAL_SCORE and round(score, 5) != 0.0,
         f"recommended_technical_score = {score}"),
    ]

    print("\n=== P1 acceptance criteria ===")
    for label, passed, detail in criteria:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}\n         {detail}")
    for failures in (agent.reset_failures, agent.respond_failures, agent.schema_failures):
        for failure in failures[:10]:
            print(f"    ! {failure}")

    print("\n=== aggregate ===")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))
    (root / "results_offline.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return 0 if all(passed for _, passed, _ in criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
