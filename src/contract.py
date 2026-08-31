"""FROZEN. The wire boundary: coerce anything into a schema-valid turn_response.

Why coercion and not validation-then-raise: the evaluator zeroes a
schema-invalid dict exactly as silently as it zeroes an exception
(local_evaluator.py:243-244). There is no error path to take -- the only useful
behaviour is to hand back the schema-valid empty form of whatever field went
wrong and keep the rest of the turn.

Schema (docs/agent_api_contract.json, turn_response):
    {"message": str,
     "ask_attribute": str|null,          enum of 10 + null
     "recommendations": [{"parent_asin": str}],   maxItems 100
     "usage": {"prompt_tokens": int>=0, "completion_tokens": int>=0}}

`additionalProperties: false` applies to the response object AND to each
recommendation object, so unknown keys are stripped rather than passed through.
The schema permits an optional numeric `score` on a recommendation; we never
emit one -- the evaluator ignores it and it is one more thing to get wrong.
"""
from __future__ import annotations

from typing import Any

from src.types import ALLOWED_ATTRIBUTES, DEFAULT_TOP_K, MAX_RECOMMENDATIONS, TurnPlan


def empty_response() -> dict:
    """A fresh schema-valid empty response.

    Built per call rather than copied from a module constant: a shallow copy
    would leave the nested `usage` dict shared between every turn of every
    session, so one turn mutating it would corrupt all of them.
    """
    return {
        "message": "",
        "ask_attribute": None,
        "recommendations": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def clamp_top_k(top_k: object) -> int:
    """Clamp `top_k` into [0, MAX_RECOMMENDATIONS].

    A negative or non-integer LIMIT makes SQLite return the whole catalog or
    raise -- the first silently violates maxItems, the second silently zeroes a
    turn. `bool` is excluded explicitly because `True == 1` would otherwise pass
    the isinstance check and read as a legitimate top_k of 1.
    """
    if isinstance(top_k, bool):
        return DEFAULT_TOP_K
    if isinstance(top_k, int):
        return max(0, min(top_k, MAX_RECOMMENDATIONS))
    return DEFAULT_TOP_K


def validated(payload: object) -> dict:
    """Coerce `payload` into a value that validates against turn_response.

    Every field independently falls back to its schema-valid empty form. A bad
    `ask_attribute` does not cost you the recommendations, and vice versa.
    """
    response = empty_response()
    if not isinstance(payload, dict):
        return response

    message = payload.get("message")
    if isinstance(message, str):
        response["message"] = message

    attribute = payload.get("ask_attribute")
    if isinstance(attribute, str) and attribute in ALLOWED_ATTRIBUTES:
        response["ask_attribute"] = attribute

    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list):
        clean: list[dict] = []
        seen: set[str] = set()
        for item in recommendations:
            if isinstance(item, dict):
                parent_asin = item.get("parent_asin")
            elif isinstance(item, str):
                parent_asin = item
            else:
                continue
            # Strip everything but parent_asin: additionalProperties is false.
            if isinstance(parent_asin, str) and parent_asin and parent_asin not in seen:
                seen.add(parent_asin)
                clean.append({"parent_asin": parent_asin})
                if len(clean) >= MAX_RECOMMENDATIONS:
                    break
        response["recommendations"] = clean

    usage = payload.get("usage")
    if isinstance(usage, dict):
        counts: dict[str, int] = {}
        for name in ("prompt_tokens", "completion_tokens"):
            value = usage.get(name)
            counts[name] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
        response["usage"] = counts

    return response


def response_from_plan(plan: Any) -> dict:
    """Turn a TurnPlan into a validated wire response. Never raises."""
    if not isinstance(plan, TurnPlan):
        return empty_response()
    return validated({
        "message": plan.message,
        "ask_attribute": plan.ask_attribute,
        "recommendations": [{"parent_asin": pa} for pa in plan.parent_asins],
        "usage": {
            "prompt_tokens": plan.prompt_tokens,
            "completion_tokens": plan.completion_tokens,
        },
    })
