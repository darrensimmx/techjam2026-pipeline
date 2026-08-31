"""The schema guard for src/contract.py.  [WS-A OWNS]

docs/agent_api_contract.json is read PROGRAMMATICALLY here, never transcribed. A
hand-copied enum or maxItems would make this file self-confirming: it would go on
passing against a contract that had changed underneath it, which is the one thing
a drift guard exists to prevent.

The validator below is a deliberately small JSON-Schema subset -- only the
keywords `turn_response` actually uses. `test_validator_understands_every_keyword`
walks the contract and fails if it grows a keyword this file would silently
ignore, so the validator cannot quietly become vacuous.

`assert_valid_turn_response` is the reusable helper; tests/test_src_agent.py
imports it. It rejects unknown keys at BOTH levels -- the response object and
each recommendation object -- because `additionalProperties: false` is set on
both.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.contract import clamp_top_k, empty_response, response_from_plan, validated
from src.types import (
    ALLOWED_ATTRIBUTES,
    DEFAULT_TOP_K,
    FORBIDDEN_ASK,
    MAX_RECOMMENDATIONS,
    MAX_TURNS,
    TurnPlan,
)

CONTRACT_PATH = Path(__file__).resolve().parent.parent / "docs" / "agent_api_contract.json"
CONTRACT: dict = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
TURN_RESPONSE: dict = CONTRACT["turn_response"]
TURN_REQUEST: dict = CONTRACT["turn_request"]

# Every keyword the validator below implements. A schema keyword outside this set
# is a constraint we would not be checking, so the walk test treats it as failure.
SUPPORTED_KEYWORDS = frozenset({
    "type", "enum", "const", "required", "properties", "additionalProperties",
    "items", "maxItems", "minItems", "minLength", "maxLength", "minimum", "maximum",
})


# ---------------------------------------------------------------------------
# A minimal JSON Schema validator. stdlib only -- `jsonschema` is third party
# and nothing in this repo may depend on one.
# ---------------------------------------------------------------------------

def _matches_type(value: object, name: str) -> bool:
    if name == "integer":
        # JSON Schema: booleans are NOT integers, even though isinstance says so.
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise ValueError(f"unsupported schema type {name!r}")


def _same_scalar(value: object, member: object) -> bool:
    """Strict equality for enum/const. `True == 1` in Python but not in JSON."""
    return type(value) is type(member) and value == member


def schema_errors(value: object, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable violations. Empty list means valid."""
    errors: list[str] = []

    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_matches_type(value, name) for name in names):
            errors.append(
                f"{path}: expected type {names}, got {type(value).__name__} ({value!r})"
            )
            return errors  # every further keyword is meaningless on a wrong type

    if "const" in schema and not _same_scalar(value, schema["const"]):
        errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")

    if "enum" in schema and not any(_same_scalar(value, m) for m in schema["enum"]):
        errors.append(f"{path}: {value!r} is not in enum {schema['enum']!r}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required key {name!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(
                        f"{path}: unknown key {key!r} (additionalProperties is false)"
                    )
        for name, subschema in properties.items():
            if name in value:
                errors.extend(schema_errors(value[name], subschema, f"{path}.{name}"))

    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items exceeds maxItems={schema['maxItems']}")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items below minItems={schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for position, item in enumerate(value):
                errors.extend(schema_errors(item, item_schema, f"{path}[{position}]"))

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than minLength={schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength={schema['maxLength']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value!r} below minimum={schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value!r} above maximum={schema['maximum']}")

    return errors


def assert_valid_turn_response(testcase: unittest.TestCase, response: object) -> None:
    """Assert `response` validates against docs/agent_api_contract.json.

    THE reusable assertion for the whole src/ suite. Unknown keys are rejected at
    both the response level and the per-recommendation level.
    """
    errors = schema_errors(response, TURN_RESPONSE)
    if errors:
        testcase.fail(
            "turn_response schema violations for {!r}:\n  {}".format(
                response, "\n  ".join(errors)
            )
        )


def _walk_keywords(schema: object) -> set[str]:
    """Every schema keyword used anywhere under `schema`."""
    found: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                for subschema in value.values():
                    found |= _walk_keywords(subschema)
                found.add(key)
            elif key == "items":
                found.add(key)
                found |= _walk_keywords(value)
            else:
                found.add(key)
    return found


# ---------------------------------------------------------------------------


class TestValidatorItself(unittest.TestCase):
    """A vacuous validator would make every other test in the suite vacuous."""

    def test_validator_understands_every_keyword(self) -> None:
        used = _walk_keywords(TURN_RESPONSE)
        unsupported = used - SUPPORTED_KEYWORDS
        self.assertEqual(
            unsupported, set(),
            "docs/agent_api_contract.json grew schema keyword(s) this validator "
            "ignores; implement them in schema_errors() before trusting it",
        )

    def test_helper_accepts_the_empty_form(self) -> None:
        assert_valid_turn_response(self, empty_response())

    def test_helper_rejects_invalid_payloads(self) -> None:
        cases = {
            "not a dict": [],
            "message missing": {"ask_attribute": None, "recommendations": []},
            "message not a string": {"message": 7, "ask_attribute": None, "recommendations": []},
            "ask_attribute off-enum": {"message": "", "ask_attribute": "vibe", "recommendations": []},
            "recommendations not a list": {"message": "", "ask_attribute": None, "recommendations": {}},
            "unknown top-level key": {
                "message": "", "ask_attribute": None, "recommendations": [], "debug": 1,
            },
            "unknown recommendation key": {
                "message": "", "ask_attribute": None,
                "recommendations": [{"parent_asin": "A1", "title": "x"}],
            },
            "recommendation missing parent_asin": {
                "message": "", "ask_attribute": None, "recommendations": [{}],
            },
            "empty parent_asin violates minLength": {
                "message": "", "ask_attribute": None, "recommendations": [{"parent_asin": ""}],
            },
            "too many recommendations": {
                "message": "", "ask_attribute": None,
                "recommendations": [{"parent_asin": f"A{i}"} for i in range(MAX_RECOMMENDATIONS + 1)],
            },
            "negative prompt_tokens": {
                "message": "", "ask_attribute": None, "recommendations": [],
                "usage": {"prompt_tokens": -1, "completion_tokens": 0},
            },
            "bool is not an integer token count": {
                "message": "", "ask_attribute": None, "recommendations": [],
                "usage": {"prompt_tokens": True, "completion_tokens": 0},
            },
            "unknown usage key": {
                "message": "", "ask_attribute": None, "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total": 0},
            },
        }
        for label, payload in cases.items():
            with self.subTest(label):
                self.assertNotEqual(
                    schema_errors(payload, TURN_RESPONSE), [],
                    f"validator accepted an invalid payload: {label}",
                )

    def test_helper_rejects_unknown_keys_at_both_levels(self) -> None:
        # Named separately from the sweep above because "both levels" is the
        # specific property the rest of the suite leans on.
        top = {"message": "", "ask_attribute": None, "recommendations": [], "extra": 1}
        nested = {
            "message": "", "ask_attribute": None,
            "recommendations": [{"parent_asin": "A1", "extra": 1}],
        }
        self.assertTrue(any("unknown key 'extra'" in e for e in schema_errors(top, TURN_RESPONSE)))
        self.assertTrue(any("unknown key 'extra'" in e for e in schema_errors(nested, TURN_RESPONSE)))


class TestContractDrift(unittest.TestCase):
    """src/types.py constants vs the contract file. Read, never transcribed."""

    def test_allowed_attributes_matches_contract_enum(self) -> None:
        enum = TURN_RESPONSE["properties"]["ask_attribute"]["enum"]
        from_contract = frozenset(value for value in enum if isinstance(value, str))
        self.assertEqual(ALLOWED_ATTRIBUTES, from_contract)
        self.assertIn(None, enum, "the contract enum must still admit a null ask")

    def test_max_recommendations_matches_contract_max_items(self) -> None:
        self.assertEqual(
            MAX_RECOMMENDATIONS,
            TURN_RESPONSE["properties"]["recommendations"]["maxItems"],
        )

    def test_default_top_k_matches_contract_const(self) -> None:
        self.assertEqual(DEFAULT_TOP_K, TURN_REQUEST["properties"]["top_k"]["const"])

    def test_max_turns_matches_contract_turn_maximum(self) -> None:
        self.assertEqual(MAX_TURNS, TURN_REQUEST["properties"]["turn"]["maximum"])

    def test_forbidden_ask_is_a_subset_of_the_contract_enum(self) -> None:
        # `other` is schema-VALID and policy-FORBIDDEN. The two must not be
        # conflated: contract.validated() lets it through by design, and
        # src/types.py is where the refusal lives.
        self.assertTrue(FORBIDDEN_ASK <= ALLOWED_ATTRIBUTES)


class TestValidated(unittest.TestCase):
    def test_strips_unknown_key_from_a_recommendation(self) -> None:
        response = validated({
            "message": "ok",
            "ask_attribute": "color",
            "recommendations": [{"parent_asin": "A1", "title": "Red Boot", "rank": 3}],
        })
        self.assertEqual(response["recommendations"], [{"parent_asin": "A1"}])
        assert_valid_turn_response(self, response)

    def test_strips_the_schema_legal_score_key_too(self) -> None:
        # `score` is permitted by the schema but contract.py deliberately never
        # emits one. Asserting it is stripped pins that decision.
        response = validated({"recommendations": [{"parent_asin": "A1", "score": 0.9}]})
        self.assertEqual(response["recommendations"], [{"parent_asin": "A1"}])
        assert_valid_turn_response(self, response)

    def test_drops_duplicates_non_dicts_and_non_strings(self) -> None:
        response = validated({"recommendations": [
            {"parent_asin": "A1"},
            "A2",                       # bare string is accepted and wrapped
            {"parent_asin": "A1"},      # duplicate dict
            "A2",                       # duplicate string
            {"parent_asin": ""},        # empty violates minLength
            {"parent_asin": None},      # wrong type
            {"parent_asin": 7},         # wrong type
            {"title": "no asin"},       # missing the required key
            None, 7, 3.5, True, ["A3"], ("A4",), {"A5"},
            {"parent_asin": "A6"},
        ]})
        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "A1"}, {"parent_asin": "A2"}, {"parent_asin": "A6"}],
        )
        assert_valid_turn_response(self, response)

    def test_truncates_at_max_recommendations(self) -> None:
        response = validated({
            "recommendations": [{"parent_asin": f"A{i}"} for i in range(MAX_RECOMMENDATIONS + 50)]
        })
        self.assertEqual(len(response["recommendations"]), MAX_RECOMMENDATIONS)
        assert_valid_turn_response(self, response)

    def test_bad_ask_attribute_does_not_cost_the_recommendations(self) -> None:
        response = validated({
            "message": "here you go",
            "ask_attribute": "vibe",              # off-enum
            "recommendations": [{"parent_asin": "A1"}],
        })
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["message"], "here you go")
        self.assertEqual(response["recommendations"], [{"parent_asin": "A1"}])
        assert_valid_turn_response(self, response)

    def test_bad_recommendations_do_not_cost_the_ask_attribute(self) -> None:
        response = validated({
            "message": "here you go",
            "ask_attribute": "material",
            "recommendations": "not a list",
        })
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["message"], "here you go")
        self.assertEqual(response["recommendations"], [])
        assert_valid_turn_response(self, response)

    def test_bad_message_does_not_cost_anything_else(self) -> None:
        response = validated({
            "message": 7,
            "ask_attribute": "budget",
            "recommendations": [{"parent_asin": "A1"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        })
        self.assertEqual(response["message"], "")
        self.assertEqual(response["ask_attribute"], "budget")
        self.assertEqual(response["recommendations"], [{"parent_asin": "A1"}])
        self.assertEqual(response["usage"], {"prompt_tokens": 5, "completion_tokens": 6})
        assert_valid_turn_response(self, response)

    def test_every_allowed_attribute_survives(self) -> None:
        for attribute in sorted(ALLOWED_ATTRIBUTES):
            with self.subTest(attribute):
                response = validated({"ask_attribute": attribute})
                self.assertEqual(response["ask_attribute"], attribute)
                assert_valid_turn_response(self, response)

    def test_usage_counts_are_coerced_independently(self) -> None:
        cases = {
            "negative": ({"prompt_tokens": -3, "completion_tokens": 4}, {"prompt_tokens": 0, "completion_tokens": 4}),
            "bool": ({"prompt_tokens": True, "completion_tokens": 4}, {"prompt_tokens": 0, "completion_tokens": 4}),
            "float": ({"prompt_tokens": 1.5, "completion_tokens": 4}, {"prompt_tokens": 0, "completion_tokens": 4}),
            "string": ({"prompt_tokens": "5", "completion_tokens": 4}, {"prompt_tokens": 0, "completion_tokens": 4}),
            "missing": ({"completion_tokens": 4}, {"prompt_tokens": 0, "completion_tokens": 4}),
            "unknown key": ({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                            {"prompt_tokens": 1, "completion_tokens": 2}),
        }
        for label, (usage, expected) in cases.items():
            with self.subTest(label):
                response = validated({"usage": usage})
                self.assertEqual(response["usage"], expected)
                assert_valid_turn_response(self, response)

    def test_non_dict_and_broken_payloads_return_the_empty_form(self) -> None:
        for payload in (None, [], "", 0, 7, 3.5, True, set(), ("message", "hi"),
                        {"message": 7}, {}, {"unknown": "key"}):
            with self.subTest(repr(payload)):
                response = validated(payload)
                self.assertEqual(response, empty_response())
                assert_valid_turn_response(self, response)

    def test_never_emits_a_key_outside_the_schema(self) -> None:
        response = validated({"message": "x", "sneaky": 1, "recommendations": []})
        self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
        assert_valid_turn_response(self, response)


class TestClampTopK(unittest.TestCase):
    def test_the_documented_cases(self) -> None:
        cases = {
            -1: 0,
            True: DEFAULT_TOP_K,      # bool is not an int here
            "10": DEFAULT_TOP_K,      # a string is not a top_k
            10 ** 12: MAX_RECOMMENDATIONS,
            3.7: DEFAULT_TOP_K,       # a float is not an int here
        }
        for value, expected in cases.items():
            with self.subTest(repr(value)):
                self.assertEqual(clamp_top_k(value), expected)

    def test_the_ordinary_cases(self) -> None:
        for value, expected in ((0, 0), (1, 1), (10, 10), (100, 100), (101, 100), (-10 ** 12, 0)):
            with self.subTest(repr(value)):
                self.assertEqual(clamp_top_k(value), expected)

    def test_false_is_not_zero(self) -> None:
        # The trap the bool branch exists for: False == 0 would silently make
        # every recommendation list empty.
        self.assertEqual(clamp_top_k(False), DEFAULT_TOP_K)

    def test_junk_falls_back_to_the_default(self) -> None:
        for value in (None, "", [], {}, set(), object(), float("nan"), float("inf")):
            with self.subTest(repr(value)):
                self.assertEqual(clamp_top_k(value), DEFAULT_TOP_K)

    def test_always_returns_a_real_int_in_range(self) -> None:
        for value in (-1, 0, 10, 10 ** 12, True, "10", 3.7, None, [], object()):
            with self.subTest(repr(value)):
                result = clamp_top_k(value)
                self.assertIsInstance(result, int)
                self.assertNotIsInstance(result, bool)
                self.assertGreaterEqual(result, 0)
                self.assertLessEqual(result, MAX_RECOMMENDATIONS)


class TestNoAliasing(unittest.TestCase):
    """A module-level constant plus a shallow copy is a real bug class: every
    response would share one nested `usage` dict, so one turn mutating it would
    corrupt all 200 sessions' reported token counts."""

    def _assert_independent(self, first: dict, second: dict) -> None:
        self.assertIsNot(first, second)
        self.assertIsNot(first["usage"], second["usage"])
        self.assertIsNot(first["recommendations"], second["recommendations"])
        first["usage"]["prompt_tokens"] = 999_999
        first["recommendations"].append({"parent_asin": "MUTATED"})
        first["message"] = "mutated"
        self.assertEqual(second["usage"]["prompt_tokens"], 0)
        self.assertEqual(second["recommendations"], [])
        assert_valid_turn_response(self, second)

    def test_empty_response_is_not_aliased(self) -> None:
        self._assert_independent(empty_response(), empty_response())

    def test_validated_is_not_aliased(self) -> None:
        self._assert_independent(validated(None), validated(None))

    def test_response_from_plan_is_not_aliased(self) -> None:
        plan = TurnPlan(message="hi", ask_attribute="color")
        self._assert_independent(response_from_plan(plan), response_from_plan(plan))

    def test_mutating_a_response_does_not_reach_a_later_empty_form(self) -> None:
        first = validated({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
        first["usage"]["prompt_tokens"] = 12345
        self.assertEqual(empty_response()["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_the_input_payload_is_not_reused_as_the_output(self) -> None:
        payload = {"message": "x", "ask_attribute": None,
                   "recommendations": [{"parent_asin": "A1"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        response = validated(payload)
        self.assertIsNot(response, payload)
        self.assertIsNot(response["usage"], payload["usage"])
        self.assertIsNot(response["recommendations"], payload["recommendations"])
        self.assertIsNot(response["recommendations"][0], payload["recommendations"][0])


class TestResponseFromPlan(unittest.TestCase):
    def test_a_well_formed_plan(self) -> None:
        plan = TurnPlan(
            message="Here are the closest matches.",
            ask_attribute="material",
            parent_asins=("A1", "A2"),
            prompt_tokens=11,
            completion_tokens=22,
        )
        response = response_from_plan(plan)
        assert_valid_turn_response(self, response)
        self.assertEqual(response["message"], "Here are the closest matches.")
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"],
                         [{"parent_asin": "A1"}, {"parent_asin": "A2"}])
        self.assertEqual(response["usage"], {"prompt_tokens": 11, "completion_tokens": 22})

    def test_a_non_plan_returns_the_empty_form(self) -> None:
        for payload in (None, {}, [], "TurnPlan", 7, object()):
            with self.subTest(repr(payload)):
                response = response_from_plan(payload)
                self.assertEqual(response, empty_response())
                assert_valid_turn_response(self, response)

    def test_a_plan_carrying_junk_is_still_coerced(self) -> None:
        # TurnPlan is frozen but not type-checked at runtime; the wire boundary
        # is the only thing standing between a bad field and a silent zero.
        plan = TurnPlan(
            message=None,                       # type: ignore[arg-type]
            ask_attribute="vibe",
            parent_asins=("A1", "A1", "", None, 7, "A2"),   # type: ignore[arg-type]
            prompt_tokens=-5,
            completion_tokens=None,             # type: ignore[arg-type]
        )
        response = response_from_plan(plan)
        assert_valid_turn_response(self, response)
        self.assertEqual(response["message"], "")
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["recommendations"],
                         [{"parent_asin": "A1"}, {"parent_asin": "A2"}])
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_a_plan_with_an_unhashable_parent_asin_does_not_raise(self) -> None:
        plan = TurnPlan(message="", ask_attribute=None,
                        parent_asins=(["A1"], {"A2": 1}, "A3"))  # type: ignore[arg-type]
        response = response_from_plan(plan)
        assert_valid_turn_response(self, response)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A3"}])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
