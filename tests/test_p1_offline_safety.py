"""P1 offline safety: the agent never crashes the evaluator and never returns a
value the evaluator will silently zero.

Two failure classes, one consequence. An exception out of reset() (or the
constructor) crashes the whole run -- the evaluator wraps respond() at
local_evaluator.py:239-244 but NOT reset() at :228 nor Agent() at :306. A
schema-invalid response is quietly replaced with an empty one at :243-244, so
it scores zero with no traceback anywhere. Both are guarded in starter.agent.

Criteria 4 and 5 (full 200-session run with networking actually revoked) need
the real catalog and a sandbox, so they live in scripts/verify_offline_safety.sh
rather than here.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from starter import agent as agent_module
from starter.agent import Agent, _empty_response, _limit, _validated
from tests.test_agent_contract import ASK_ATTRIBUTE_ENUM, CONTRACT, assert_valid_turn_response

FIXTURES = Path(__file__).parent / "fixtures"

UNUSABLE_CATALOGS = ("/nonexistent/catalog.jsonl", "/dev/null", str(FIXTURES), None, 12345)
SESSION_IDS = ("session-1", "", ["unhashable"], {"also": "unhashable"}, None, 42)
PROFILES = ({}, None, [], "a string", 42, {"preference_tags": None}, {"nested": {"a": [1, {"b": None}]}})
MESSAGES = ("cotton boots", "", None, 42, [], {"a": 1}, "NEAR NOT AND OR", 'a" OR "b', "\x00\x01", "x" * 20000)
TOP_KS = (10, 0, -1, 200, 5000, None, 3.7, "10", True, [], 10 ** 12)


class TestConstructorSafety(unittest.TestCase):
    """Criterion 4: Agent() is not wrapped, so a raise here ends the run."""

    def test_constructor_never_raises_on_an_unusable_catalog(self) -> None:
        for catalog in UNUSABLE_CATALOGS:
            with self.subTest(catalog=catalog):
                agent = Agent(catalog)  # type: ignore[arg-type]
                agent.reset("s", {})
                assert_valid_turn_response(self, agent.respond("s", "boots", 1, 10))


class TestResetSafety(unittest.TestCase):
    """Criterion 1: reset() is not wrapped, so a raise here ends the run."""

    def test_reset_never_raises_for_any_session_id_or_profile(self) -> None:
        agent = Agent(FIXTURES / "catalog.jsonl")
        for session_id in SESSION_IDS:
            for profile in PROFILES:
                with self.subTest(session_id=session_id, profile=profile):
                    agent.reset(session_id, profile)  # type: ignore[arg-type]


class TestRespondSafety(unittest.TestCase):
    """Criteria 2 and 3: never raises, and never returns a zeroable value."""

    def setUp(self) -> None:
        self.agent = Agent(FIXTURES / "catalog.jsonl")
        self.agent.reset("session-1", {"summary": "test"})

    def test_respond_stays_valid_across_hostile_messages_and_top_k(self) -> None:
        for message in MESSAGES:
            for top_k in TOP_KS:
                with self.subTest(message=message, top_k=top_k):
                    response = self.agent.respond("session-1", message, 1, top_k)  # type: ignore[arg-type]
                    assert_valid_turn_response(self, response)

    def test_respond_stays_valid_for_out_of_range_turns(self) -> None:
        for turn in (0, 1, 10, 11, -5, None, "3"):
            with self.subTest(turn=turn):
                assert_valid_turn_response(self, self.agent.respond("session-1", "boots", turn, 10))  # type: ignore[arg-type]

    def test_usage_is_not_shared_between_responses(self) -> None:
        """A shallow copy of a module-level constant aliases its nested usage
        dict, so one mutating caller would corrupt every later turn."""
        first = self.agent.respond("session-1", "boots", 1, 10)
        first["usage"]["prompt_tokens"] = 999
        second = self.agent.respond("session-1", "boots", 2, 10)
        self.assertEqual(second["usage"]["prompt_tokens"], 0)
        self.assertEqual(_empty_response()["usage"]["prompt_tokens"], 0)


class TestContractEnforcement(unittest.TestCase):
    """Criterion 3, at the unit level: the fixture catalog is far too small to
    produce an over-long result, but a real catalog with top_k=-1 returned
    47,602 rows -- an unbounded SQLite LIMIT, which raises nothing."""

    def test_limit_clamps_into_contract_bounds(self) -> None:
        for top_k in TOP_KS:
            with self.subTest(top_k=top_k):
                limit = _limit(top_k)
                self.assertIsInstance(limit, int)
                self.assertGreaterEqual(limit, 0)
                self.assertLessEqual(limit, agent_module._MAX_RECOMMENDATIONS)

    def test_validated_truncates_an_over_long_recommendation_list(self) -> None:
        oversized = {"message": "m", "ask_attribute": None,
                     "recommendations": [{"parent_asin": f"B{index}"} for index in range(5000)]}
        response = _validated(oversized)
        self.assertEqual(len(response["recommendations"]), agent_module._MAX_RECOMMENDATIONS)
        assert_valid_turn_response(self, response)

    def test_validated_coerces_hostile_payloads(self) -> None:
        payloads = (
            None, 42, [], "a string",
            {"ask_attribute": ["unhashable"]},
            {"ask_attribute": "NOT_IN_THE_ENUM"},
            {"message": None},
            {"recommendations": "not a list"},
            {"recommendations": [{"parent_asin": ""}, {"wrong_key": 1}, None, {"parent_asin": "OK"}]},
            {"usage": {"prompt_tokens": -1, "completion_tokens": 0}},
            {"usage": "not a dict"},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                assert_valid_turn_response(self, _validated(payload))

    def test_guard_constants_match_the_vendored_contract(self) -> None:
        """The guard must enforce the contract's values, not a stale copy."""
        enum_without_null = {value for value in ASK_ATTRIBUTE_ENUM if value is not None}
        self.assertEqual(set(agent_module._ALLOWED_ATTRIBUTES), enum_without_null)
        self.assertEqual(
            agent_module._MAX_RECOMMENDATIONS,
            CONTRACT["turn_response"]["properties"]["recommendations"]["maxItems"],
        )


if __name__ == "__main__":
    unittest.main()
