"""Proves respond() matches docs/agent_api_contract.json and never raises.

Checked against the vendored contract file directly (not hand-duplicated
magic strings) so this test can't silently drift from the real schema.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from starter.agent import Agent

FIXTURES = Path(__file__).parent / "fixtures"
CONTRACT = json.loads((Path(__file__).parent.parent / "docs" / "agent_api_contract.json").read_text())
ASK_ATTRIBUTE_ENUM = set(CONTRACT["turn_response"]["properties"]["ask_attribute"]["enum"])


def assert_valid_turn_response(test: unittest.TestCase, response: dict) -> None:
    for key in CONTRACT["turn_response"]["required"]:
        test.assertIn(key, response)
    test.assertIsInstance(response["message"], str)
    test.assertIn(response["ask_attribute"], ASK_ATTRIBUTE_ENUM)
    test.assertIsInstance(response["recommendations"], list)
    test.assertLessEqual(len(response["recommendations"]), 100)
    for item in response["recommendations"]:
        test.assertIn("parent_asin", item)
        test.assertIsInstance(item["parent_asin"], str)
        test.assertTrue(item["parent_asin"])


class TestAgentContract(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent(FIXTURES / "catalog.jsonl")
        self.agent.reset("session-1", {"summary": "test"})

    def test_first_turn_response_matches_contract(self) -> None:
        response = self.agent.respond("session-1", "Looking for waterproof leather boots.", 1, 10)
        assert_valid_turn_response(self, response)
        self.assertTrue(any(rec["parent_asin"] == "T0001" for rec in response["recommendations"]))

    def test_asks_and_returns_recommendations_in_the_same_turn(self) -> None:
        """Standing finding: never withhold results to make room for a question."""
        response = self.agent.respond("session-1", "Looking for boots.", 1, 10)
        self.assertIsNotNone(response["ask_attribute"])
        self.assertGreater(len(response["recommendations"]), 0)

    def test_never_raises_on_missing_reset(self) -> None:
        response = self.agent.respond("never-reset-session", "hello", 1, 10)
        assert_valid_turn_response(self, response)

    def test_never_raises_on_malformed_input(self) -> None:
        response = self.agent.respond("session-1", None, 1, 10)  # type: ignore[arg-type]
        assert_valid_turn_response(self, response)

    def test_does_not_reask_an_already_asked_attribute(self) -> None:
        seen_attributes = []
        for turn in range(1, 8):
            response = self.agent.respond("session-1", f"turn {turn} message", turn, 10)
            assert_valid_turn_response(self, response)
            if response["ask_attribute"] is not None:
                seen_attributes.append(response["ask_attribute"])
        self.assertEqual(len(seen_attributes), len(set(seen_attributes)))


if __name__ == "__main__":
    unittest.main()
