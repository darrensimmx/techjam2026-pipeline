"""Unit tests for the two pieces of state the agent depends on."""
from __future__ import annotations

import unittest

from starter.ledger import SessionState
from starter.scheduler import FIXED_SCHEDULE, next_attribute


class TestLedger(unittest.TestCase):
    def test_accumulates_across_turns(self) -> None:
        state = SessionState()
        state.record_message("I want waterproof boots.")
        state.record_message("Prefer leather.")
        self.assertIn("waterproof boots", state.disclosed_constraints)
        self.assertIn("leather", state.disclosed_constraints)

    def test_skips_every_content_free_reply(self) -> None:
        """All three simulator templates that disclose nothing. Missing any of
        them injects noise tokens into the query for the rest of the session —
        the third fires on every turn 7-10, once the schedule is exhausted."""
        for message in (
            "I don't have a preference for color; please use your judgment.",
            "I don't have an additional preference for material.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
        ):
            with self.subTest(message=message):
                state = SessionState()
                state.record_message(message)
                self.assertEqual(state.disclosed_constraints, "")

    def test_appends_replies_that_disclose_something(self) -> None:
        """Including the override sentence — it carries the new value."""
        for message in (
            "For that, what matters is: waterproof leather.",
            "Actually, ignore my earlier preference. What I need is: black leather.",
        ):
            with self.subTest(message=message):
                state = SessionState()
                state.record_message(message)
                self.assertEqual(state.disclosed_constraints, message)


class TestScheduler(unittest.TestCase):
    def test_cycles_through_all_six_without_repeats(self) -> None:
        state = SessionState()
        for _ in range(len(FIXED_SCHEDULE)):
            attribute = next_attribute(state)
            self.assertIsNotNone(attribute)
            self.assertNotIn(attribute, state.asked_attributes)
            state.mark_asked(attribute)  # type: ignore[arg-type]

    def test_returns_none_once_exhausted(self) -> None:
        state = SessionState()
        for attribute in FIXED_SCHEDULE:
            state.mark_asked(attribute)
        self.assertIsNone(next_attribute(state))


if __name__ == "__main__":
    unittest.main()
