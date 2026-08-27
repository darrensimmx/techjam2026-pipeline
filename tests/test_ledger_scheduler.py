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

    def test_skips_boundary_decline(self) -> None:
        state = SessionState()
        state.record_message("I don't have a preference for color; please use your judgment.")
        self.assertEqual(state.disclosed_constraints, "")


class TestScheduler(unittest.TestCase):
    def test_cycles_through_all_six_without_repeats(self) -> None:
        asked: list[str] = []
        for _ in range(len(FIXED_SCHEDULE)):
            attribute = next_attribute(asked)
            self.assertIsNotNone(attribute)
            self.assertNotIn(attribute, asked)
            asked.append(attribute)  # type: ignore[arg-type]

    def test_returns_none_once_exhausted(self) -> None:
        self.assertIsNone(next_attribute(list(FIXED_SCHEDULE)))


if __name__ == "__main__":
    unittest.main()
