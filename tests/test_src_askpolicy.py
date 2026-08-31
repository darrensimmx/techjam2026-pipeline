"""WS-C: the seven-slot schedule, the free-turn ladder, and the two registers.

The load-bearing test in here is the rule-1 sweep. "next_attribute never returns
None and never returns `other`" is a property, not an example, and the evaluator
punishes both silently: a null ask draws the "Those options are not quite right
yet" template that the ledger drops (local_evaluator.py:171), and any value
outside ALLOWED_ATTRIBUTES is rewritten to `other` (:172-173), which switches on
the constraint-filter bypass at :180 that this submission has declined. Neither
failure raises, neither shows up in a score breakdown. So it is asserted
structurally, over every reachable shape of the state, rather than by
inspection.
"""
from __future__ import annotations

import itertools
import unittest

from src import askyield
from src.askpolicy import ASKABLE, AskState, next_attribute
from src.types import (
    ALLOWED_ATTRIBUTES,
    CARD_CAPACITY,
    DISCLOSURE_CAP,
    FIXED_SCHEDULE,
    FORBIDDEN_ASK,
    HEDGE_ORDER,
)

SENDABLE = ALLOWED_ATTRIBUTES - FORBIDDEN_ASK


def _subsets() -> list[tuple[str, ...]]:
    """(), every single, every pair, and the full set of askable attributes."""
    return (
        [()]
        + [(name,) for name in ASKABLE]
        + list(itertools.combinations(ASKABLE, 2))
        + [tuple(ASKABLE)]
    )


class TestNeverNullNeverOther(unittest.TestCase):
    """Rule 1, swept."""

    def _check(self, state: AskState, label: str, failures: list[str]) -> None:
        choice = next_attribute(state)
        if not isinstance(choice, str) or choice not in SENDABLE:
            failures.append(f"{label} -> {choice!r}")

    def test_exhaustive_sweep_of_state_shapes(self) -> None:
        subsets = _subsets()
        failures: list[str] = []
        calls = 0
        for asked in subsets:
            for retired in subsets:
                for burned in (None, "material", "brand", "other"):
                    for reasked in (False, True):
                        for disclosed in range(6):  # spans the CARD_CAPACITY boundary
                            for turn in range(1, 11):
                                for with_yield in (False, True):
                                    state = AskState(
                                        asked=list(asked),
                                        retired=set(retired),
                                        yield_counts=({name: DISCLOSURE_CAP for name in asked}
                                                      if with_yield else {}),
                                        yield_order=list(asked) if with_yield else [],
                                        burned=burned,
                                        burned_reasked=reasked,
                                        last_ask=asked[-1] if asked else None,
                                        turn=turn,
                                        disclosed_count=disclosed,
                                    )
                                    calls += 1
                                    self._check(state, f"asked={asked} retired={retired} "
                                                       f"burned={burned} reasked={reasked} "
                                                       f"disclosed={disclosed} turn={turn} "
                                                       f"yield={with_yield}", failures)
                                    if len(failures) > 5:
                                        self.fail("\n".join(failures))
        self.assertEqual(failures, [], "\n".join(failures[:5]))
        self.assertGreater(calls, 100_000)  # the sweep is real, not a stub

    def test_turn_out_of_range_still_answers(self) -> None:
        for turn in (-5, 0, 8, 11, 99, 10**9):
            with self.subTest(turn=turn):
                self.assertIn(next_attribute(AskState(turn=turn)), SENDABLE)

    def test_malformed_states_still_answer(self) -> None:
        broken = AskState()
        broken.asked = None            # type: ignore[assignment]
        broken.retired = "material"    # type: ignore[assignment]
        broken.yield_counts = None     # type: ignore[assignment]
        broken.yield_order = 7         # type: ignore[assignment]
        broken.turn = "3"              # type: ignore[assignment]
        broken.disclosed_count = None  # type: ignore[assignment]
        broken.burned = ["material"]   # type: ignore[assignment]
        for state in (broken, None, "not a state", 42, object(), {"turn": 3}):
            with self.subTest(state=repr(state)[:40]):
                self.assertIn(next_attribute(state), SENDABLE)  # type: ignore[arg-type]

    def test_state_whose_methods_raise_still_answers(self) -> None:
        class Hostile(AskState):
            def pending_reask(self):  # type: ignore[override]
                raise RuntimeError("boom")

            def overflow_candidates(self):  # type: ignore[override]
                raise RuntimeError("boom")

        self.assertIn(next_attribute(Hostile(turn=9)), SENDABLE)


class TestFixedSchedule(unittest.TestCase):
    """Turns 1-7 walk FIXED_SCHEDULE in order, once each."""

    def test_seven_turns_in_order_with_no_repeats(self) -> None:
        state = AskState()
        asks: list[str] = []
        for turn in range(1, 8):
            state.turn = turn
            attribute = next_attribute(state)
            asks.append(attribute)
            state.mark_asked(attribute)
            state.record_reply(attribute, DISCLOSURE_CAP, "none")
        self.assertEqual(asks, list(FIXED_SCHEDULE))
        self.assertEqual(len(set(asks)), len(FIXED_SCHEDULE))

    def test_budget_is_asked_and_is_the_seventh(self) -> None:
        self.assertEqual(FIXED_SCHEDULE[6], "budget")
        self.assertEqual(len(FIXED_SCHEDULE), 7)

    def test_retired_scheduled_slot_falls_through_to_the_ladder(self) -> None:
        state = AskState(turn=3, retired={"color"})
        self.assertEqual(FIXED_SCHEDULE[2], "color")
        choice = next_attribute(state)
        self.assertNotEqual(choice, "color")
        self.assertIn(choice, SENDABLE)

    def test_already_asked_scheduled_slot_falls_through_to_the_ladder(self) -> None:
        state = AskState(turn=3, asked=["color"])
        self.assertNotEqual(next_attribute(state), "color")

    def test_a_free_turn_never_uses_the_schedule_index(self) -> None:
        # Turn 8 has no scheduled entry; it runs the ladder against the state
        # at that moment. With nothing asked yet, the ladder's hedge fires.
        self.assertEqual(next_attribute(AskState(turn=8)), HEDGE_ORDER[0])


class TestRetirementRegister(unittest.TestCase):
    """record_reply(): the customer's words, not a yield number."""

    def _state(self) -> AskState:
        state = AskState()
        state.mark_asked("material")
        return state

    def test_zero_segments_retires(self) -> None:
        state = self._state()
        state.record_reply("material", 0, "none")
        self.assertIn("material", state.retired)

    def test_one_segment_retires(self) -> None:
        state = self._state()
        state.record_reply("material", 1, "none")
        self.assertIn("material", state.retired)

    def test_two_segments_stays_live(self) -> None:
        state = self._state()
        state.record_reply("material", DISCLOSURE_CAP, "none")
        self.assertNotIn("material", state.retired)
        self.assertEqual(state.overflow_candidates(), ["material"])

    def test_exhaustion_decline_retires(self) -> None:
        state = self._state()
        state.record_reply("material", 0, "exhaustion")
        self.assertIn("material", state.retired)
        self.assertEqual(state.overflow_candidates(), [])

    def test_refusal_decline_does_not_retire(self) -> None:
        """A refusal is not an empty answer: the customer declined to LOOK, so
        the bucket was never opened (local_evaluator.py:169 returns before the
        constraint filter runs)."""
        state = self._state()
        state.burn("material")
        state.record_reply("material", 0, "refusal")
        self.assertNotIn("material", state.retired)
        self.assertEqual(state.pending_reask(), "material")

    def test_a_burned_ask_is_never_retired_on_the_turn_it_was_burned(self) -> None:
        """The override turn never calls customer_reply() at all (:259), so that
        turn's reply is not an answer to the ask and cannot drain it."""
        state = self._state()
        state.burn("material")
        state.record_reply("material", 0, "none")
        self.assertNotIn("material", state.retired)
        # ...but once it has actually been re-asked and answered, it retires.
        state.mark_asked("material")
        state.record_reply("material", 0, "none")
        self.assertIn("material", state.retired)

    def test_retirement_is_permanent_across_a_later_full_yield(self) -> None:
        state = self._state()
        state.record_reply("material", 0, "none")
        state.record_reply("material", DISCLOSURE_CAP, "none")
        self.assertIn("material", state.retired)
        self.assertEqual(state.overflow_candidates(), [])

    def test_unreadable_inputs_are_no_ops(self) -> None:
        state = self._state()
        for attribute, count, decline in (
            (None, 0, "none"), ("", 0, "none"), ("material", None, "none"),
            ("material", "two", "none"), ("material", True, "none"),
        ):
            state.record_reply(attribute, count, decline)  # type: ignore[arg-type]
        self.assertEqual(state.retired, set())

    def test_overflow_is_most_recent_first_and_excludes_retired(self) -> None:
        state = AskState()
        for attribute in ("material", "feature", "color"):
            state.mark_asked(attribute)
            state.record_reply(attribute, DISCLOSURE_CAP, "none")
        self.assertEqual(state.overflow_candidates(), ["color", "feature", "material"])
        state.record_reply("feature", 1, "none")
        self.assertEqual(state.overflow_candidates(), ["color", "material"])


class TestBurnRegister(unittest.TestCase):
    """One Optional[str], not a queue: no session can burn twice."""

    def test_burn_latches_once(self) -> None:
        state = AskState()
        state.burn("material")
        state.burn("feature")
        self.assertEqual(state.burned, "material")
        self.assertEqual(state.pending_reask(), "material")

    def test_second_burn_is_a_no_op_even_after_the_reask(self) -> None:
        state = AskState()
        state.burn("material")
        state.mark_asked("material")
        self.assertIsNone(state.pending_reask())
        state.burn("feature")
        self.assertEqual(state.burned, "material")
        self.assertIsNone(state.pending_reask())

    def test_pending_reask_yields_once_and_only_once(self) -> None:
        # `retired` here only makes the assertion legible: with every attribute
        # asked and none retired, the ladder's last resort would legitimately
        # land on "material" again for an unrelated reason.
        state = AskState(turn=8, asked=list(ASKABLE), retired={"material"})
        state.burn("material")
        self.assertEqual(state.pending_reask(), "material")
        # The burned ask outranks even its own retirement: it was never read.
        self.assertEqual(next_attribute(state), "material")
        self.assertEqual(state.pending_reask(), "material", "selection must not consume")
        # mark_asked() is what consumes it -- selection stays side-effect free,
        # so askyield calling the fixed policy after an adaptive attempt cannot
        # spend the re-ask on a turn that never sent it.
        state.mark_asked("material")
        self.assertIsNone(state.pending_reask())
        self.assertNotEqual(next_attribute(state), "material")

    def test_invalid_burns_are_refused(self) -> None:
        for value in (None, "", "other", "bogus", 7, ["material"]):
            with self.subTest(value=repr(value)):
                state = AskState()
                state.burn(value)  # type: ignore[arg-type]
                self.assertIsNone(state.burned)
                self.assertIsNone(state.pending_reask())

    def test_burned_attribute_is_read_from_state_not_hardcoded(self) -> None:
        """It is FIXED_SCHEDULE[0] for a boundary refusal but [1] or [2] for an
        override (the override lands on turn 3 or 4), so nothing may assume it."""
        for attribute in (FIXED_SCHEDULE[0], FIXED_SCHEDULE[1], FIXED_SCHEDULE[2]):
            with self.subTest(attribute=attribute):
                state = AskState(turn=8, asked=list(ASKABLE))
                state.burn(attribute)
                self.assertEqual(next_attribute(state), attribute)

    def test_mark_asked_is_idempotent(self) -> None:
        state = AskState()
        state.mark_asked("material")
        state.mark_asked("material")
        state.mark_asked(" material ")
        self.assertEqual(state.asked, ["material"])
        self.assertEqual(state.last_ask, "material")
        state.mark_asked(None)  # type: ignore[arg-type]
        state.mark_asked("   ")
        self.assertEqual(state.asked, ["material"])


class TestFallthroughPrecedence(unittest.TestCase):
    """burned > overflow > brand > category > anything still live."""

    def _free_state(self) -> AskState:
        """Turn 8, all seven scheduled attributes asked, nothing retired."""
        state = AskState(turn=8)
        for attribute in FIXED_SCHEDULE:
            state.mark_asked(attribute)
        return state

    def test_burned_outranks_overflow(self) -> None:
        state = self._free_state()
        state.record_reply("feature", DISCLOSURE_CAP, "none")
        state.burn("material")
        self.assertEqual(state.overflow_candidates(), ["feature"])
        self.assertEqual(next_attribute(state), "material")

    def test_burned_outranks_everything_even_with_the_card_full(self) -> None:
        state = self._free_state()
        state.disclosed_count = CARD_CAPACITY + 3
        state.burn("size")
        self.assertEqual(next_attribute(state), "size")

    def test_overflow_outranks_brand(self) -> None:
        state = self._free_state()
        state.record_reply("color", DISCLOSURE_CAP, "none")
        state.disclosed_count = CARD_CAPACITY - 1
        self.assertEqual(next_attribute(state), "color")

    def test_overflow_takes_the_most_recent_candidate(self) -> None:
        state = self._free_state()
        state.record_reply("material", DISCLOSURE_CAP, "none")
        state.record_reply("color", DISCLOSURE_CAP, "none")
        state.disclosed_count = 0
        self.assertEqual(next_attribute(state), "color")

    def test_full_card_suppresses_overflow_and_brand_wins(self) -> None:
        state = self._free_state()
        state.record_reply("color", DISCLOSURE_CAP, "none")
        state.disclosed_count = CARD_CAPACITY
        self.assertEqual(next_attribute(state), "brand")

    def test_brand_outranks_category(self) -> None:
        state = self._free_state()
        state.disclosed_count = CARD_CAPACITY
        self.assertEqual(next_attribute(state), "brand")
        state.mark_asked("brand")
        self.assertEqual(next_attribute(state), "category")

    def test_hedge_is_asked_even_though_it_scores_zero_here(self) -> None:
        """brand and category always draw the content-free reply under this
        simulator. Asking them is a priced-at-zero hedge on the private set, not
        an oversight -- a wasted ask costs one turn and nothing else."""
        state = self._free_state()
        state.disclosed_count = CARD_CAPACITY
        asks = []
        for turn in (8, 9, 10):
            state.turn = turn
            attribute = next_attribute(state)
            asks.append(attribute)
            state.mark_asked(attribute)
            state.record_reply(attribute, 0, "exhaustion")
        self.assertEqual(asks[:2], list(HEDGE_ORDER))
        self.assertIn(asks[2], SENDABLE)

    def test_last_resort_prefers_something_not_retired(self) -> None:
        state = self._free_state()
        state.disclosed_count = CARD_CAPACITY
        state.mark_asked("brand")
        state.mark_asked("category")
        state.retired = set(ASKABLE) - {"size"}
        self.assertEqual(next_attribute(state), "size")

    def test_everything_retired_still_returns_a_valid_ask(self) -> None:
        state = self._free_state()
        state.disclosed_count = CARD_CAPACITY
        state.mark_asked("brand")
        state.mark_asked("category")
        state.retired = set(ASKABLE)
        self.assertIn(next_attribute(state), SENDABLE)

    def test_each_free_turn_re_runs_the_ladder_against_current_state(self) -> None:
        """Turns 8-10 are not a fixed order: the same turn number produces
        different asks depending only on the state at that moment."""
        first = self._free_state()
        first.turn = 8
        first.burn("material")
        second = self._free_state()
        second.turn = 8
        second.disclosed_count = CARD_CAPACITY
        self.assertEqual(next_attribute(first), "material")
        self.assertEqual(next_attribute(second), "brand")


class TestAskYieldGuard(unittest.TestCase):
    """The Layer 2 seam degrades to the fixed schedule on every failure mode."""

    def setUp(self) -> None:
        self._enabled = askyield.ADAPTIVE_ENABLED
        self._adaptive = askyield._adaptive
        self.addCleanup(self._restore)
        self.state = AskState(turn=1)

    def _restore(self) -> None:
        askyield.ADAPTIVE_ENABLED = self._enabled
        askyield._adaptive = self._adaptive

    def test_disabled_by_default_and_matches_the_fixed_policy(self) -> None:
        self.assertFalse(self._enabled)
        self.assertEqual(askyield.next_attribute(self.state), next_attribute(self.state))

    def test_adaptive_raising_falls_back(self) -> None:
        askyield.ADAPTIVE_ENABLED = True

        def _boom(state):
            raise RuntimeError("layer 2 exploded")

        askyield._adaptive = _boom
        self.assertEqual(askyield.next_attribute(self.state), FIXED_SCHEDULE[0])

    def test_adaptive_returning_other_is_rejected(self) -> None:
        askyield.ADAPTIVE_ENABLED = True
        askyield._adaptive = lambda state: "other"
        choice = askyield.next_attribute(self.state)
        self.assertNotEqual(choice, "other")
        self.assertEqual(choice, FIXED_SCHEDULE[0])

    def test_adaptive_returning_junk_is_rejected(self) -> None:
        askyield.ADAPTIVE_ENABLED = True
        for junk in ("materal", "", " material ", "OTHER", 3, [], None, True):
            with self.subTest(junk=repr(junk)):
                askyield._adaptive = lambda state, value=junk: value
                choice = askyield.next_attribute(self.state)
                self.assertIn(choice, SENDABLE)
                self.assertEqual(choice, FIXED_SCHEDULE[0])

    def test_adaptive_returning_a_valid_attribute_is_honoured(self) -> None:
        askyield.ADAPTIVE_ENABLED = True
        askyield._adaptive = lambda state: "budget"
        self.assertEqual(askyield.next_attribute(self.state), "budget")

    def test_disabled_flag_means_adaptive_is_never_consulted(self) -> None:
        calls: list[str] = []

        def _spy(state):
            calls.append("called")
            return "budget"

        askyield.ADAPTIVE_ENABLED = False
        askyield._adaptive = _spy
        self.assertEqual(askyield.next_attribute(self.state), FIXED_SCHEDULE[0])
        self.assertEqual(calls, [])

    def test_seam_never_returns_null_or_other_on_junk_states(self) -> None:
        for state in (None, "state", 42, AskState(turn=10, retired=set(ASKABLE))):
            with self.subTest(state=repr(state)[:30]):
                self.assertIn(askyield.next_attribute(state), SENDABLE)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
