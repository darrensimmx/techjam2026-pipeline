"""Per-turn orchestration: never-repeat, backfill, the override guard.  [WS-E]

Everything the pipeline orchestrates belongs to another workstream, so these
tests supply their own: a FakeIndex with a scripted pool, a FakeLedger that is
functional rather than inert, and a scripted sequence of Decode objects patched
in over Tier 1. That is not just isolation for its own sake -- it means a red
test here is an ORCHESTRATION bug, never a retrieval or parser bug, and the
suite stays green regardless of when the other workstreams land.

The two tests that matter most:

  - BACKFILL. A six-product pool must still return six picks on turn 10, not
    zero. This is exactly the case a filter-based shown-set breaks, and the
    evaluator reports the resulting empty list as a perfectly ordinary miss.
  - THE OVERRIDE GUARD. The evaluator's hit check is off for the early turns of
    an intent_override session (local_evaluator.py:234, :252), so products shown
    then are NOT confirmed wrong. Suppressing the recording and restoring the
    set is worth roughly five or six sessions.
"""
from __future__ import annotations

import unittest
from typing import Sequence
from unittest import mock

from src.pipeline import Deps, run_turn
from src.session import new_session
from src.shown import ShownRegistry
from src.types import ALLOWED_ATTRIBUTES, FORBIDDEN_ASK, Candidate, Decode, TurnPlan


# --------------------------------------------------------------------------
# Fakes. Hand-built, so the other five workstreams cannot turn this file red.
# --------------------------------------------------------------------------

class FakeIndex:
    """A scripted pool. Query-blind on purpose: what is under test here is what
    the turn does with a pool, not how the pool was chosen."""

    def __init__(self, count: int = 100, prefix: str = "P") -> None:
        self.pool = [Candidate(parent_asin=f"{prefix}{index:04d}", rowid=index,
                               rank=index + 1, score=-float(index))
                     for index in range(count)]
        self.queries: list[str] = []

    @property
    def size(self) -> int:
        return len(self.pool)

    def is_empty(self) -> bool:
        return not self.pool

    def search(self, query_text: str, limit: int) -> list[Candidate]:
        self.queries.append(query_text)
        return list(self.pool[:limit])

    def hydrate(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        # Identity, and `text` stays "" -- which also makes overlap.gate a
        # stable no-op whatever WS-D lands, since nothing can overlap "".
        return list(candidates)


class ExplodingIndex:
    def search(self, query_text: str, limit: int):
        raise RuntimeError("index is on fire")

    def hydrate(self, candidates):
        raise RuntimeError("index is still on fire")


class FakeLedger:
    """Duck-types ConstraintLedger and actually works, so these tests assert on
    real ledger content today rather than waiting for WS-B. It records the calls
    it received, which is how "the restore never touches the ledger" is checked.

    Note what it does NOT have: clear, remove, replace, pop. Neither does the
    real one -- "never erased, not even on intent override" is enforced by the
    absence of an API, so a fake that grew one would be testing a fiction.
    """

    def __init__(self) -> None:
        self._entries: list[str] = []
        self._segments: list[str] = []
        self.appends: list[str] = []
        self.segment_calls: list[tuple[str, ...]] = []

    def append(self, payload: str) -> None:
        self.appends.append(payload)
        if isinstance(payload, str) and payload.strip():
            self._entries.append(payload)

    def record_segments(self, segments: Sequence[str]) -> None:
        self.segment_calls.append(tuple(segments))
        for segment in segments:
            if segment not in self._segments:
                self._segments.append(segment)

    @property
    def query(self) -> str:
        return " ".join(self._entries).strip()

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def segments(self) -> tuple[str, ...]:
        return tuple(self._segments)

    def distinct_segment_count(self) -> int:
        return len(set(self._segments))

    def __len__(self) -> int:
        return len(self._entries)


class SpyShown(ShownRegistry):
    """The real registry plus a call log, so the override guard's sequencing is
    observable and not merely inferred from the picks."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def suppress(self) -> None:
        self.calls.append("suppress")
        super().suppress()

    def release(self) -> None:
        self.calls.append("release")
        super().release()

    def restore_all(self) -> None:
        self.calls.append("restore_all")
        super().restore_all()

    def record(self, parent_asins) -> None:
        self.calls.append("record")
        super().record(parent_asins)


# --------------------------------------------------------------------------
# Scripted decodes. Built directly -- Decode is a frozen dataclass.
# --------------------------------------------------------------------------

def buying_open(constraint: str = "leather upper") -> Decode:
    message = f"I'm looking for boots. A key requirement is: {constraint}."
    return Decode(frame="buying_open", payload=message, segments=(constraint,),
                  scenario_signal="buying")


def browsing_open() -> Decode:
    message = "I'm looking for boots, but I'm still exploring."
    return Decode(frame="browsing_open", payload=message, scenario_signal="browsing")


def override_open(old_value: str = "I prefer a relaxed fit.") -> Decode:
    message = f"I'm looking for boots. {old_value}"
    return Decode(frame="override_open", payload=message, scenario_signal="intent_override")


def override(new_value: str = "waterproof leather") -> Decode:
    message = f"Actually, ignore my earlier preference. What I need is: {new_value}."
    return Decode(frame="override", payload=message, segments=(new_value,),
                  scenario_signal="intent_override")


def disclosure(*segments: str) -> Decode:
    body = "; ".join(segments)
    return Decode(frame="disclosure", payload=f"For that, what matters is: {body}.",
                  segments=tuple(segments))


def refusal(attribute: str = "material") -> Decode:
    return Decode(frame="refusal", payload="", decline="refusal", attribute=attribute,
                  scenario_signal="boundary")


def exhaustion(attribute: str = "color") -> Decode:
    return Decode(frame="exhaustion", payload="", decline="exhaustion", attribute=attribute)


def null_nudge() -> Decode:
    return Decode(frame="null_nudge", payload="")


def unknown(text: str = "hmm") -> Decode:
    return Decode(frame="unknown", payload=text, source="none")


class Script:
    """Replaces frames.decode. Repeats the last entry once the script runs out,
    which is what a ten-turn losing session actually looks like."""

    def __init__(self, decodes: Sequence[Decode]) -> None:
        self.decodes = list(decodes)
        self.calls: list[str] = []

    def __call__(self, message: object) -> Decode:
        self.calls.append(message if isinstance(message, str) else "")
        index = min(len(self.calls) - 1, len(self.decodes) - 1)
        return self.decodes[index]


class PipelineCase(unittest.TestCase):
    def build(self, index=None, ledger=None, shown=None, deps=None):
        session = new_session("session-1", {"preference_tags": ["fit"]})
        session.ledger = ledger if ledger is not None else FakeLedger()
        if shown is not None:
            session.shown = shown
        return session, (deps if deps is not None else Deps(index=index))

    def play(self, session, deps, decodes, turns=10, top_k=10, messages=None):
        """Run `turns` turns with Tier 1 scripted. Returns the TurnPlans."""
        script = Script(decodes)
        plans: list[TurnPlan] = []
        with mock.patch("src.frames.decode", new=script):
            for turn in range(1, turns + 1):
                message = messages[turn - 1] if messages else f"customer message {turn}"
                plans.append(run_turn(session, message, turn, top_k, deps))
        self.script = script
        return plans


# --------------------------------------------------------------------------
# The never-repeat frontier.
# --------------------------------------------------------------------------

class TestNeverRepeat(PipelineCase):
    def test_no_product_is_shown_in_two_turns(self):
        session, deps = self.build(index=FakeIndex(100))
        plans = self.play(session, deps, [buying_open(), disclosure("cotton lining")])

        every_pick = [asin for plan in plans for asin in plan.parent_asins]
        self.assertEqual(len(every_pick), 100, "ten full turns of ten")
        self.assertEqual(len(set(every_pick)), 100, "a repeat means the frontier stalled")

    def test_a_session_walks_the_whole_pool_in_order(self):
        session, deps = self.build(index=FakeIndex(100))
        plans = self.play(session, deps, [buying_open()])

        self.assertEqual(plans[0].parent_asins,
                         tuple(f"P{index:04d}" for index in range(10)))
        self.assertEqual(plans[9].parent_asins,
                         tuple(f"P{index:04d}" for index in range(90, 100)))

    def test_picks_never_contain_a_duplicate(self):
        """normalize_recommendations() drops a repeated parent_asin silently, so
        a duplicate costs a slot out of the ten with no error anywhere."""
        session, deps = self.build(index=FakeIndex(100))
        for plan in self.play(session, deps, [buying_open()]):
            self.assertEqual(len(set(plan.parent_asins)), len(plan.parent_asins))


# --------------------------------------------------------------------------
# The backfill. The one a filter() breaks.
# --------------------------------------------------------------------------

class TestBackfill(PipelineCase):
    def test_a_six_product_pool_still_returns_six_every_turn(self):
        session, deps = self.build(index=FakeIndex(6))
        plans = self.play(session, deps, [buying_open()], turns=10)

        for turn, plan in enumerate(plans, start=1):
            with self.subTest(turn=turn):
                self.assertEqual(len(plan.parent_asins), 6,
                                 "the pool drained and the list went short")

    def test_the_drained_turns_re_show_the_same_six(self):
        session, deps = self.build(index=FakeIndex(6))
        plans = self.play(session, deps, [buying_open()], turns=10)
        self.assertEqual(set(plans[0].parent_asins), set(plans[9].parent_asins))

    def test_a_pool_shorter_than_top_k_is_not_padded_with_junk(self):
        session, deps = self.build(index=FakeIndex(3))
        plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertEqual(plan.parent_asins, ("P0000", "P0001", "P0002"))

    def test_an_empty_pool_is_an_empty_list_not_a_crash(self):
        session, deps = self.build(index=FakeIndex(0))
        for plan in self.play(session, deps, [buying_open()], turns=3):
            self.assertEqual(plan.parent_asins, ())
            self.assertIn(plan.ask_attribute, ALLOWED_ATTRIBUTES)


# --------------------------------------------------------------------------
# The override guard.
# --------------------------------------------------------------------------

class TestOverrideGuard(PipelineCase):
    def script_for_override_at_turn_four(self):
        return [override_open(), disclosure("cotton lining"), disclosure("crew neck"),
                override("waterproof leather"), disclosure("size 9")]

    def test_nothing_is_recorded_before_the_override_lands(self):
        shown = SpyShown()
        session, deps = self.build(index=FakeIndex(100), shown=shown)
        plans = self.play(session, deps, self.script_for_override_at_turn_four(), turns=3)

        self.assertIn("suppress", shown.calls)
        self.assertEqual(len(shown), 0, "the hit check is off; nothing is confirmed wrong")
        self.assertTrue(shown.suppressed)
        # Every early turn therefore shows the true top ten, so the first
        # SCORED turn tests our best list rather than our second-best.
        self.assertEqual(plans[0].parent_asins, plans[1].parent_asins)
        self.assertEqual(plans[1].parent_asins, plans[2].parent_asins)

    def test_the_override_restores_the_set_and_re_offers_turn_one(self):
        shown = SpyShown()
        session, deps = self.build(index=FakeIndex(100), shown=shown)
        plans = self.play(session, deps, self.script_for_override_at_turn_four(), turns=4)

        self.assertIn("restore_all", shown.calls)
        self.assertFalse(shown.suppressed)
        self.assertTrue(session.override_applied)
        self.assertEqual(set(plans[3].parent_asins), set(plans[0].parent_asins))
        self.assertEqual(len(plans[3].parent_asins), 10)

    def test_the_restore_does_not_touch_the_ledger(self):
        """Different set, different rule. The shown-set is restored on override;
        the ledger is never erased -- the abandoned preference was generated
        from the same target listing and still describes it."""
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(100), shown=SpyShown(), ledger=ledger)
        self.play(session, deps, self.script_for_override_at_turn_four(), turns=4)

        self.assertEqual(len(ledger.entries), 4, "one verbatim append per turn, none lost")
        self.assertIn("I prefer a relaxed fit.", ledger.query)
        self.assertIn("cotton lining", ledger.query)
        self.assertIn("waterproof leather", ledger.query)

    def test_the_override_turn_still_appends_and_still_asks(self):
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(100), ledger=ledger)
        plans = self.play(session, deps, self.script_for_override_at_turn_four(), turns=4)

        self.assertIn("What I need is: waterproof leather.", ledger.entries[-1])
        self.assertIn(plans[3].ask_attribute, ALLOWED_ATTRIBUTES)
        self.assertNotIn(plans[3].ask_attribute, FORBIDDEN_ASK)

    def test_an_override_at_turn_three_is_scored_on_the_best_list(self):
        shown = SpyShown()
        session, deps = self.build(index=FakeIndex(100), shown=shown)
        script = [override_open(), disclosure("cotton lining"), override("waterproof leather"),
                  disclosure("size 9")]
        plans = self.play(session, deps, script, turns=4)

        self.assertEqual(set(plans[2].parent_asins), set(plans[0].parent_asins))
        self.assertFalse(shown.suppressed)
        self.assertEqual(len(shown), 20,
                         "recording resumed on the override turn and kept going")

    def test_a_false_positive_self_heals_after_turn_three(self):
        """behavior_for() never picks an override turn later than 4, so a
        session still suppressed at turn 4 was a turn-1 misread."""
        shown = SpyShown()
        session, deps = self.build(index=FakeIndex(100), shown=shown)
        plans = self.play(session, deps, [override_open(), disclosure("cotton lining")], turns=6)

        self.assertFalse(shown.suppressed)
        self.assertIn("release", shown.calls)
        self.assertEqual(len(shown), 30, "turns 4, 5 and 6 each recorded their ten")
        self.assertEqual(plans[0].parent_asins, plans[3].parent_asins,
                         "turns 1-4 all show the true top ten")
        self.assertEqual(len(set(plans[3].parent_asins) & set(plans[4].parent_asins)), 0,
                         "and the frontier moves again from turn 5")

    def test_a_non_override_session_records_from_turn_one(self):
        shown = SpyShown()
        session, deps = self.build(index=FakeIndex(100), shown=shown)
        self.play(session, deps, [buying_open(), disclosure("cotton lining")], turns=1)

        self.assertNotIn("suppress", shown.calls)
        self.assertEqual(len(shown), 10)


# --------------------------------------------------------------------------
# The ledger. Verbatim in, and it IS the query.
# --------------------------------------------------------------------------

class TestLedger(PipelineCase):
    def test_content_free_frames_leave_the_query_byte_identical(self):
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(100), ledger=ledger)
        script = [buying_open(), refusal("material"), null_nudge(), exhaustion("color")]
        self.play(session, deps, script, turns=4)

        before = " ".join(ledger.entries[:1])
        self.assertEqual(ledger.query, before)
        self.assertEqual(len(ledger.entries), 1)
        self.assertEqual([payload for payload in ledger.appends if payload.strip()],
                         list(ledger.entries))

    def test_the_query_is_the_ledger(self):
        ledger = FakeLedger()
        index = FakeIndex(100)
        session, deps = self.build(index=index, ledger=ledger)
        self.play(session, deps, [buying_open(), disclosure("cotton lining")], turns=2)

        self.assertEqual(index.queries[-1], ledger.query)
        self.assertIn("leather upper", index.queries[-1])
        self.assertIn("cotton lining", index.queries[-1])

    def test_an_empty_ledger_falls_back_to_the_raw_message(self):
        """Turn 1 of a browsing session discloses nothing, so the opener is all
        there is -- but an empty query would search nothing at all."""
        ledger = FakeLedger()
        index = FakeIndex(100)
        session, deps = self.build(index=index, ledger=ledger)

        class SilentLedger(FakeLedger):
            def append(self, payload):
                self.appends.append(payload)

        session.ledger = SilentLedger()
        opener = "I'm looking for boots, but I'm still exploring."
        self.play(session, deps, [browsing_open()], turns=1, messages=[opener])
        self.assertEqual(index.queries[-1], opener)

    def test_segments_reach_the_ledger(self):
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(100), ledger=ledger)
        self.play(session, deps, [buying_open(), disclosure("cotton lining", "crew neck")],
                  turns=2)
        self.assertEqual(ledger.segments, ("leather upper", "cotton lining", "crew neck"))


# --------------------------------------------------------------------------
# The ask. Never None, never "other".
# --------------------------------------------------------------------------

class TestAsk(PipelineCase):
    SCRIPTS = {
        "buying": [buying_open(), disclosure("cotton lining")],
        "browsing": [browsing_open(), disclosure("crew neck")],
        "boundary": [buying_open(), refusal("material"), disclosure("cotton lining")],
        "intent_override": [override_open(), disclosure("cotton lining"),
                            disclosure("crew neck"), override("waterproof leather"),
                            disclosure("size 9")],
    }

    def test_every_turn_of_every_scenario_asks_something_valid(self):
        for scenario, script in self.SCRIPTS.items():
            session, deps = self.build(index=FakeIndex(100))
            plans = self.play(session, deps, script, turns=10)
            for turn, plan in enumerate(plans, start=1):
                with self.subTest(scenario=scenario, turn=turn):
                    self.assertIsInstance(plan, TurnPlan)
                    self.assertIsNotNone(plan.ask_attribute)
                    self.assertIn(plan.ask_attribute, ALLOWED_ATTRIBUTES)
                    self.assertNotIn(plan.ask_attribute, FORBIDDEN_ASK)
                    self.assertNotEqual(plan.ask_attribute, "other")

    def test_the_ask_survives_a_policy_that_returns_junk(self):
        for junk in (None, "", "other", "nonsense", 7, object()):
            with self.subTest(junk=repr(junk)):
                session, deps = self.build(index=FakeIndex(20))
                with mock.patch("src.askyield.next_attribute", return_value=junk):
                    plans = self.play(session, deps, [buying_open()], turns=3)
                for plan in plans:
                    self.assertIn(plan.ask_attribute, ALLOWED_ATTRIBUTES)
                    self.assertNotIn(plan.ask_attribute, FORBIDDEN_ASK)

    def test_the_ask_survives_a_policy_that_raises(self):
        session, deps = self.build(index=FakeIndex(20))
        with mock.patch("src.askyield.next_attribute", side_effect=RuntimeError("boom")):
            plans = self.play(session, deps, [buying_open()], turns=3)
        for plan in plans:
            self.assertIn(plan.ask_attribute, ALLOWED_ATTRIBUTES)

    def test_a_decline_on_an_unknown_frame_still_burns_the_ask(self):
        """Tier 1.5 returns frame="unknown" with decline="refusal" for a decline
        it could not match to a template. Burning the ask is half of what that
        hedge is for, so the decline is read alongside the frame, not under it."""
        session, deps = self.build(index=FakeIndex(20))
        hedged = Decode(frame="unknown", payload="", decline="refusal")
        self.play(session, deps, [buying_open(), hedged], turns=2)

        self.assertEqual(session.asks.burned, "material", "turn 1's ask was burned")
        self.assertEqual(session.asks.pending_reask(), "material")
        self.assertNotIn("material", session.asks.retired, "a refusal never retires")

    def test_an_unknown_frame_carrying_exhaustion_retires_the_attribute(self):
        session, deps = self.build(index=FakeIndex(20))
        hedged = Decode(frame="unknown", payload="", decline="exhaustion")
        self.play(session, deps, [buying_open(), hedged], turns=2)
        self.assertIn("material", session.asks.retired)

    def test_the_message_is_prose_and_carries_the_question(self):
        session, deps = self.build(index=FakeIndex(20))
        plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertIsInstance(plan.message, str)
        self.assertTrue(plan.message.strip())
        self.assertTrue(plan.message.endswith("?"))


# --------------------------------------------------------------------------
# Tier 2 fires only on a Tier 1 unknown, and never overrides Tier 1.
# --------------------------------------------------------------------------

class TestTierTwo(PipelineCase):
    def test_is_not_consulted_when_tier_one_decoded(self):
        session, deps = self.build(index=FakeIndex(20))
        with mock.patch("src.semantic.safe_decode", return_value=None) as safe_decode:
            self.play(session, deps, [buying_open()], turns=3)
        safe_decode.assert_not_called()

    def test_is_consulted_on_unknown_and_its_decode_is_used(self):
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(20), ledger=ledger)
        rescued = Decode(frame="disclosure", payload="rescued by tier 2",
                         segments=("rescued",), source="tier2")
        with mock.patch("src.semantic.safe_decode", return_value=rescued) as safe_decode:
            self.play(session, deps, [unknown("mystery text")], turns=1)
        safe_decode.assert_called_once()
        self.assertEqual(ledger.entries, ("rescued by tier 2",))

    def test_an_abstention_leaves_the_tier_one_decode_alone(self):
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(20), ledger=ledger)
        with mock.patch("src.semantic.safe_decode", return_value=None):
            self.play(session, deps, [unknown("mystery text")], turns=1)
        self.assertEqual(ledger.entries, ("mystery text",))

    def test_a_tier_two_that_raises_costs_nothing(self):
        ledger = FakeLedger()
        session, deps = self.build(index=FakeIndex(20), ledger=ledger)
        with mock.patch("src.semantic.safe_decode", side_effect=RuntimeError("boom")):
            plans = self.play(session, deps, [unknown("mystery text")], turns=1)
        self.assertEqual(ledger.entries, ("mystery text",))
        self.assertEqual(len(plans[0].parent_asins), 10)


# --------------------------------------------------------------------------
# Hostile input. Every one of these runs on every turn of every session.
# --------------------------------------------------------------------------

class TestNeverRaises(PipelineCase):
    def assertUsable(self, plan):
        self.assertIsInstance(plan, TurnPlan)
        self.assertIsInstance(plan.message, str)
        self.assertIn(plan.ask_attribute, ALLOWED_ATTRIBUTES)
        self.assertNotIn(plan.ask_attribute, FORBIDDEN_ASK)
        self.assertIsInstance(plan.parent_asins, tuple)

    def test_hostile_messages(self):
        for message in (None, 5, b"bytes", ["a", "list"], object(), "", "   "):
            with self.subTest(message=repr(message)):
                session, deps = self.build(index=FakeIndex(20))
                script = Script([unknown("")])
                with mock.patch("src.frames.decode", new=script):
                    self.assertUsable(run_turn(session, message, 1, 10, deps))

    def test_hostile_turn_numbers(self):
        for turn in ("3", -1, 0, 999, 3.7, None, "not a turn", True, object()):
            with self.subTest(turn=repr(turn)):
                session, deps = self.build(index=FakeIndex(20))
                script = Script([buying_open()])
                with mock.patch("src.frames.decode", new=script):
                    plan = run_turn(session, "hello", turn, 10, deps)
                self.assertUsable(plan)
                self.assertEqual(len(plan.parent_asins), 10)
                self.assertGreaterEqual(session.turn, 1)
                self.assertLessEqual(session.turn, 10)

    def test_hostile_top_k(self):
        for top_k, expected in ((0, 0), (1, 1), (3, 3), (-5, 0), (None, 10), (True, 10),
                                ("10", 10), (1000, 20)):
            with self.subTest(top_k=repr(top_k)):
                session, deps = self.build(index=FakeIndex(20))
                script = Script([buying_open()])
                with mock.patch("src.frames.decode", new=script):
                    plan = run_turn(session, "hello", 1, top_k, deps)
                self.assertUsable(plan)
                self.assertEqual(len(plan.parent_asins), expected)

    def test_a_null_index_still_answers(self):
        session, deps = self.build(deps=Deps())
        plans = self.play(session, deps, [buying_open(), disclosure("cotton lining")], turns=10)
        for plan in plans:
            self.assertUsable(plan)
            self.assertEqual(plan.parent_asins, ())

    def test_an_index_that_raises_still_answers(self):
        session, deps = self.build(deps=Deps(index=ExplodingIndex()))
        for plan in self.play(session, deps, [buying_open()], turns=3):
            self.assertUsable(plan)
            self.assertEqual(plan.parent_asins, ())

    def test_hostile_session_and_deps(self):
        script = Script([buying_open()])
        with mock.patch("src.frames.decode", new=script):
            for session in (None, object(), "not a session"):
                with self.subTest(session=repr(session)):
                    self.assertUsable(run_turn(session, "hello", 1, 10, Deps()))
            self.assertUsable(run_turn(new_session("s", {}), "hello", 1, 10, None))
            self.assertUsable(run_turn(new_session("s", {}), "hello", 1, 10, "not deps"))

    def test_a_tier_one_decoder_that_raises_still_answers(self):
        session, deps = self.build(index=FakeIndex(20))
        with mock.patch("src.frames.decode", side_effect=RuntimeError("boom")):
            plan = run_turn(session, "hello", 1, 10, deps)
        self.assertUsable(plan)
        self.assertEqual(len(plan.parent_asins), 10)

    def test_a_decode_of_the_wrong_type_still_answers(self):
        session, deps = self.build(index=FakeIndex(20))
        with mock.patch("src.frames.decode", return_value={"frame": "disclosure"}):
            plan = run_turn(session, "hello", 1, 10, deps)
        self.assertUsable(plan)
        self.assertEqual(len(plan.parent_asins), 10)

    def test_collaborators_that_raise_cost_nothing(self):
        targets = ("src.overlap.gate", "src.rerank.safe_rerank", "src.slots.classify_local",
                   "src.slots.apply_override")
        for target in targets:
            with self.subTest(target=target):
                session, deps = self.build(index=FakeIndex(100))
                deps = Deps(index=deps.index, reranker=object())
                with mock.patch(target, side_effect=RuntimeError("boom")):
                    plans = self.play(session, deps,
                                      [buying_open(), override("waterproof leather")], turns=3)
                for plan in plans:
                    self.assertUsable(plan)
                    self.assertEqual(len(plan.parent_asins), 10)


# --------------------------------------------------------------------------
# Order-only stages stay order-only, whatever they hand back.
# --------------------------------------------------------------------------

class TestOrderOnlyStages(PipelineCase):
    def test_a_gate_that_drops_candidates_is_ignored(self):
        session, deps = self.build(index=FakeIndex(100))
        with mock.patch("src.overlap.gate", side_effect=lambda c, s: list(c)[:2]):
            plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertEqual(len(plan.parent_asins), 10, "a dropped candidate is retrieval")

    def test_a_gate_that_invents_candidates_is_ignored(self):
        session, deps = self.build(index=FakeIndex(100))
        forged = [Candidate(parent_asin=f"X{index}") for index in range(50)]
        with mock.patch("src.overlap.gate", return_value=forged):
            plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertEqual(plan.parent_asins, tuple(f"P{index:04d}" for index in range(10)))

    def test_a_genuine_reordering_is_honoured(self):
        session, deps = self.build(index=FakeIndex(100))
        with mock.patch("src.overlap.gate", side_effect=lambda c, s: list(reversed(list(c)))):
            plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertEqual(plan.parent_asins[0], "P0049")
        self.assertEqual(len(plan.parent_asins), 10)

    def test_a_hydrate_that_loses_candidates_is_ignored(self):
        class LossyIndex(FakeIndex):
            def hydrate(self, candidates):
                return list(candidates)[:1]

        session, deps = self.build(index=LossyIndex(100))
        plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertEqual(len(plan.parent_asins), 10)

    def test_a_partition_that_loses_candidates_is_ignored(self):
        class LossyShown(ShownRegistry):
            def partition(self, candidates):
                return list(candidates)[:1], []

        session, deps = self.build(index=FakeIndex(100), shown=LossyShown())
        plan = self.play(session, deps, [buying_open()], turns=1)[0]
        self.assertEqual(len(plan.parent_asins), 10)


# --------------------------------------------------------------------------
# Diagnostics.
# --------------------------------------------------------------------------

class TestDiagnostics(PipelineCase):
    def test_frame_counts_are_kept(self):
        session, deps = self.build(index=FakeIndex(20))
        self.play(session, deps, [buying_open(), disclosure("cotton lining")], turns=4)
        self.assertEqual(session.frame_counts.get("buying_open"), 1)
        self.assertEqual(session.frame_counts.get("disclosure"), 3)
        self.assertNotIn("null_nudge", session.frame_counts)

    def test_the_scenario_signal_is_recorded_and_never_downgraded(self):
        session, deps = self.build(index=FakeIndex(20))
        self.play(session, deps, [override_open(), disclosure("cotton lining")], turns=3)
        self.assertEqual(session.scenario, "intent_override")


if __name__ == "__main__":
    unittest.main()
