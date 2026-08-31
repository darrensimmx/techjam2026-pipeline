"""Tier 1 frame decode.  [WS-B]

The strongest test in this file is TestAgainstTheVendoredGenerator, and it is
written the way it is on purpose: it IMPORTS initial_message() and
customer_reply() out of evaluator/local_evaluator.py and drives them, rather
than hand-copying the eight f-strings into a fixture. Frame coverage is then a
property of the real harness. A hand-copied fixture would keep passing after
the harness changed a single word, which is exactly the failure it would exist
to catch.

The hand-written strings that do appear below are all in tests of
DISAMBIGUATION (which of two overlapping patterns wins) or of hostile input --
cases the generator cannot produce and where the literal text is the point.
"""
from __future__ import annotations

import unittest

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)
from src import frames
from src.frames import decode, is_content_free, normalise, split_disclosure
from src.types import CONTENT_FREE_FRAMES
from tests import synthetic

ALL_FRAMES = {
    "buying_open", "browsing_open", "override_open", "refusal",
    "null_nudge", "exhaustion", "disclosure", "override",
}


def build_sessions(per_scenario: int = 4):
    """Materialise hidden fields exactly the way evaluate() does.

    build_samples() deliberately carries no intent_card and no behavior, so
    materialize_hidden_fields() takes its fallback branch and manufactures the
    card out of the target product's own listing -- the real code path.
    """
    products = synthetic.build_products(n=80)
    by_id = {str(item["parent_asin"]): item for item in products}
    categories = {
        str(item["parent_asin"]): [str(value) for value in item["categories"]]
        for item in products
    }
    sessions = []
    for sample in synthetic.build_samples(products, per_scenario=per_scenario):
        card, behavior = materialize_hidden_fields(sample, by_id)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        target = str(sample["ground_truth"]["parent_asin"])
        sessions.append((effective, coarse_category(categories.get(target, []))))
    return sessions


class TestAgainstTheVendoredGenerator(unittest.TestCase):
    """Every string initial_message() and customer_reply() can emit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sessions = build_sessions()

    def test_every_initial_message_decodes(self) -> None:
        for sample, category in self.sessions:
            with self.subTest(scenario=sample["scenario_type"]):
                disclosed: set[str] = set()
                message = initial_message(sample, category, disclosed)
                result = decode(message)

                self.assertNotEqual(result.frame, "unknown", message)
                self.assertEqual(result.decline, "none")
                self.assertEqual(result.payload, message)
                self.assertIsNone(result.attribute)

                scenario = sample["scenario_type"]
                if scenario == "buying":
                    self.assertEqual(result.frame, "buying_open")
                    self.assertEqual(result.scenario_signal, "buying")
                    # The evaluator marks exactly the constraint it just spoke.
                    self.assertEqual(result.segments, tuple(disclosed))
                    self.assertEqual(len(disclosed), 1)
                elif scenario == "intent_override":
                    self.assertEqual(result.frame, "override_open")
                    self.assertEqual(result.scenario_signal, "intent_override")
                    old_value = str(sample["behavior"]["override"]["old_value"])
                    self.assertEqual(result.segments, (old_value,))
                    # The opener of an override session has NO trailing period:
                    # old_value came through _clean_constraint().
                    self.assertFalse(message.endswith("."), message)
                else:
                    # boundary falls through to the exploring template too.
                    self.assertEqual(result.frame, "browsing_open")
                    self.assertEqual(result.segments, ())

    def test_every_customer_reply_decodes(self) -> None:
        """Drive every reply branch: refusal, null nudge, disclosure, exhaustion.

        Each attribute is asked twice so that every bucket is driven past the
        point where the constraint filter comes back empty -- that is the only
        way to reach the exhaustion branch at local_evaluator.py:183.
        """
        seen_frames: set[str] = set()
        asks = sorted(ALLOWED_ATTRIBUTES) * 2 + [None]

        for sample, category in self.sessions:
            disclosed: set[str] = set()
            initial_message(sample, category, disclosed)
            boundary_used = False

            for ask in asks:
                before = set(disclosed)
                message, boundary_used = customer_reply(
                    sample, ask, disclosed, boundary_used)
                result = decode(message)
                seen_frames.add(result.frame)

                with self.subTest(scenario=sample["scenario_type"], ask=ask,
                                  message=message):
                    self.assertNotEqual(result.frame, "unknown", message)

                    if result.frame == "null_nudge":
                        self.assertIsNone(ask)
                        self.assertEqual(result.payload, "")
                        self.assertEqual(result.segments, ())
                        self.assertEqual(result.decline, "none")
                    elif result.frame == "refusal":
                        # Returns BEFORE the filter runs: nothing is spent.
                        self.assertEqual(sample["scenario_type"], "boundary")
                        self.assertEqual(result.decline, "refusal")
                        self.assertEqual(result.attribute, ask)
                        self.assertEqual(result.payload, "")
                        self.assertEqual(result.segments, ())
                        self.assertEqual(disclosed, before)
                    elif result.frame == "exhaustion":
                        # Returns AFTER the filter found nothing.
                        self.assertEqual(result.decline, "exhaustion")
                        self.assertEqual(result.attribute, ask)
                        self.assertEqual(result.payload, "")
                        self.assertEqual(result.segments, ())
                        self.assertEqual(disclosed, before)
                    elif result.frame == "disclosure":
                        self.assertEqual(result.decline, "none")
                        self.assertEqual(result.payload, message)
                        # Checked against the generator's own side effect,
                        # disclosed.update(matches), not against a re-parse.
                        self.assertEqual(set(result.segments), disclosed - before)
                        self.assertEqual(
                            len(result.segments), len(set(result.segments)))
                        for segment in result.segments:
                            self.assertIn(segment, message)
                    else:
                        self.fail(f"unexpected frame {result.frame!r}: {message!r}")

        self.assertEqual(
            seen_frames, {"refusal", "null_nudge", "exhaustion", "disclosure"})

    def test_override_message_decodes(self) -> None:
        """behavior_for()'s override sentence, plus evaluate()'s :264 fallback."""
        overrides = 0
        for sample, _category in self.sessions:
            override = sample.get("behavior", {}).get("override") or {}
            if not override:
                continue
            overrides += 1
            message = str(override["message"])
            result = decode(message)
            self.assertEqual(result.frame, "override")
            self.assertEqual(result.payload, message)
            self.assertEqual(result.segments, (str(override["new_value"]),))
            self.assertEqual(result.scenario_signal, "intent_override")
            self.assertEqual(result.decline, "none")
        self.assertGreater(overrides, 0)

        # evaluate():264's default, reached when a sample supplies its own
        # `behavior` with no override["message"]. A ninth emittable string.
        fallback = decode("Actually, please ignore my earlier preference.")
        self.assertEqual(fallback.frame, "override")
        self.assertEqual(fallback.scenario_signal, "intent_override")

    def test_all_eight_frames_are_reachable(self) -> None:
        """Union of both generators covers the whole enum but `unknown`."""
        seen: set[str] = set()
        for sample, category in self.sessions:
            disclosed: set[str] = set()
            seen.add(decode(initial_message(sample, category, disclosed)).frame)
            boundary_used = False
            for ask in sorted(ALLOWED_ATTRIBUTES) * 2 + [None]:
                message, boundary_used = customer_reply(
                    sample, ask, disclosed, boundary_used)
                seen.add(decode(message).frame)
            override = sample.get("behavior", {}).get("override") or {}
            if override:
                seen.add(decode(str(override["message"])).frame)
        self.assertEqual(seen, ALL_FRAMES)


class TestDeclineSplit(unittest.TestCase):
    """One token, two opposite meanings. The bug this design exists to fix."""

    REFUSAL = "I don't have a preference for use_case; please use your judgment."
    EXHAUSTION = "I don't have an additional preference for use_case."

    def test_refusal_is_not_exhaustion(self) -> None:
        result = decode(self.REFUSAL)
        self.assertEqual(result.frame, "refusal")
        self.assertEqual(result.decline, "refusal")
        self.assertEqual(result.attribute, "use_case")
        self.assertEqual(result.scenario_signal, "boundary")

    def test_exhaustion_is_not_refusal(self) -> None:
        result = decode(self.EXHAUSTION)
        self.assertEqual(result.frame, "exhaustion")
        self.assertEqual(result.decline, "exhaustion")
        self.assertEqual(result.attribute, "use_case")

    def test_the_two_differ_by_one_token(self) -> None:
        self.assertEqual(
            self.EXHAUSTION,
            "I don't have an additional preference for use_case.")
        # Same sentence minus `additional` (and the article) is the refusal stem.
        stem = self.EXHAUSTION.replace("an additional preference", "a preference")
        self.assertEqual(decode(stem).decline, "refusal")
        self.assertNotEqual(decode(stem).decline, decode(self.EXHAUSTION).decline)

    def test_every_attribute_name_survives_both_frames(self) -> None:
        for attribute in sorted(ALLOWED_ATTRIBUTES):
            refusal = decode(
                f"I don't have a preference for {attribute}; please use your judgment.")
            exhaustion = decode(
                f"I don't have an additional preference for {attribute}.")
            self.assertEqual(refusal.decline, "refusal", attribute)
            self.assertEqual(exhaustion.decline, "exhaustion", attribute)
            self.assertEqual(refusal.attribute, attribute)
            self.assertEqual(exhaustion.attribute, attribute)


class TestOpenerDisambiguation(unittest.TestCase):
    """1 -> 3 -> 2: frame 1 is a special case of frame 2; frame 3 has the comma."""

    def test_buying_opener(self) -> None:
        message = ("I'm looking for Shoes & Jewelry Boots. "
                   "A key requirement is: leather upper.")
        result = decode(message)
        self.assertEqual(result.frame, "buying_open")
        self.assertEqual(result.segments, ("leather upper",))
        self.assertEqual(result.scenario_signal, "buying")

    def test_buying_constraint_may_contain_internal_periods(self) -> None:
        message = ("I'm looking for Boots. "
                   "A key requirement is: budget around $29.99.")
        result = decode(message)
        self.assertEqual(result.frame, "buying_open")
        self.assertEqual(result.segments, ("budget around $29.99",))

    def test_browsing_opener_is_told_apart_by_its_comma(self) -> None:
        message = "I'm looking for Shoes & Jewelry Boots, but I'm still exploring."
        result = decode(message)
        self.assertEqual(result.frame, "browsing_open")
        self.assertEqual(result.segments, ())
        self.assertEqual(result.scenario_signal, "browsing")

    def test_override_opener_has_no_trailing_period(self) -> None:
        """The regression this ordering exists for.

        old_value is soft_preferences[-1], already stripped of trailing ` -;,.`
        by _clean_constraint(), so the real message ends mid-word. A pattern
        demanding a final `.` misses every intent_override session's turn 1.
        """
        message = "I'm looking for Shoes & Jewelry Boots. casual cut"
        result = decode(message)
        self.assertEqual(result.frame, "override_open")
        self.assertEqual(result.segments, ("casual cut",))
        self.assertEqual(result.scenario_signal, "intent_override")

    def test_override_opener_old_value_may_contain_a_period(self) -> None:
        message = "I'm looking for Boots. budget around $29.99"
        result = decode(message)
        self.assertEqual(result.frame, "override_open")
        self.assertEqual(result.segments, ("budget around $29.99",))

    def test_the_three_openers_are_mutually_exclusive(self) -> None:
        frames = {
            decode("I'm looking for Boots. A key requirement is: leather.").frame,
            decode("I'm looking for Boots, but I'm still exploring.").frame,
            decode("I'm looking for Boots. casual cut").frame,
        }
        self.assertEqual(frames, {"buying_open", "browsing_open", "override_open"})


class TestPayloadIsTheOriginalMessage(unittest.TestCase):
    """Match on a collapsed copy; hand back the untouched string."""

    CONTENT_BEARING = (
        "I'm looking for Boots. A key requirement is: leather upper.",
        "I'm looking for Boots. casual cut",
        "I'm looking for Boots, but I'm still exploring.",
        "For that, what matters is: cotton; budget around $29.99.",
        "Actually, ignore my earlier preference. What I need is: leather upper.",
        "some sentence the simulator has never emitted",
    )

    def test_payload_is_byte_identical(self) -> None:
        for message in self.CONTENT_BEARING:
            with self.subTest(message=message):
                self.assertEqual(decode(message).payload, message)

    def test_payload_survives_irregular_whitespace(self) -> None:
        message = "  I'm looking for  Boots.\tA key requirement is:  leather upper. "
        result = decode(message)
        self.assertEqual(result.frame, "buying_open")
        self.assertEqual(result.payload, message)      # byte-identical, not trimmed
        self.assertEqual(result.segments, ("leather upper",))

    def test_content_free_frames_carry_nothing(self) -> None:
        for message in (
            "I don't have a preference for color; please use your judgment.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            "I don't have an additional preference for color.",
        ):
            with self.subTest(message=message):
                result = decode(message)
                self.assertIn(result.frame, CONTENT_FREE_FRAMES)
                self.assertEqual(result.payload, "")
                self.assertEqual(result.segments, ())
                self.assertTrue(is_content_free(message))

    def test_content_bearing_frames_are_not_content_free(self) -> None:
        for message in self.CONTENT_BEARING:
            with self.subTest(message=message):
                self.assertFalse(is_content_free(message))


class TestAnchoring(unittest.TestCase):
    """Patterns are whole-message templates, not substring sniffs."""

    def test_a_frame_phrase_mid_message_does_not_fire(self) -> None:
        message = "Thanks. For that, what matters is: leather."
        result = decode(message)
        self.assertEqual(result.frame, "unknown")
        self.assertEqual(result.payload, message)      # still reaches the ledger
        self.assertEqual(result.segments, ())

    def test_an_opener_mid_message_does_not_fire(self) -> None:
        message = "Hi there -- I'm looking for Boots, but I'm still exploring."
        self.assertEqual(decode(message).frame, "unknown")
        self.assertEqual(decode(message).payload, message)


class TestTier15Hedge(unittest.TestCase):
    """Unknown + an unanchored decline sniff -> bias toward refusal.

    Deliberately asymmetric. A wrong `refusal` costs one re-ask on an otherwise
    idle turn; a wrong `exhaustion` retires a live bucket for the whole session.
    """

    def test_paraphrased_decline_is_read_as_a_refusal(self) -> None:
        for message in (
            "Honestly I dont have any strong preference on that one",
            "I don't have a strong preference there, sorry!",
        ):
            with self.subTest(message=message):
                result = decode(message)
                self.assertEqual(result.frame, "unknown")
                self.assertEqual(result.decline, "refusal")
                self.assertEqual(result.payload, "")

    def test_hedge_does_not_fire_on_ordinary_text(self) -> None:
        result = decode("I would like something warm for winter walks")
        self.assertEqual(result.frame, "unknown")
        self.assertEqual(result.decline, "none")
        self.assertEqual(result.payload, "I would like something warm for winter walks")

    def test_hedge_never_produces_an_exhaustion(self) -> None:
        # Exhaustion retires permanently, so Tier 1.5 must never guess it.
        result = decode("no additional preference i think, whatever you like")
        self.assertNotEqual(result.decline, "exhaustion")

    def test_the_hedge_is_the_one_place_anchoring_is_given_up(self) -> None:
        """And it is switchable, so the cost is visible rather than buried.

        A decline phrase after real content is content-bearing under pure
        anchoring; the hedge overrides that and drops the payload. Today's
        simulator cannot emit such a string -- this only bites a paraphrased
        private set, where a missed decline is the more expensive error.
        """
        message = "I want cotton boots. I don't have a preference for color."
        self.assertTrue(frames.TIER_15_HEDGE)
        self.assertEqual(decode(message).payload, "")

        frames.TIER_15_HEDGE = False
        try:
            result = decode(message)
            self.assertEqual(result.payload, message)
            self.assertEqual(result.decline, "none")
        finally:
            frames.TIER_15_HEDGE = True


class TestHostileInput(unittest.TestCase):
    """respond() may not raise; neither may anything under it."""

    CASES = (
        None, "", "   ", 123, 0, 12.5, True, [], {}, (), b"bytes", object(),
        "\x00\x01", "\n\t\r", "%s %d {} {0}", "*" * 64, "'; DROP TABLE --",
        "MATCH NEAR() OR AND", "([)]{", "\\", "😀 emoji",
        "I'm looking for " + "x" * 20000,
        "x" * 20000,
        "For that, what matters is: " + "; ".join(["a"] * 500) + ".",
    )

    def test_decode_never_raises_and_always_returns_a_decode(self) -> None:
        for case in self.CASES:
            with self.subTest(case=repr(case)[:60]):
                result = decode(case)
                self.assertIsInstance(result.frame, str)
                self.assertIsInstance(result.payload, str)
                self.assertIsInstance(result.segments, tuple)
                self.assertIn(result.decline, ("none", "refusal", "exhaustion"))
                self.assertTrue(
                    result.attribute is None or isinstance(result.attribute, str))

    def test_non_strings_decode_to_an_empty_unknown(self) -> None:
        for case in (None, 123, 12.5, [], {}, b"bytes", object()):
            with self.subTest(case=repr(case)[:60]):
                result = decode(case)
                self.assertEqual(result.frame, "unknown")
                self.assertEqual(result.payload, "")

    def test_helpers_never_raise(self) -> None:
        for case in self.CASES:
            with self.subTest(case=repr(case)[:60]):
                self.assertIsInstance(normalise(case), str)
                self.assertIsInstance(is_content_free(case), bool)
                self.assertIsInstance(split_disclosure(case), tuple)


class TestSplitDisclosure(unittest.TestCase):
    def test_splits_and_strips(self) -> None:
        self.assertEqual(split_disclosure("a; b"), ("a", "b"))
        self.assertEqual(split_disclosure("  a ;   b  "), ("a", "b"))

    def test_drops_empties(self) -> None:
        self.assertEqual(split_disclosure("a;;b;"), ("a", "b"))
        self.assertEqual(split_disclosure(";"), ())
        self.assertEqual(split_disclosure(""), ())

    def test_dedupes_preserving_order(self) -> None:
        # intent_card() puts soft_preferences = cleaned[2:4] or cleaned[:1], so
        # one card can hold the same cleaned string twice and disclose it twice.
        self.assertEqual(split_disclosure("a; b; a"), ("a", "b"))
        self.assertEqual(split_disclosure("leather; leather"), ("leather",))

    def test_single_segment(self) -> None:
        self.assertEqual(split_disclosure("budget around $29.99"),
                         ("budget around $29.99",))

    def test_duplicate_disclosure_decodes_to_one_segment(self) -> None:
        result = decode("For that, what matters is: leather; leather.")
        self.assertEqual(result.frame, "disclosure")
        self.assertEqual(result.segments, ("leather",))


if __name__ == "__main__":
    unittest.main()
