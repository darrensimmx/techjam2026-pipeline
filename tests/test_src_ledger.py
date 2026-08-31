"""The constraint ledger.  [WS-B]

Two properties are worth more than all the others here:

  * the query IS the concatenation of the raw replies -- not a summary of them,
    not a parse of them;
  * there is NO way to delete anything, and that is asserted against the class
    surface rather than trusted to a comment.
"""
from __future__ import annotations

import re
import unittest

from evaluator.local_evaluator import (
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)
from src.frames import decode
from src.ledger import ConstraintLedger
from tests import synthetic

# Any of these on the class surface would be a way to erase a constraint.
DELETION_API = re.compile(r"clear|remove|pop|reset|delete|replace|discard|truncate",
                          re.IGNORECASE)

# The three content-free templates, verbatim from local_evaluator.py:169, :171
# and :183. They are the only replies that must leave the query untouched.
CONTENT_FREE = (
    "I don't have a preference for material; please use your judgment.",
    "Those options are not quite right yet. Ask me about one specific attribute.",
    "I don't have an additional preference for material.",
)


class TestAccumulation(unittest.TestCase):
    def test_query_is_the_concatenation(self) -> None:
        ledger = ConstraintLedger()
        ledger.append("leather upper")
        ledger.append("For that, what matters is: casual cut; designed for hiking.")
        self.assertEqual(
            ledger.query,
            "leather upper "
            "For that, what matters is: casual cut; designed for hiking.")
        self.assertEqual(len(ledger), 2)

    def test_replies_go_in_verbatim(self) -> None:
        message = "For that, what matters is: budget around $29.99."
        ledger = ConstraintLedger()
        ledger.append(message)
        self.assertEqual(ledger.entries, (message,))
        self.assertIn(message, ledger.query)

    def test_query_grows_monotonically_across_turns(self) -> None:
        ledger = ConstraintLedger()
        seen: list[str] = []
        for message in ("I'm looking for Boots. A key requirement is: leather.",
                        "For that, what matters is: casual cut.",
                        "For that, what matters is: designed for hiking."):
            ledger.append(message)
            seen.append(ledger.query)
        for earlier, later in zip(seen, seen[1:]):
            self.assertTrue(later.startswith(earlier))
            self.assertGreater(len(later), len(earlier))

    def test_query_is_rebuilt_not_cached(self) -> None:
        ledger = ConstraintLedger()
        ledger.append("leather")
        first = ledger.query
        ledger.append("hiking")
        self.assertNotEqual(ledger.query, first)
        self.assertEqual(ledger.query, "leather hiking")

    def test_empty_ledger(self) -> None:
        ledger = ConstraintLedger()
        self.assertEqual(ledger.query, "")
        self.assertEqual(ledger.entries, ())
        self.assertEqual(ledger.segments, ())
        self.assertEqual(len(ledger), 0)
        self.assertEqual(ledger.distinct_segment_count(), 0)

    def test_repetition_is_kept(self) -> None:
        # A repeated term is signal to BM25, not noise to be deduped away.
        ledger = ConstraintLedger()
        ledger.append("leather")
        ledger.append("leather")
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger.query, "leather leather")


class TestNoOps(unittest.TestCase):
    def test_empty_and_whitespace_are_no_ops(self) -> None:
        ledger = ConstraintLedger()
        ledger.append("leather")
        before = ledger.query
        for payload in ("", " ", "\t", "\n\n", "   \r\n  "):
            ledger.append(payload)
        self.assertEqual(ledger.query, before)
        self.assertEqual(len(ledger), 1)

    def test_the_three_content_free_templates_leave_the_query_identical(self) -> None:
        """This is how they drop out: decode() hands back payload="" and
        append() no-ops on it. No filter is written anywhere."""
        ledger = ConstraintLedger()
        ledger.append(decode("I'm looking for Boots. A key requirement is: leather.").payload)
        before = ledger.query
        self.assertTrue(before)

        for message in CONTENT_FREE:
            with self.subTest(message=message):
                result = decode(message)
                self.assertEqual(result.payload, "")
                ledger.append(result.payload)
                ledger.record_segments(result.segments)
                self.assertEqual(ledger.query, before)   # byte-identical
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.distinct_segment_count(), 0)

    def test_non_strings_are_no_ops(self) -> None:
        ledger = ConstraintLedger()
        for payload in (None, 0, 1, 12.5, True, [], {}, (), b"bytes", object()):
            with self.subTest(payload=repr(payload)[:40]):
                ledger.append(payload)
        self.assertEqual(len(ledger), 0)
        self.assertEqual(ledger.query, "")


class TestOverrideIsAppended(unittest.TestCase):
    """The override sentence carries the NEW value, so it is content-bearing.

    And the old value stays: old_value and new_value are both manufactured from
    the same target listing (local_evaluator.py:79-80), so the abandoned
    preference still describes the answer.
    """

    OPENER = "I'm looking for Shoes & Jewelry Boots. casual cut"
    OVERRIDE = "Actually, ignore my earlier preference. What I need is: leather upper."

    def test_override_sentence_is_appended(self) -> None:
        ledger = ConstraintLedger()
        ledger.append(decode(self.OPENER).payload)
        ledger.append(decode(self.OVERRIDE).payload)
        self.assertEqual(len(ledger), 2)
        self.assertIn("leather upper", ledger.query)

    def test_the_abandoned_preference_is_not_erased(self) -> None:
        ledger = ConstraintLedger()
        ledger.append(decode(self.OPENER).payload)
        ledger.append(decode(self.OVERRIDE).payload)
        self.assertIn("casual cut", ledger.query)


class TestNoDeletionApi(unittest.TestCase):
    """"Never erase" is enforced by the ABSENCE of an API, not by a comment."""

    def test_class_exposes_no_deletion_method(self) -> None:
        offenders = [name for name in dir(ConstraintLedger)
                     if DELETION_API.search(name)]
        self.assertEqual(offenders, [])

    def test_instance_exposes_no_deletion_method(self) -> None:
        offenders = [name for name in dir(ConstraintLedger())
                     if DELETION_API.search(name)]
        self.assertEqual(offenders, [])

    def test_public_surface_is_exactly_the_frozen_signature(self) -> None:
        public = {name for name in dir(ConstraintLedger) if not name.startswith("_")}
        self.assertEqual(public, {
            "append", "record_segments", "query", "entries", "segments",
            "distinct_segment_count",
        })

    def test_entries_view_cannot_mutate_the_ledger(self) -> None:
        ledger = ConstraintLedger()
        ledger.append("leather")
        self.assertIsInstance(ledger.entries, tuple)   # not the live list
        self.assertIsInstance(ledger.segments, tuple)
        self.assertEqual(ledger.query, "leather")


class TestSegments(unittest.TestCase):
    def test_records_and_preserves_order(self) -> None:
        ledger = ConstraintLedger()
        ledger.record_segments(("leather", "casual cut"))
        self.assertEqual(ledger.segments, ("leather", "casual cut"))
        self.assertEqual(ledger.distinct_segment_count(), 2)

    def test_distinct_count_dedupes_within_one_reply(self) -> None:
        # soft_preferences = cleaned[2:4] or cleaned[:1] can repeat a hard
        # constraint, so one reply can legitimately carry the same string twice.
        ledger = ConstraintLedger()
        ledger.record_segments(("leather", "leather"))
        self.assertEqual(ledger.distinct_segment_count(), 1)
        self.assertEqual(ledger.segments, ("leather",))

    def test_distinct_count_dedupes_across_replies(self) -> None:
        ledger = ConstraintLedger()
        ledger.record_segments(("leather", "casual cut"))
        ledger.record_segments(("casual cut", "designed for hiking"))
        self.assertEqual(ledger.distinct_segment_count(), 3)
        self.assertEqual(
            ledger.segments, ("leather", "casual cut", "designed for hiking"))

    def test_segments_are_stripped_and_empties_dropped(self) -> None:
        ledger = ConstraintLedger()
        ledger.record_segments(("  leather ", "", "   ", "casual cut"))
        self.assertEqual(ledger.segments, ("leather", "casual cut"))

    def test_segments_never_reach_the_query(self) -> None:
        ledger = ConstraintLedger()
        ledger.record_segments(("leather",))
        self.assertEqual(ledger.query, "")
        self.assertEqual(len(ledger), 0)

    def test_hostile_segment_input_is_a_no_op(self) -> None:
        ledger = ConstraintLedger()
        for segments in (None, 0, 12.5, object(), [None, 1, object()], {}, b"x"):
            with self.subTest(segments=repr(segments)[:40]):
                ledger.record_segments(segments)
        self.assertEqual(ledger.segments, ())

    def test_a_bare_string_is_taken_as_one_segment(self) -> None:
        # Not exploded into characters, which is what iterating a str would do.
        ledger = ConstraintLedger()
        ledger.record_segments("leather upper")
        self.assertEqual(ledger.segments, ("leather upper",))


class TestAgainstTheVendoredGenerator(unittest.TestCase):
    """A whole session, driven by the real harness, through decode + ledger."""

    def test_a_full_session_accumulates_only_disclosed_text(self) -> None:
        products = synthetic.build_products(n=40)
        by_id = {str(item["parent_asin"]): item for item in products}
        categories = {str(item["parent_asin"]): [str(v) for v in item["categories"]]
                      for item in products}

        for sample in synthetic.build_samples(products, per_scenario=2):
            card, behavior = materialize_hidden_fields(sample, by_id)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            category = coarse_category(
                categories.get(str(sample["ground_truth"]["parent_asin"]), []))

            with self.subTest(scenario=sample["scenario_type"]):
                ledger = ConstraintLedger()
                disclosed: set[str] = set()
                boundary_used = False
                message = initial_message(effective, category, disclosed)

                for ask in ("material", "feature", "color", "style", "size",
                            "use_case", "budget", "material", "feature"):
                    result = decode(message)
                    ledger.append(result.payload)
                    ledger.record_segments(result.segments)
                    message, boundary_used = customer_reply(
                        effective, ask, disclosed, boundary_used)

                # Everything the customer actually disclosed is in the query,
                # and the query never went backwards.
                for constraint in disclosed:
                    self.assertIn(constraint, ledger.query, constraint)
                # Nothing content-free leaked in.
                self.assertNotIn("please use your judgment", ledger.query)
                self.assertNotIn("additional preference", ledger.query)
                self.assertNotIn("not quite right yet", ledger.query)
                self.assertLessEqual(
                    ledger.distinct_segment_count(), len(disclosed))


if __name__ == "__main__":
    unittest.main()
