"""The already-shown registry: the partition, and the override guard.  [WS-E]

These tests are hermetic -- ShownRegistry has no collaborators -- so they stay
green regardless of where the other workstreams are.

The claim under test is narrow and load-bearing: partition() is a PARTITION.
A filter() here reads identically at a glance and is the single change that can
hand the evaluator an empty recommendation list on a drained pool.
"""
from __future__ import annotations

import unittest

from src.shown import EVALUATOR_VISIBLE, ShownRegistry, parent_asin_of
from src.types import Candidate


def candidates(count: int, prefix: str = "P") -> list[Candidate]:
    return [Candidate(parent_asin=f"{prefix}{index:04d}", rowid=index, rank=index + 1,
                      score=-float(index))
            for index in range(count)]


class TestPartition(unittest.TestCase):
    def test_is_a_true_partition(self):
        registry = ShownRegistry()
        pool = candidates(20)
        registry.record([item.parent_asin for item in pool[:7]])

        fresh, seen = registry.partition(pool)

        self.assertEqual(len(fresh) + len(seen), len(pool))
        self.assertEqual(len(fresh + seen), 20)
        self.assertEqual(
            sorted(item.parent_asin for item in fresh + seen),
            sorted(item.parent_asin for item in pool),
        )

    def test_never_returns_a_shortened_list(self):
        """Every product shown: fresh is empty, but nothing is lost."""
        registry = ShownRegistry()
        pool = candidates(6)
        registry.record([item.parent_asin for item in pool])

        fresh, seen = registry.partition(pool)

        self.assertEqual(fresh, [])
        self.assertEqual(len(seen), 6)
        self.assertEqual(len(fresh) + len(seen), 6)

    def test_preserves_incoming_order_within_each_side(self):
        registry = ShownRegistry()
        pool = candidates(10)
        registry.record(["P0001", "P0004", "P0009"])

        fresh, seen = registry.partition(pool)

        self.assertEqual([item.parent_asin for item in seen], ["P0001", "P0004", "P0009"])
        self.assertEqual([item.parent_asin for item in fresh],
                         ["P0000", "P0002", "P0003", "P0005", "P0006", "P0007", "P0008"])

    def test_empty_pool_is_two_empty_lists(self):
        self.assertEqual(ShownRegistry().partition([]), ([], []))

    def test_never_raises_on_hostile_input(self):
        registry = ShownRegistry()
        registry.record(["P0000"])
        for pool in (None, [None, 5, object()], ["P0000", "P0001"], (c for c in candidates(3))):
            with self.subTest(pool=type(pool).__name__):
                fresh, seen = registry.partition(pool)
                self.assertIsInstance(fresh, list)
                self.assertIsInstance(seen, list)


class TestRecord(unittest.TestCase):
    def test_records_what_the_evaluator_saw(self):
        registry = ShownRegistry()
        registry.record(["A", "B", "C"])
        self.assertEqual(len(registry), 3)
        self.assertTrue(registry.is_shown("B"))
        self.assertFalse(registry.is_shown("Z"))

    def test_accepts_candidates_as_well_as_ids(self):
        registry = ShownRegistry()
        registry.record(candidates(3))
        self.assertTrue(registry.is_shown("P0000"))

    def test_stops_at_the_ten_the_evaluator_reads(self):
        """normalize_recommendations() keeps the first ten and drops the rest,
        so an eleventh id was never actually shown to anybody."""
        registry = ShownRegistry()
        registry.record([f"A{index:03d}" for index in range(25)])
        self.assertEqual(len(registry), EVALUATOR_VISIBLE)
        self.assertTrue(registry.is_shown("A009"))
        self.assertFalse(registry.is_shown("A010"))

    def test_blank_ids_do_not_consume_a_slot(self):
        registry = ShownRegistry()
        registry.record(["", "   ", None, "A", "B"])
        self.assertEqual(len(registry), 2)

    def test_is_a_no_op_while_suppressed(self):
        registry = ShownRegistry()
        registry.suppress()
        registry.record(["A", "B", "C"])
        self.assertEqual(len(registry), 0)
        self.assertFalse(registry.is_shown("A"))

    def test_recording_resumes_after_release(self):
        registry = ShownRegistry()
        registry.suppress()
        registry.record(["A"])
        registry.release()
        registry.record(["B"])
        self.assertFalse(registry.suppressed)
        self.assertEqual(len(registry), 1)
        self.assertTrue(registry.is_shown("B"))

    def test_never_raises_on_hostile_input(self):
        registry = ShownRegistry()
        for payload in (None, "ABC", b"ABC", 5, object()):
            with self.subTest(payload=repr(payload)):
                registry.record(payload)
        # A bare string is iterable one character at a time; recording it would
        # poison the set with single letters.
        self.assertEqual(len(registry), 0)


class TestOverrideGuard(unittest.TestCase):
    def test_suppress_sets_the_flag(self):
        registry = ShownRegistry()
        self.assertFalse(registry.suppressed)
        registry.suppress()
        self.assertTrue(registry.suppressed)

    def test_restore_all_empties_the_shown_set(self):
        registry = ShownRegistry()
        registry.record(["A", "B", "C"])
        self.assertEqual(len(registry), 3)

        registry.restore_all()

        self.assertEqual(len(registry), 0)
        self.assertFalse(registry.is_shown("A"))

    def test_restore_all_does_not_change_the_suppression_flag(self):
        """restore and release are separate levers on purpose: one puts products
        back in play, the other decides whether to start recording again."""
        registry = ShownRegistry()
        registry.suppress()
        registry.restore_all()
        self.assertTrue(registry.suppressed)

    def test_restored_products_are_fresh_again(self):
        registry = ShownRegistry()
        pool = candidates(5)
        registry.record([item.parent_asin for item in pool])
        registry.restore_all()

        fresh, seen = registry.partition(pool)

        self.assertEqual(len(fresh), 5)
        self.assertEqual(seen, [])


class TestParentAsinOf(unittest.TestCase):
    def test_reads_candidates_strings_and_junk(self):
        self.assertEqual(parent_asin_of(Candidate(parent_asin="A1")), "A1")
        self.assertEqual(parent_asin_of("  A2 "), "A2")
        self.assertEqual(parent_asin_of(None), "")
        self.assertEqual(parent_asin_of(object()), "")

    def test_survives_a_property_that_raises(self):
        class Exploding:
            @property
            def parent_asin(self):
                raise RuntimeError("boom")

        self.assertEqual(parent_asin_of(Exploding()), "")


if __name__ == "__main__":
    unittest.main()
