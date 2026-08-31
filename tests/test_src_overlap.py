"""The verbatim-overlap gate: order-only, stable, and never a filter.

The two invariants under test are structural, not statistical. gate() must
return exactly as many candidates as it was given (the pool is the recall
floor), and it must be a stable sort (otherwise it silently overwrites whatever
the reranker just did instead of composing with it).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.overlap import gate, measure, normalise, overlap_count, variants
from src.retrieval import Bm25Index
from src.types import Candidate, OverlapReport
from tests.synthetic import build_catalog, rare_product


def candidate(index: int, text: str = "") -> Candidate:
    return Candidate(parent_asin=f"P{index:03d}", rowid=index + 1, rank=index + 1,
                     score=-float(index), text=text)


class TestNormalise(unittest.TestCase):
    def test_casefolds_and_collapses_whitespace(self) -> None:
        self.assertEqual(normalise("  Brown   LEATHER\tBoot\n"), "brown leather boot")

    def test_never_raises(self) -> None:
        for value in (None, 12345, b"x", [], object()):
            with self.subTest(value=repr(value)[:20]):
                self.assertEqual(normalise(value), "")


class TestVariants(unittest.TestCase):
    def test_a_colon_segment_yields_the_tail(self) -> None:
        """intent_card() emits "color: brown"; the listing is indexed as
        "color brown", so only the tail is a verbatim substring."""
        forms = variants("color: brown")
        self.assertIn("color: brown", forms)
        self.assertIn("brown", forms)
        listing = "Brown Nylon Boots ... color brown size 6 department casual"
        self.assertEqual(overlap_count(["color: brown"], listing), 1)

    def test_a_budget_segment_yields_a_dollar_stripped_form(self) -> None:
        forms = variants("budget around $29.99")
        self.assertIn("budget around $29.99", forms)
        self.assertIn("budget around 29.99", forms)
        self.assertTrue(all(form.strip() for form in forms))
        self.assertTrue(any("$" not in form for form in forms))

    def test_a_colon_segment_carrying_a_price_yields_both_derivations(self) -> None:
        forms = variants("price: $29.99")
        self.assertIn("29.99", forms)

    def test_a_plain_segment_is_just_itself(self) -> None:
        self.assertEqual(variants("leather"), ("leather",))

    def test_forms_are_deduped_and_never_degenerate(self) -> None:
        for segment in ("color: brown", "budget around $29.99", "a: b", ": tail", "x:"):
            with self.subTest(segment=segment):
                forms = variants(segment)
                self.assertEqual(len(forms), len(set(forms)))
                self.assertTrue(all(len(form) >= 2 for form in forms))

    def test_never_raises_and_empty_input_yields_nothing(self) -> None:
        for value in ("", "   ", None, 12345, [], object()):
            with self.subTest(value=repr(value)[:20]):
                self.assertEqual(variants(value), ())


class TestOverlapCount(unittest.TestCase):
    TEXT = "Brown Nylon Boots Clothing Boots nylon upper casual cut designed for winter color brown"

    def test_counts_each_segment_once(self) -> None:
        self.assertEqual(overlap_count(["nylon", "winter", "color: brown"], self.TEXT), 3)
        self.assertEqual(overlap_count(["nylon", "nylon"], self.TEXT), 2, "per segment, not per unique string")
        self.assertEqual(overlap_count(["silk", "hiking"], self.TEXT), 0)

    def test_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(overlap_count(["  NYLON   UPPER "], self.TEXT), 1)

    def test_an_unhydrated_candidate_scores_zero(self) -> None:
        self.assertEqual(overlap_count(["nylon"], ""), 0)

    def test_never_raises(self) -> None:
        for segments in (None, "nylon", 17, [None, 5, object()], []):
            with self.subTest(segments=repr(segments)[:25]):
                self.assertIsInstance(overlap_count(segments, self.TEXT), int)
        for text in (None, 17, [], object()):
            with self.subTest(text=repr(text)[:25]):
                self.assertEqual(overlap_count(["nylon"], text), 0)


class TestGateIsOrderOnly(unittest.TestCase):
    """len(gate(x, s)) == len(x) for EVERY input. Nothing downstream of
    retrieval may shorten the candidate list."""

    def test_length_is_preserved_across_every_shape(self) -> None:
        pools = [
            [],
            [candidate(0, "wool scarf")],
            [candidate(i, "wool scarf" if i % 2 else "silk tie") for i in range(20)],
            [candidate(i) for i in range(20)],                      # unhydrated
            [candidate(i, "") for i in range(5)] + [candidate(9, "wool")],
        ]
        segment_sets = [(), [], ("wool",), ("wool", "silk", "nothing here"),
                        ("",), (None, 17), ["color: brown"] * 50]
        for pool in pools:
            for segments in segment_sets:
                with self.subTest(n=len(pool), segments=repr(segments)[:25]):
                    result = gate(pool, segments)
                    self.assertEqual(len(result), len(pool))
                    self.assertCountEqual([c.parent_asin for c in result],
                                          [c.parent_asin for c in pool])

    def test_no_segments_leaves_the_order_exactly_as_it_arrived(self) -> None:
        pool = [candidate(i, "wool scarf" if i % 3 else "silk tie") for i in range(12)]
        self.assertEqual(gate(pool, ()), pool)
        self.assertEqual(gate(pool, []), pool)

    def test_segments_that_match_nothing_leave_the_order_untouched(self) -> None:
        pool = [candidate(i, "wool scarf") for i in range(12)]
        self.assertEqual(gate(pool, ("chartreuse alpaca",)), pool)

    def test_unhydrated_candidates_are_not_reordered_and_do_not_raise(self) -> None:
        pool = [candidate(i) for i in range(10)]
        self.assertEqual(gate(pool, ("wool", "color: brown")), pool)

    def test_never_raises_on_junk(self) -> None:
        self.assertEqual(gate(None, ("wool",)), [])
        self.assertEqual(gate([], ("wool",)), [])
        self.assertEqual(len(gate([object(), object()], ("wool",))), 2)
        self.assertEqual(len(gate([candidate(0, "wool"), object()], ("wool",))), 2)


class TestGateIsStable(unittest.TestCase):
    """Stability is what makes the gate COMPOSE with the reranker. A non-stable
    sort would silently erase the cross-encoder's +0.047 the day it is enabled."""

    def test_equal_overlap_candidates_keep_their_relative_order(self) -> None:
        # 40 candidates, alternating overlap 1 / overlap 0. A non-stable sort
        # (heapsort, or a sort keyed on overlap alone with an arbitrary
        # tiebreak) scrambles within each group; a stable one cannot.
        pool = [candidate(i, "wool scarf" if i % 2 == 0 else "silk tie") for i in range(40)]
        result = gate(pool, ("wool",))

        self.assertEqual(len(result), 40)
        matched = [c.parent_asin for c in result[:20]]
        unmatched = [c.parent_asin for c in result[20:]]
        self.assertEqual(matched, [c.parent_asin for c in pool if c.text == "wool scarf"])
        self.assertEqual(unmatched, [c.parent_asin for c in pool if c.text == "silk tie"])

    def test_it_composes_with_an_upstream_reordering(self) -> None:
        """Whatever order the reranker handed us survives inside each overlap
        tier -- here, a full reversal."""
        pool = [candidate(i, "wool scarf" if i % 2 == 0 else "silk tie") for i in range(20)]
        reranked = list(reversed(pool))
        result = gate(reranked, ("wool",))
        self.assertEqual([c.parent_asin for c in result[:10]],
                         [c.parent_asin for c in reranked if c.text == "wool scarf"])
        self.assertEqual([c.parent_asin for c in result[10:]],
                         [c.parent_asin for c in reranked if c.text == "silk tie"])

    def test_higher_overlap_outranks_lower(self) -> None:
        pool = [
            candidate(0, "silk tie"),                       # 0
            candidate(1, "wool scarf"),                     # 1
            candidate(2, "wool scarf for winter hiking"),    # 3
            candidate(3, "winter coat"),                    # 1
        ]
        result = gate(pool, ("wool", "winter", "hiking"))
        self.assertEqual([c.parent_asin for c in result], ["P002", "P001", "P003", "P000"])


class TestMeasure(unittest.TestCase):
    def test_reports_a_rate_in_the_unit_interval(self) -> None:
        pool = [candidate(0, "wool scarf"), candidate(1, "silk tie for winter")]
        report = measure(pool, ("wool", "winter", "chartreuse"))
        self.assertIsInstance(report, OverlapReport)
        self.assertEqual(report.segments, 3)
        self.assertEqual(report.matched, 2)
        self.assertGreaterEqual(report.rate, 0.0)
        self.assertLessEqual(report.rate, 1.0)
        self.assertAlmostEqual(report.rate, 2 / 3, places=5)

    def test_top_overlap_is_the_head_candidate_alone(self) -> None:
        pool = [candidate(0, "wool scarf"), candidate(1, "silk tie for winter")]
        report = measure(pool, ("wool", "winter"))
        self.assertEqual(report.matched, 2)
        self.assertEqual(report.top_overlap, 1)

    def test_no_segments_reports_zero_rather_than_dividing_by_zero(self) -> None:
        report = measure([candidate(0, "wool")], ())
        self.assertEqual((report.segments, report.matched, report.rate, report.top_overlap),
                         (0, 0, 0.0, 0))

    def test_an_unhydrated_pool_reports_zero(self) -> None:
        report = measure([candidate(i) for i in range(5)], ("wool", "winter"))
        self.assertEqual(report.segments, 2)
        self.assertEqual(report.matched, 0)
        self.assertEqual(report.rate, 0.0)

    def test_never_raises(self) -> None:
        for pool, segments in ((None, ("wool",)), ([], ("wool",)), ([object()], ("wool",)),
                               ([candidate(0, "wool")], None), (17, 17)):
            with self.subTest(pool=repr(pool)[:20]):
                report = measure(pool, segments)
                self.assertIsInstance(report, OverlapReport)
                self.assertGreaterEqual(report.rate, 0.0)
                self.assertLessEqual(report.rate, 1.0)


class TestOverlapAgainstRealIndexedText(unittest.TestCase):
    """The instrument end: the segments the evaluator actually manufactures,
    measured against text this repo's own index produced."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        build_catalog(path, n=250, planted=[rare_product()])
        cls.index = Bm25Index(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.index.close()
        cls._tmp.cleanup()

    def test_manufactured_segments_are_verbatim_in_the_hydrated_listing(self) -> None:
        hydrated = self.index.hydrate(self.index.search("chartreuse alpaca", 5))
        self.assertEqual(len(hydrated), 1)
        # The shapes intent_card() produces for this product: a bare material-ish
        # token, a "color: X" pair, a "key: value" from `details`, and a budget.
        segments = ["chartreuse alpaca weave", "color: green", "department: formal",
                    "hand-loomed", "budget around $249.0"]
        report = measure(hydrated, segments)
        self.assertEqual(report.segments, 5)
        self.assertEqual(report.matched, 4, "everything but the budget line is verbatim")
        self.assertEqual(report.top_overlap, 4)
        self.assertGreater(report.rate, 0.75)

    def test_the_gate_lifts_the_quoting_product_without_dropping_anything(self) -> None:
        pool = self.index.hydrate(self.index.search("winter cloak alpaca", 50))
        self.assertGreater(len(pool), 5)
        gated = gate(pool, ["chartreuse alpaca weave", "color: green"])
        self.assertEqual(len(gated), len(pool))
        self.assertCountEqual([c.parent_asin for c in gated], [c.parent_asin for c in pool])
        self.assertEqual(gated[0].parent_asin, "RARE0001")


if __name__ == "__main__":
    unittest.main()
