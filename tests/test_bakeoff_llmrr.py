"""The LLMRR output contract: guards, banding, and the response-cache key.

Imports `bakeoff.llmrr_contract` and NOTHING ELSE from `bakeoff/`. That is a
hard constraint, not a style choice: `bakeoff/followup_llmrr_esci.py` reaches
`bakeoff.dense`, which does `import numpy` at module scope, and CI installs only
the comments-only `requirements.txt`. A test that imports the probe would break
the green build on a machine with no third-party packages -- which is every CI
runner this repo has.

These are bakeoff tests, not submission tests. They guard a measurement harness
rather than the shipped agent, and they exist because the harness decides
whether a paid run's numbers mean anything. `test_wrong_length_scales_to_short_head`
in particular encodes a bug that a fixed "exactly 10" rule booked as a ~3%
model-failure rate on the ESCI set when it was really the harness's own.
"""
from __future__ import annotations

import unittest

from bakeoff.llmrr_contract import (
    RETURN_K, apply_indices, band, cache_key, check_indices, check_permutation,
    metrics, normalise, phrase_overlap, tercile_cuts,
)

HEAD = ["A%02d" % i for i in range(50)]


class TestPermutationGuard(unittest.TestCase):
    """The shipped contract: an exact permutation of the input ids."""

    def test_accepts_a_reordering(self):
        self.assertIsNone(check_permutation(list(reversed(HEAD)), HEAD))

    def test_accepts_the_identity(self):
        self.assertIsNone(check_permutation(list(HEAD), HEAD))

    def test_rejects_a_dropped_id(self):
        self.assertEqual(check_permutation(HEAD[:-1], HEAD), "wrong_length")

    def test_rejects_a_duplicated_id(self):
        self.assertEqual(check_permutation(HEAD[:-1] + [HEAD[0]], HEAD),
                         "permutation_mismatch")

    def test_rejects_a_hallucinated_id_of_the_right_length(self):
        """The failure mode the index encoding is argued to be safer against: a
        made-up ASIN can be a real product from elsewhere in the catalog, so
        only the multiset check catches it."""
        self.assertEqual(check_permutation(HEAD[:-1] + ["B99"], HEAD),
                         "permutation_mismatch")

    def test_rejects_a_bare_string(self):
        self.assertEqual(check_permutation("A00", HEAD), "wrong_type")

    def test_rejects_integers_in_the_permutation_arm(self):
        self.assertEqual(check_permutation(list(range(50)), HEAD), "wrong_type")


class TestIndexGuard(unittest.TestCase):
    """Exactly min(k, n) distinct positions, every one inside [0, n)."""

    def test_accepts_ten_distinct_in_range(self):
        self.assertIsNone(check_indices(list(range(10)), len(HEAD)))

    def test_accepts_an_arbitrary_order(self):
        self.assertIsNone(check_indices([9, 0, 4, 1, 8, 2, 7, 3, 6, 5], len(HEAD)))

    def test_rejects_out_of_range(self):
        self.assertEqual(check_indices(list(range(9)) + [99], len(HEAD)),
                         "out_of_range")

    def test_rejects_negative(self):
        self.assertEqual(check_indices(list(range(9)) + [-1], len(HEAD)),
                         "out_of_range")

    def test_rejects_duplicates(self):
        self.assertEqual(check_indices([0] * 10, len(HEAD)), "duplicate")

    def test_rejects_nine(self):
        self.assertEqual(check_indices(list(range(9)), len(HEAD)), "wrong_length")

    def test_rejects_eleven(self):
        self.assertEqual(check_indices(list(range(11)), len(HEAD)), "wrong_length")

    def test_rejects_a_float(self):
        self.assertEqual(check_indices([0.5] + list(range(1, 10)), len(HEAD)),
                         "wrong_type")

    def test_rejects_a_bool(self):
        """`isinstance(True, int)` is True in Python, so a JSON `true` would
        pass a naive int check and then index the list at position 1."""
        self.assertEqual(check_indices([True] + list(range(1, 10)), len(HEAD)),
                         "wrong_type")

    def test_rejects_a_string_digit(self):
        self.assertEqual(check_indices(["0"] + list(range(1, 10)), len(HEAD)),
                         "wrong_type")

    def test_wrong_length_scales_to_short_head(self):
        """A fixed "exactly 10" is unsatisfiable when the shortlist is shorter.

        18 of the 600 ESCI queries retrieve fewer than 10 BM25 candidates. Under
        a fixed rule every one of them is a rejected-but-correct answer, and the
        run reports a ~3% contract-failure rate that belongs to the harness, not
        the model.
        """
        self.assertIsNone(check_indices(list(range(7)), 7))
        self.assertEqual(check_indices(list(range(7)), 7, k=RETURN_K) or "ok", "ok")
        self.assertEqual(check_indices(list(range(6)), 7), "wrong_length")

    def test_empty_head_demands_an_empty_answer(self):
        """Five ESCI queries retrieve nothing at all."""
        self.assertIsNone(check_indices([], 0))
        self.assertEqual(check_indices([0], 0), "wrong_length")


class TestApplyIndices(unittest.TestCase):

    def test_chosen_first_then_the_rest_in_incoming_order(self):
        self.assertEqual(apply_indices([2, 0], ["a", "b", "c"]), ["c", "a", "b"])

    def test_is_a_permutation_of_the_head(self):
        chosen = [9, 0, 4, 1, 8, 2, 7, 3, 6, 5]
        result = apply_indices(chosen, HEAD)
        self.assertEqual(sorted(result), sorted(HEAD))
        self.assertEqual(len(result), len(HEAD))

    def test_identity_selection_reproduces_the_head(self):
        """The property the `echo` fake arm rests on: selecting 0..9 in order
        must leave the ranking untouched, so an echo arm's metrics equal the
        CE baseline's to the last digit."""
        self.assertEqual(apply_indices(list(range(RETURN_K)), HEAD), HEAD)


class TestBanding(unittest.TestCase):

    def test_ties_push_left(self):
        """Documented, not fixed. Re-cutting the terciles after seeing the data
        is re-deriving the segmentation, and it would invalidate the committed
        baseline artifact and report.md section 1 together."""
        cuts = (0.5, 0.75)
        self.assertEqual(band(0.5, cuts), "vague")
        self.assertEqual(band(0.75, cuts), "mid")
        self.assertEqual(band(0.7500001, cuts), "literal")

    def test_cuts_come_from_the_sorted_values(self):
        self.assertEqual(tercile_cuts([0.0, 0.5, 1.0]), (0.5, 1.0))

    def test_empty_values_do_not_raise(self):
        self.assertEqual(tercile_cuts([]), (1.0, 1.0))


class TestPhraseOverlap(unittest.TestCase):

    def test_a_fully_copied_query_scores_one(self):
        self.assertEqual(
            phrase_overlap("waterproof leather boots",
                           ["mens waterproof leather boots in brown"]), 1.0)

    def test_a_wholly_absent_query_scores_zero(self):
        self.assertEqual(phrase_overlap("xyzzy plugh", ["a red shirt"]), 0.0)

    def test_it_measures_the_longest_consecutive_run(self):
        """Phrase-level, not token-level. Both tokens appear, but never
        adjacently, so this must be 1/2 rather than 1.0 -- the token-level
        measure is the one part5_realqueries.py retracted."""
        self.assertEqual(
            phrase_overlap("waterproof boots", ["waterproof leather boots"]), 0.5)

    def test_it_is_case_and_whitespace_insensitive(self):
        self.assertEqual(
            phrase_overlap("Waterproof   Boots", ["waterproof boots"]), 1.0)

    def test_an_empty_query_is_maximally_literal(self):
        self.assertEqual(phrase_overlap("", ["anything"]), 1.0)


class TestCacheKey(unittest.TestCase):
    """The key hashes the RENDERED prompt, so nothing that changes the call can
    reuse an answer computed against a different shortlist."""

    def test_stable_for_identical_inputs(self):
        self.assertEqual(cache_key("m", "e", "sys", "prompt"),
                         cache_key("m", "e", "sys", "prompt"))

    def test_moves_with_the_model(self):
        self.assertNotEqual(cache_key("m1", "e", "s", "p"),
                            cache_key("m2", "e", "s", "p"))

    def test_moves_with_the_encoding(self):
        self.assertNotEqual(cache_key("m", "permutation", "s", "p"),
                            cache_key("m", "indices", "s", "p"))

    def test_moves_with_the_instruction(self):
        self.assertNotEqual(cache_key("m", "e", "sys a", "p"),
                            cache_key("m", "e", "sys b", "p"))

    def test_moves_with_the_prompt(self):
        """Covers candidate order, depth, LISTING_CHARS and the CE cache
        version at once -- all of them change the rendered prompt, and none of
        them would change an ingredient-based key."""
        self.assertNotEqual(cache_key("m", "e", "s", "[A] x\n[B] y"),
                            cache_key("m", "e", "s", "[B] y\n[A] x"))

    def test_field_boundaries_cannot_be_forged(self):
        """Concatenation without a separator would collide these two."""
        self.assertNotEqual(cache_key("ab", "c", "s", "p"),
                            cache_key("a", "bc", "s", "p"))


class TestMetrics(unittest.TestCase):

    def test_an_empty_slice_reports_none_not_zero(self):
        """A 0.0 from an empty slice is indistinguishable from a measured total
        miss, and this repo has a standing rule about plausible-looking numbers
        that are actually an absent measurement."""
        self.assertEqual(metrics([]),
                         {"recall@10": None, "mrr@10": None, "n": 0})

    def test_misses_are_none_and_count_against_the_total(self):
        self.assertEqual(metrics([1, None])["recall@10"], 0.5)

    def test_rank_beyond_k_does_not_count(self):
        self.assertEqual(metrics([11])["recall@10"], 0.0)
        self.assertEqual(metrics([11])["mrr@10"], 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(metrics([2])["mrr@10"], 0.5)


class TestNormalise(unittest.TestCase):

    def test_collapses_and_casefolds(self):
        self.assertEqual(normalise("  A  B\tC\n"), "a b c")


if __name__ == "__main__":
    unittest.main()
