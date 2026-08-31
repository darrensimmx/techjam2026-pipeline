"""BM25 retrieval: ranking quality, the two-phase split, and never raising.

Built on tests/synthetic.py, NOT on tests/fixtures/catalog.jsonl. The fixture has
six products against top_k=10, so every query returns the whole thing and a
query-blind ranker passes. Half of the assertions below are about which product
comes back FIRST out of 251, which the fixture cannot express at all.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval import Bm25Index, terms
from src.types import Candidate
from tests.synthetic import build_catalog, rare_product


class Bm25TestBase(unittest.TestCase):
    """One catalog of 250 generated products plus the planted rare one."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.catalog_path = cls.root / "catalog.jsonl"
        cls.products = build_catalog(cls.catalog_path, n=250, planted=[rare_product()])
        cls.index = Bm25Index(cls.catalog_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.index.close()
        cls._tmp.cleanup()

    def rank_of(self, query: str, parent_asin: str, limit: int = 300):
        """(1-based rank, pool size). rank is None when the target is absent."""
        results = self.index.search(query, limit)
        for candidate in results:
            if candidate.parent_asin == parent_asin:
                return candidate.rank, len(results)
        return None, len(results)


class TestIndexBuild(Bm25TestBase):
    def test_indexes_every_product(self) -> None:
        self.assertEqual(self.index.size, 251)
        self.assertFalse(self.index.is_empty())


class TestRankingQuality(Bm25TestBase):
    """The discriminating tests. A query-blind ranker fails every one."""

    def test_planted_rare_bigram_ranks_first_out_of_251(self) -> None:
        results = self.index.search("chartreuse alpaca", 300)
        self.assertTrue(results, "the rare bigram must retrieve its product")
        self.assertEqual(results[0].parent_asin, "RARE0001")
        self.assertEqual(results[0].rank, 1)

    def test_rare_term_still_wins_inside_a_wide_pool(self) -> None:
        """rank 1 out of a large pool -- ranking, not filtering.

        "winter" alone drags in ~46 products; the rare bigram has to beat all of
        them on score rather than by being the only row that matches.
        """
        rank, pool = self.rank_of("chartreuse alpaca winter cloak", "RARE0001")
        self.assertGreaterEqual(pool, 40, "the pool must be wide enough to be a ranking test")
        self.assertEqual(rank, 1)

    def test_rank_monotonicity_on_the_planted_product(self) -> None:
        broad, _ = self.rank_of("winter", "RARE0001")
        self.assertIsNotNone(broad)
        self.assertGreater(broad, 1, "the broad query must NOT already be perfect")
        narrow, _ = self.rank_of("winter chartreuse", "RARE0001")
        self.assertLess(narrow, broad)
        self.assertEqual(narrow, 1)

    def test_rank_monotonicity_on_an_ordinary_product(self) -> None:
        """Adding discriminating terms strictly improves the target's rank."""
        target = str(self.products[0]["parent_asin"])
        self.assertEqual(self.products[0]["title"], "Black Cotton Boots")
        broad, _ = self.rank_of("boots", target)
        self.assertIsNotNone(broad)
        self.assertGreater(broad, 1)
        narrow, _ = self.rank_of("black cotton boots", target)
        self.assertLess(narrow, broad)

    def test_score_is_ordered_best_first(self) -> None:
        """SQLite's bm25() is more-negative-is-better, and we sort ascending."""
        results = self.index.search("leather boots winter", 300)
        self.assertGreater(len(results), 5)
        scores = [candidate.score for candidate in results]
        self.assertEqual(scores, sorted(scores))


class TestPoolShape(Bm25TestBase):
    def test_pool_is_capped_contiguous_and_untextured(self) -> None:
        results = self.index.search("cotton leather boots jacket winter", 300)
        self.assertLessEqual(len(results), 300)
        self.assertGreater(len(results), 1)
        self.assertEqual([c.rank for c in results], list(range(1, len(results) + 1)))
        self.assertTrue(all(c.text == "" for c in results), "search() must not materialise text")
        self.assertTrue(all(isinstance(c.rowid, int) and c.rowid > 0 for c in results))
        self.assertEqual(len({c.parent_asin for c in results}), len(results))

    def test_limit_is_respected(self) -> None:
        results = self.index.search("cotton leather boots jacket winter", 5)
        self.assertEqual(len(results), 5)
        self.assertEqual([c.rank for c in results], [1, 2, 3, 4, 5])

    def test_hydrate_fills_exactly_those_and_preserves_order(self) -> None:
        pool = self.index.search("cotton leather boots jacket winter", 300)
        window = pool[:50]
        hydrated = self.index.hydrate(window)
        self.assertEqual(len(hydrated), len(window))
        self.assertEqual(
            [c.parent_asin for c in hydrated], [c.parent_asin for c in window],
            "hydrate() must preserve input order",
        )
        self.assertTrue(all(c.text for c in hydrated))
        self.assertTrue(all(c.text == "" for c in pool), "the input candidates are untouched")
        self.assertTrue(all(c.text == "" for c in pool[50:]), "nothing outside the window is hydrated")

    def test_hydrated_text_flattens_a_dict_as_key_value(self) -> None:
        """The overlap gate depends on this: `details` is indexed as "color brown",
        so the evaluator's manufactured "color: brown" only matches after its
        prefix is stripped."""
        hydrated = self.index.hydrate(self.index.search("chartreuse alpaca", 5))
        self.assertEqual(len(hydrated), 1)
        text = hydrated[0].text.lower()
        self.assertIn("chartreuse alpaca cloak", text)
        self.assertIn("color green", text)
        self.assertNotIn("color: green", text)

    def test_hydrate_tolerates_candidates_it_cannot_resolve(self) -> None:
        unknown = [Candidate(parent_asin="NOPE", rowid=0), Candidate(parent_asin="ALSO", rowid=10 ** 9)]
        result = self.index.hydrate(unknown)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(c.text == "" for c in result))

    def test_hydrate_is_idempotent(self) -> None:
        once = self.index.hydrate(self.index.search("alpaca", 10))
        twice = self.index.hydrate(once)
        self.assertEqual([c.text for c in once], [c.text for c in twice])


class TestQuerySafety(Bm25TestBase):
    """A customer reply is arbitrary text. FTS5 MATCH is a query language."""

    HOSTILE = (
        'a" OR "b',
        "NEAR NOT AND OR",
        "\x00\x01",
        "*",
        '"',
        "boots AND NOT jacket",
        "col*umn ^anchor NEAR/3 x",
        "'; DROP TABLE products; --",
        "café naïve",
        "x" * 5000,
        " ".join(f"term{i}" for i in range(200)),
    )

    def test_hostile_queries_return_a_list_and_never_raise(self) -> None:
        for query in self.HOSTILE:
            with self.subTest(query=query[:40]):
                results = self.index.search(query, 10)
                self.assertIsInstance(results, list)
                self.assertLessEqual(len(results), 10)

    def test_empty_and_stopword_only_queries_return_nothing(self) -> None:
        for query in ("", "   ", "the and of to", "I am looking for a"):
            with self.subTest(query=query):
                self.assertEqual(self.index.search(query, 10), [])

    def test_non_string_and_absurd_arguments_never_raise(self) -> None:
        for query in (None, 12345, b"boots", ["boots"], {"q": "boots"}, object()):
            with self.subTest(query=repr(query)[:30]):
                self.assertIsInstance(self.index.search(query, 10), list)
        for limit in (None, -5, 0, "10", 1.5, True, object()):
            with self.subTest(limit=repr(limit)[:30]):
                self.assertIsInstance(self.index.search("boots", limit), list)
        self.assertEqual(self.index.search("boots", 0), [])
        self.assertEqual(self.index.search("boots", -5), [])

    def test_hydrate_never_raises_on_junk(self) -> None:
        for payload in (None, [], [object()], "boots", 17, [Candidate(parent_asin="x")]):
            with self.subTest(payload=repr(payload)[:30]):
                self.assertIsInstance(self.index.hydrate(payload), list)


class TestTerms(unittest.TestCase):
    def test_lowercases_drops_short_tokens_and_stopwords(self) -> None:
        self.assertEqual(
            terms("I'm looking for a Waterproof LEATHER boot, size 10!"),
            ["waterproof", "leather", "boot", "size", "10"],
        )

    def test_never_raises(self) -> None:
        for value in (None, 12345, b"x", [], {}, object()):
            with self.subTest(value=repr(value)[:20]):
                self.assertEqual(terms(value), [])
        self.assertEqual(terms(""), [])
        self.assertEqual(terms("!!! ??? ---"), [])


class TestDegradedConstruction(unittest.TestCase):
    """__init__ is NOT wrapped by the evaluator (local_evaluator.py:306). A raise
    here kills the entire 200-session run, so every one of these must build."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_catalog_builds_empty(self) -> None:
        index = Bm25Index(self.root / "does-not-exist.jsonl")
        self.assertEqual(index.size, 0)
        self.assertIs(index.is_empty(), True)
        self.assertEqual(index.search("boots", 10), [])
        self.assertEqual(index.hydrate([Candidate(parent_asin="x", rowid=1)])[0].text, "")

    def test_empty_file_builds_empty(self) -> None:
        path = self.root / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        index = Bm25Index(path)
        self.assertEqual(index.size, 0)
        self.assertIs(index.is_empty(), True)

    def test_malformed_lines_are_skipped_not_fatal(self) -> None:
        products = build_catalog(self.root / "good.jsonl", n=4)
        path = self.root / "mixed.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(products[0]) + "\n")
            handle.write("{not json at all\n")
            handle.write("\n")
            handle.write(json.dumps(products[1]) + "\n")
            handle.write('{"no_parent_asin": true}\n')
            handle.write("[1, 2, 3]\n")
            handle.write('{"parent_asin": "OK1", "title": null, "features": 7}\n')
        index = Bm25Index(path)
        self.assertEqual(index.size, 3, "two good rows plus the sparse-but-valid one")
        found = {c.parent_asin for c in index.search("boots jacket", 10)}
        self.assertIn(str(products[0]["parent_asin"]), found)

    def test_garbage_paths_build_empty(self) -> None:
        for path in (None, self.root, 12345, b"", object()):
            with self.subTest(path=repr(path)[:30]):
                index = Bm25Index(path)
                self.assertEqual(index.size, 0)
                self.assertIs(index.is_empty(), True)

    def test_a_catalog_of_only_garbage_builds_empty(self) -> None:
        path = self.root / "junk.jsonl"
        path.write_text("not json\n[]\n{}\nnull\n7\n", encoding="utf-8")
        index = Bm25Index(path)
        self.assertEqual(index.size, 0)
        self.assertEqual(index.search("boots", 10), [])


if __name__ == "__main__":
    unittest.main()
