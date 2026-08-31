"""The rerank seam. Almost every assertion here is about safe_rerank refusing
a result, because that refusal is the only thing standing between an optional
Layer 3 and a silent change to what got retrieved.
"""
from __future__ import annotations

import unittest

from src.rerank import NullReranker, load_reranker, safe_rerank
from src.types import Candidate


def pool(n: int = 5) -> list[Candidate]:
    return [Candidate(parent_asin=f"P{i:03d}", rowid=i + 1, rank=i + 1, score=-float(i))
            for i in range(n)]


class _Stub:
    """A reranker whose behaviour is whatever the test hands it."""

    name = "stub"

    def __init__(self, behaviour) -> None:
        self._behaviour = behaviour
        self.calls = 0

    def rerank(self, query, candidates):
        self.calls += 1
        return self._behaviour(query, candidates)


class TestLoadReranker(unittest.TestCase):
    def test_disabled_returns_the_null_pass_through(self) -> None:
        reranker = load_reranker(enabled=False)
        self.assertIsInstance(reranker, NullReranker)
        self.assertEqual(reranker.name, "null")

    def test_enabled_without_a_vendored_checkpoint_degrades_to_null(self) -> None:
        """Checkpoint chosen (ms-marco-MiniLM-L-6-v2), but if it is not vendored
        on this machine -- or sentence_transformers is not installed -- loading
        must degrade, never raise. Points CE_MODEL_PATH at a path that cannot
        exist so this is true regardless of what is actually vendored locally."""
        import src.rerank as rerank_module
        original_path = rerank_module.CE_MODEL_PATH
        try:
            rerank_module.CE_MODEL_PATH = original_path / "definitely-not-here"
            self.assertIsInstance(load_reranker(enabled=True), NullReranker)
            self.assertIsInstance(load_reranker(enabled=True, timeout_s=0.001), NullReranker)
        finally:
            rerank_module.CE_MODEL_PATH = original_path

    def test_never_raises_on_junk_arguments(self) -> None:
        for args in ((None, None), ("yes", "soon"), (object(), object())):
            with self.subTest(args=repr(args)[:40]):
                self.assertTrue(callable(getattr(load_reranker(*args), "rerank", None)))

    def test_null_reranker_is_the_identity(self) -> None:
        candidates = pool()
        self.assertEqual(NullReranker().rerank("q", candidates), candidates)


class TestSafeRerank(unittest.TestCase):
    def test_identity_passes_through(self) -> None:
        candidates = pool()
        self.assertEqual(safe_rerank(NullReranker(), "boots", candidates), candidates)

    def test_a_legitimate_permutation_is_honoured(self) -> None:
        candidates = pool()
        reversed_pool = list(reversed(candidates))
        stub = _Stub(lambda q, c: list(reversed(c)))
        result = safe_rerank(stub, "boots", candidates)
        self.assertEqual(stub.calls, 1)
        self.assertEqual([c.parent_asin for c in result], [c.parent_asin for c in reversed_pool])

    def test_a_raising_reranker_returns_the_input(self) -> None:
        def explode(query, candidates):
            raise RuntimeError("checkpoint went sideways")

        candidates = pool()
        self.assertEqual(safe_rerank(_Stub(explode), "boots", candidates), candidates)

    def test_a_dropped_candidate_returns_the_input(self) -> None:
        """The pool is the recall floor. Nothing downstream may shorten it."""
        candidates = pool()
        stub = _Stub(lambda q, c: list(c)[:-1])
        self.assertEqual(safe_rerank(stub, "boots", candidates), candidates)
        self.assertEqual(stub.calls, 1, "it ran; its answer was discarded")

    def test_a_duplicated_candidate_returns_the_input(self) -> None:
        candidates = pool()
        stub = _Stub(lambda q, c: list(c) + [c[0]])
        self.assertEqual(safe_rerank(stub, "boots", candidates), candidates)

    def test_a_same_length_duplicate_returns_the_input(self) -> None:
        """The case a length check alone would wave through: five in, five out,
        but one candidate has been overwritten by a copy of another."""
        candidates = pool()
        stub = _Stub(lambda q, c: [c[0], c[0], c[1], c[2], c[3]])
        result = safe_rerank(stub, "boots", candidates)
        self.assertEqual(result, candidates)
        self.assertEqual(len({c.parent_asin for c in result}), 5)

    def test_an_invented_candidate_returns_the_input(self) -> None:
        candidates = pool()
        stub = _Stub(lambda q, c: [Candidate(parent_asin="GHOST")] + list(c)[1:])
        self.assertEqual(safe_rerank(stub, "boots", candidates), candidates)

    def test_a_rewritten_parent_asin_returns_the_input(self) -> None:
        """Same length, same positions, one identifier changed."""
        candidates = pool()

        def rewrite(query, items):
            first = items[0]
            return [Candidate(parent_asin="TYPO", rowid=first.rowid, rank=first.rank,
                              score=first.score, text=first.text), *list(items)[1:]]

        self.assertEqual(safe_rerank(_Stub(rewrite), "boots", candidates), candidates)

    def test_a_nonsense_return_type_returns_the_input(self) -> None:
        candidates = pool()
        for behaviour in (
            lambda q, c: None,
            lambda q, c: "P000 P001",
            lambda q, c: 17,
            lambda q, c: {"P000": 1},
            lambda q, c: (item for item in c),
            lambda q, c: [None, None, None, None, None],
            lambda q, c: [object() for _ in c],
        ):
            with self.subTest(behaviour=behaviour):
                self.assertEqual(safe_rerank(_Stub(behaviour), "boots", candidates), candidates)

    def test_a_broken_reranker_object_returns_the_input(self) -> None:
        candidates = pool()
        for reranker in (None, object(), "reranker", 42):
            with self.subTest(reranker=repr(reranker)[:20]):
                self.assertEqual(safe_rerank(reranker, "boots", candidates), candidates)

    def test_junk_candidates_and_queries_never_raise(self) -> None:
        self.assertEqual(safe_rerank(NullReranker(), "boots", None), [])
        self.assertEqual(safe_rerank(NullReranker(), "boots", []), [])
        self.assertEqual(safe_rerank(NullReranker(), None, pool()), pool())
        self.assertEqual(safe_rerank(NullReranker(), 12345, pool()), pool())

    def test_a_tuple_of_candidates_comes_back_as_a_list(self) -> None:
        candidates = tuple(pool())
        result = safe_rerank(NullReranker(), "boots", candidates)
        self.assertIsInstance(result, list)
        self.assertEqual(result, list(candidates))


if __name__ == "__main__":
    unittest.main()
