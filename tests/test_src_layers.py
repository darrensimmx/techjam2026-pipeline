"""The Layer 3 seams, and the invariant that keeps the core dependency-free.

Everything under test here is INERT by design, so these tests are not checking
that a feature works -- they are checking that a feature that does not exist
cannot hurt us. Three properties, all of them about failure:

  1. Both seams default to the null implementation.
  2. Turning a seam on with nothing behind it returns the null implementation
     rather than raising. This is the one that matters: `load_*` runs inside
     Agent.__init__, which the evaluator does NOT wrap (local_evaluator.py:306),
     so a raise there zeroes all 200 sessions with no traceback.
  3. `safe_decode` converts every possible Tier 2 misbehaviour into `None`,
     which is exactly what the null decoder returns -- so a Tier 2 failure falls
     back to Tier 1's miss handling UNCHANGED.

Plus the guard on `requirements.txt` being empty, which is a real invariant of
this project rather than a coincidence: the graded path is standard library
only, deliberately, because the organizer reserves the right to run the
submission with the network disabled.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from src import llm_rerank, semantic
from src.llm_rerank import NullLlmReranker, load_llm_reranker
from src.optional_deps import available, try_import
from src.semantic import NullSemanticDecoder, load_semantic_decoder, safe_decode
from src.types import Candidate, Decode

REPO_ROOT = Path(__file__).resolve().parents[1]


class _RaisingDecoder:
    name = "raises"

    def decode(self, message: str) -> Decode | None:
        raise RuntimeError("the exact failure the seam exists to absorb")


class _NonDecodeDecoder:
    """Returns something plausible that is not a `Decode`. The dangerous case:
    it does not throw, so only a type check catches it."""

    name = "wrong-type"

    def decode(self, message: str):
        return {"frame": "disclosure", "payload": message}


class _UnknownFrameDecoder:
    name = "unknown-frame"

    def decode(self, message: str) -> Decode:
        return Decode(frame="unknown", source="tier2")


class _GoodDecoder:
    name = "good"

    def decode(self, message: str) -> Decode:
        return Decode(frame="disclosure", payload=message, source="tier2")


class _NotADecoder:
    """Has a `name` but no `decode`. Duck typing means nothing rejects this at
    construction, so `safe_decode` has to."""

    name = "no-decode-method"


class TestSeamsDefaultInert(unittest.TestCase):
    """Semantic (Tier 2, rung 3) and LLM ranking escalation (Gemini) are both
    now LIVE -- docs/todo.md items 1 and 3. Both still degrade to their null
    implementation whenever their real dependency isn't reachable -- see
    TestEnabledWithoutDependencies below, which is the test that actually
    matters for the graded path."""

    def test_module_flags_are_on(self):
        self.assertTrue(semantic.TIER2_ENABLED)
        self.assertTrue(llm_rerank.LLM_RERANK_ENABLED)

    def test_a_rung_and_a_model_are_selected(self):
        """Rung 3 (docs/todo.md item 1) and Gemini (item 3) are both settled."""
        self.assertEqual(semantic.SELECTED_RUNG, "rung3_centroid")
        self.assertEqual(llm_rerank.SELECTED_MODEL, "gemini-3.5-flash")

    def test_builders_are_registered(self):
        self.assertEqual(set(semantic.RUNG_BUILDERS), {"rung3_centroid"})
        self.assertEqual(set(llm_rerank.MODEL_BUILDERS), {"gemini-3.5-flash"})

    def test_load_semantic_decoder_defaults_to_null_when_disabled(self):
        self.assertIsInstance(load_semantic_decoder(enabled=False), NullSemanticDecoder)

    def test_load_llm_reranker_defaults_to_null(self):
        self.assertIsInstance(load_llm_reranker(), NullLlmReranker)

    def test_null_decoder_always_abstains(self):
        decoder = NullSemanticDecoder()
        self.assertIsNone(decoder.decode(""))
        self.assertIsNone(decoder.decode("a message in a shape nobody predicted"))


class TestEnabledWithoutDependencies(unittest.TestCase):
    """enabled=True with nothing installed and nothing chosen must degrade, not
    raise. Called from an unwrapped __init__, a raise here costs the whole run."""

    def test_semantic_enabled_with_no_rung_selected_returns_null(self):
        """Flag on, but the rung choice cleared -- must still degrade safely."""
        original = semantic.SELECTED_RUNG
        try:
            semantic.SELECTED_RUNG = None
            self.assertIsInstance(load_semantic_decoder(enabled=True), NullSemanticDecoder)
        finally:
            semantic.SELECTED_RUNG = original

    def test_llm_enabled_with_no_model_selected_returns_null(self):
        """Flag on, but the model choice cleared -- must still degrade safely."""
        original = llm_rerank.SELECTED_MODEL
        try:
            llm_rerank.SELECTED_MODEL = None
            self.assertIsInstance(load_llm_reranker(enabled=True), NullLlmReranker)
        finally:
            llm_rerank.SELECTED_MODEL = original

    def test_llm_enabled_without_credentials_returns_null(self):
        """No GEMINI_API_KEY set (or an invalid one) -- must still degrade,
        never raise. This is the actual graded-path condition: the key lives
        in an environment variable that a given machine may simply not have."""
        import os
        original = os.environ.pop("GEMINI_API_KEY", None)
        original_google = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            self.assertIsInstance(load_llm_reranker(enabled=True), NullLlmReranker)
        finally:
            if original is not None:
                os.environ["GEMINI_API_KEY"] = original
            if original_google is not None:
                os.environ["GOOGLE_API_KEY"] = original_google

    def test_semantic_enabled_with_unregistered_rung_returns_null(self):
        """Flag on AND a rung named, but no builder registered for it."""
        original = semantic.SELECTED_RUNG
        try:
            semantic.SELECTED_RUNG = "a_rung_nobody_declared"
            self.assertIsInstance(load_semantic_decoder(enabled=True), NullSemanticDecoder)
        finally:
            semantic.SELECTED_RUNG = original
        self.assertEqual(semantic.SELECTED_RUNG, "rung3_centroid")

    def test_llm_enabled_with_unregistered_model_returns_null(self):
        original = llm_rerank.SELECTED_MODEL
        try:
            llm_rerank.SELECTED_MODEL = "some-model-nobody-declared"
            self.assertIsInstance(load_llm_reranker(enabled=True), NullLlmReranker)
        finally:
            llm_rerank.SELECTED_MODEL = original
        self.assertEqual(llm_rerank.SELECTED_MODEL, "gemini-3.5-flash")

    def test_semantic_survives_a_builder_that_raises(self):
        """A rung whose constructor throws -- a missing checkpoint, a CUDA probe,
        a network call at load time -- must still hand back the null decoder."""
        def _explode():
            raise RuntimeError("no weights on this box")

        original_deps = semantic.RUNG_DEPENDENCIES["rung3_centroid"]
        original_builder = semantic.RUNG_BUILDERS["rung3_centroid"]
        try:
            semantic.RUNG_DEPENDENCIES["rung3_centroid"] = ()  # skip the dep gate
            semantic.RUNG_BUILDERS["rung3_centroid"] = _explode
            self.assertIsInstance(load_semantic_decoder(enabled=True), NullSemanticDecoder)
        finally:
            semantic.RUNG_BUILDERS["rung3_centroid"] = original_builder
            semantic.RUNG_DEPENDENCIES["rung3_centroid"] = original_deps
        self.assertEqual(set(semantic.RUNG_BUILDERS), {"rung3_centroid"})

    def test_semantic_rejects_a_builder_returning_a_non_decoder(self):
        original_deps = semantic.RUNG_DEPENDENCIES["rung3_centroid"]
        original_builder = semantic.RUNG_BUILDERS["rung3_centroid"]
        try:
            semantic.RUNG_DEPENDENCIES["rung3_centroid"] = ()
            semantic.RUNG_BUILDERS["rung3_centroid"] = _NotADecoder
            self.assertIsInstance(load_semantic_decoder(enabled=True), NullSemanticDecoder)
        finally:
            semantic.RUNG_BUILDERS["rung3_centroid"] = original_builder
            semantic.RUNG_DEPENDENCIES["rung3_centroid"] = original_deps


class TestSafeDecode(unittest.TestCase):
    def test_none_decoder(self):
        self.assertIsNone(safe_decode(None, "anything"))

    def test_null_decoder(self):
        self.assertIsNone(safe_decode(NullSemanticDecoder(), "anything"))

    def test_decoder_that_raises(self):
        self.assertIsNone(safe_decode(_RaisingDecoder(), "anything"))

    def test_decoder_returning_a_non_decode(self):
        self.assertIsNone(safe_decode(_NonDecodeDecoder(), "anything"))

    def test_decoder_with_no_decode_method(self):
        self.assertIsNone(safe_decode(_NotADecoder(), "anything"))

    def test_decoder_returning_an_unknown_frame_abstains(self):
        """Tier 2 saying `unknown` is Tier 2 failing too -- an abstention, not a
        decode. Handing it back would look like a successful Tier 2 result."""
        self.assertIsNone(safe_decode(_UnknownFrameDecoder(), "anything"))

    def test_a_real_decode_passes_through(self):
        """The positive control. Without it, a safe_decode that returned None
        unconditionally would pass every other test in this class."""
        result = safe_decode(_GoodDecoder(), "linen, and machine washable")
        self.assertIsInstance(result, Decode)
        self.assertEqual(result.frame, "disclosure")
        self.assertEqual(result.payload, "linen, and machine washable")
        self.assertEqual(result.source, "tier2")

    def test_a_non_string_message_never_raises(self):
        for message in (None, 0, [], {}, object()):
            with self.subTest(message=type(message).__name__):
                self.assertIsNone(safe_decode(NullSemanticDecoder(), message))
                self.assertIsNone(safe_decode(_RaisingDecoder(), message))


class TestNullLlmReranker(unittest.TestCase):
    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(parent_asin="B000000001", rank=1, score=-9.5),
            Candidate(parent_asin="B000000002", rank=2, score=-8.0),
            Candidate(parent_asin="B000000003", rank=3, score=-7.25),
        ]

    def test_rerank_is_the_identity(self):
        reranker = NullLlmReranker()
        candidates = self._candidates()
        result = reranker.rerank("waterproof leather boots", candidates)
        self.assertIsInstance(result, list)
        self.assertEqual(result, candidates)
        self.assertEqual(
            [c.parent_asin for c in result],
            [c.parent_asin for c in candidates],
        )

    def test_rerank_returns_a_new_list(self):
        """Order-only means the caller's list is never mutated underneath it."""
        reranker = NullLlmReranker()
        candidates = self._candidates()
        result = reranker.rerank("q", candidates)
        self.assertIsNot(result, candidates)
        result.reverse()
        self.assertEqual(candidates[0].parent_asin, "B000000001")

    def test_rerank_of_an_empty_shortlist(self):
        self.assertEqual(NullLlmReranker().rerank("q", []), [])

    def test_usage_is_zero(self):
        self.assertEqual(NullLlmReranker().usage(), (0, 0))

    def test_usage_counts_are_non_negative_ints(self):
        """`usage` goes on the wire; the schema requires int >= 0."""
        prompt_tokens, completion_tokens = NullLlmReranker().usage()
        for value in (prompt_tokens, completion_tokens):
            self.assertIsInstance(value, int)
            self.assertNotIsInstance(value, bool)
            self.assertGreaterEqual(value, 0)

    def test_name_is_reported(self):
        self.assertEqual(NullLlmReranker().name, "null-llm")
        self.assertEqual(NullSemanticDecoder().name, "null")


class TestTryImport(unittest.TestCase):
    def test_missing_module_returns_none(self):
        self.assertIsNone(try_import("a_module_that_does_not_exist_9f3c1a"))

    def test_missing_module_does_not_raise(self):
        try:
            try_import("torch_but_misspelled_zzzz")
            try_import("")
            try_import("....")
        except Exception as exc:  # pragma: no cover - the assertion IS the test
            self.fail(f"try_import raised {exc!r}; it must always return None instead")

    def test_present_module_is_returned(self):
        """The control: a try_import that returned None unconditionally would
        pass the two tests above and silently disable every optional layer."""
        module = try_import("json")
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, "loads"))

    def test_available(self):
        self.assertTrue(available("json"))
        self.assertFalse(available("a_module_that_does_not_exist_9f3c1a"))


class TestRequirementsStaysEmpty(unittest.TestCase):
    """`requirements.txt` is comments-only DELIBERATELY -- the graded path is
    standard library only, because the organizer reserves the right to run the
    submission under network restrictions. Optional-layer candidates belong in
    `requirements-optional.txt`, commented out, and never move across."""

    def test_no_uncommented_dependency_line(self):
        path = REPO_ROOT / "requirements.txt"
        self.assertTrue(path.is_file(), f"{path} is missing")
        offenders = [
            (number, line.rstrip())
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            offenders,
            [],
            "requirements.txt must contain no uncommented dependency line; found "
            + "; ".join(f"line {n}: {text!r}" for n, text in offenders),
        )

    def test_requirements_optional_is_entirely_commented_out(self):
        """Nothing is chosen yet, so every candidate stays commented. An
        uncommented line here would install a dependency on a `pip install -r`
        without anyone having made the decision it implies."""
        path = REPO_ROOT / "requirements-optional.txt"
        self.assertTrue(path.is_file(), f"{path} is missing")
        offenders = [
            (number, line.rstrip())
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(offenders, [], f"uncommented entries in {path.name}: {offenders}")


if __name__ == "__main__":
    unittest.main()
