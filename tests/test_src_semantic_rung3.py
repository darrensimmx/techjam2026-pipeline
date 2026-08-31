"""Rung 3 (embedding nearest-centroid) decision logic.  [WS-F]

Uses CentroidSemanticDecoder's injectable `embed`/`anchors` constructor
arguments so none of this needs model2vec installed or the vendored potion-8m
weights present -- it runs on the graded, stdlib-only path. The real loader
(`_build_rung3_centroid`, which does need model2vec) is exercised separately
in test_src_layers.py's dependency-gate tests, which patch around it rather
than requiring it to succeed.
"""
from __future__ import annotations

import unittest

from src.semantic import CentroidSemanticDecoder
from src.types import Decode

# Tiny hand-built vectors in a toy 3-D space -- orthogonal-ish so cosine
# similarity behaves predictably and the test does not depend on any real
# embedding model's actual geometry.
_ANCHORS = {
    "refusal": (1.0, 0.0, 0.0),
    "exhaustion": (0.95, 0.05, 0.0),   # deliberately close to refusal
    "disclosure": (0.0, 1.0, 0.0),
    "buying_open": (0.0, 0.0, 1.0),
    "null_nudge": (-1.0, 0.0, 0.0),
}


def _embed_factory(vector_by_text: dict):
    def _embed(text: str):
        return vector_by_text.get(text, (0.0, 0.0, 0.0))
    return _embed


class TestExactMatch(unittest.TestCase):
    def test_exact_anchor_match_recovers_the_frame(self):
        embed = _embed_factory({"quoted text": (0.0, 1.0, 0.0)})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS)
        result = decoder.decode("quoted text")
        self.assertIsInstance(result, Decode)
        self.assertEqual(result.frame, "disclosure")
        self.assertEqual(result.source, "tier2")


class TestThreshold(unittest.TestCase):
    def test_below_threshold_abstains(self):
        # Orthogonal to every anchor -> similarity ~0 everywhere.
        embed = _embed_factory({"nothing like the anchors": (0.0, 0.0, -1.0)})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS, threshold=0.52)
        self.assertIsNone(decoder.decode("nothing like the anchors"))

    def test_at_or_above_threshold_does_not_abstain(self):
        embed = _embed_factory({"close enough": (0.0, 1.0, 0.0)})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS, threshold=0.52)
        self.assertIsNotNone(decoder.decode("close enough"))


class TestRefusalBias(unittest.TestCase):
    """The safety-critical case: misreading refusal as exhaustion silently
    retires a live constraint bucket forever. Misreading exhaustion as refusal
    costs one re-ask on an idle turn. So ties -- and near-ties -- must resolve
    to refusal, never exhaustion."""

    def test_near_tie_between_refusal_and_exhaustion_resolves_to_refusal(self):
        # exhaustion anchor is (0.95, 0.05, 0), refusal is (1, 0, 0) -- a probe
        # vector between them, closer to exhaustion, but well within margin.
        embed = _embed_factory({"ambiguous decline": (0.97, 0.03, 0.0)})
        decoder = CentroidSemanticDecoder(
            embed=embed, anchors=_ANCHORS, threshold=0.52, refusal_margin=0.15)
        result = decoder.decode("ambiguous decline")
        self.assertIsNotNone(result)
        self.assertEqual(result.frame, "refusal")
        self.assertEqual(result.decline, "refusal")

    def test_confident_exhaustion_still_wins_when_margin_is_wide(self):
        # Nearly exact exhaustion match, refusal far behind -- must clear the
        # margin and report exhaustion, or the layer never recovers it at all.
        anchors = {
            "refusal": (1.0, 0.0, 0.0),
            "exhaustion": (0.0, 1.0, 0.0),
        }
        embed = _embed_factory({"clearly exhausted": (0.0, 1.0, 0.0)})
        decoder = CentroidSemanticDecoder(
            embed=embed, anchors=anchors, threshold=0.52, refusal_margin=0.15)
        result = decoder.decode("clearly exhausted")
        self.assertEqual(result.frame, "exhaustion")
        self.assertEqual(result.decline, "exhaustion")

    def test_exhaustion_confusion_against_a_third_frame_still_resolves_safely(self):
        """Regression test for a real bug found during implementation: an
        early version only compared against whichever frame ranked second,
        which missed the case where "override" (not "refusal") is the
        runner-up but refusal's own score is still too close to trust
        exhaustion."""
        anchors = {
            "refusal": (0.70, 0.10, 0.0),
            "exhaustion": (0.75, 0.05, 0.0),   # best match, barely
            "override": (0.72, 0.20, 0.30),    # ranks second, NOT refusal
        }
        embed = _embed_factory({"light paraphrase": (0.75, 0.05, 0.0)})
        decoder = CentroidSemanticDecoder(
            embed=embed, anchors=anchors, threshold=0.0, refusal_margin=0.15)
        result = decoder.decode("light paraphrase")
        self.assertEqual(result.frame, "refusal")

    def test_missing_refusal_score_resolves_to_refusal_not_exhaustion(self):
        """The guard must fail SAFE when refusal cannot be scored at all.

        `scored` only carries frames whose `_cosine` returned a number, so a
        malformed refusal anchor drops refusal out of it entirely. That score
        used to default to 0.0, which cleared any sane margin and committed to
        exhaustion -- retiring a constraint bucket forever at precisely the
        moment the refusal signal had gone missing. Note the probe vector is
        the same one `test_confident_exhaustion_still_wins_when_margin_is_wide`
        uses: with a VALID refusal anchor it must still report exhaustion, so
        the pair is what proves only the missing-anchor case changed.
        """
        anchors = {
            "refusal": (1.0, 0.0),             # wrong length -> _cosine -> None
            "exhaustion": (0.0, 1.0, 0.0),
        }
        embed = _embed_factory({"clearly exhausted": (0.0, 1.0, 0.0)})
        decoder = CentroidSemanticDecoder(
            embed=embed, anchors=anchors, threshold=0.52, refusal_margin=0.15)
        result = decoder.decode("clearly exhausted")
        self.assertIsNotNone(result)
        self.assertEqual(result.frame, "refusal")
        self.assertEqual(result.decline, "refusal")


class TestContentFreeVsContentBearing(unittest.TestCase):
    def test_refusal_carries_no_payload(self):
        embed = _embed_factory({"msg": (1.0, 0.0, 0.0)})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS)
        result = decoder.decode("msg")
        self.assertEqual(result.frame, "refusal")
        self.assertEqual(result.payload, "")
        self.assertEqual(result.segments, ())

    def test_buying_open_carries_the_original_message_as_payload(self):
        embed = _embed_factory({"I need waterproof boots": (0.0, 0.0, 1.0)})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS)
        result = decoder.decode("I need waterproof boots")
        self.assertEqual(result.frame, "buying_open")
        self.assertEqual(result.payload, "I need waterproof boots")
        self.assertEqual(result.segments, ("I need waterproof boots",))


class TestAttributeScope(unittest.TestCase):
    def test_attribute_is_always_none(self):
        """Rung 3 recovers WHICH FRAME, not WHICH ATTRIBUTE -- documents the
        deliberate scope limit. src/pipeline.py::_ask_bookkeeping falls back
        to session.asks.last_ask when decode.attribute is None, so this is
        safe rather than a missing feature."""
        embed = _embed_factory({"msg": (1.0, 0.0, 0.0)})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS)
        result = decoder.decode("msg")
        self.assertIsNone(result.attribute)


class TestNeverRaises(unittest.TestCase):
    def test_a_raising_embedder_returns_none_not_a_traceback(self):
        def _explode(text: str):
            raise RuntimeError("model crashed mid-inference")

        decoder = CentroidSemanticDecoder(embed=_explode, anchors=_ANCHORS)
        self.assertIsNone(decoder.decode("anything"))

    def test_empty_message_abstains(self):
        embed = _embed_factory({})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS)
        self.assertIsNone(decoder.decode(""))

    def test_non_string_message_never_raises(self):
        embed = _embed_factory({})
        decoder = CentroidSemanticDecoder(embed=embed, anchors=_ANCHORS)
        self.assertIsNone(decoder.decode(None))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
