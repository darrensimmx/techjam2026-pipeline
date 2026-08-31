"""Layer 3 -- Tier 2 semantic fallback.  [WS-F OWNS]

LIVE as of 1 Sep 2026: rung 3 (embedding nearest-centroid), potion-8m.

Fires ONLY on a Tier 1 `unknown`, and never overrides Tier 1. It exists for the
PRIVATE SET, where the organizers have reserved the right to reword things
(`competition_specification.md`, "If natural-language paraphrasing is added by
the organizer, it cannot decide correctness"); against today's simulator every
customer utterance is one of eight f-strings, Tier 1 decodes all eight, and this
layer never fires at all.

Encoder only. No SLM, no LLM, nothing that generates text. The reason is not
cost or speed: a generative model emits tokens and has NO CALIBRATED SCORE TO
THRESHOLD, so it cannot abstain -- and abstaining toward refusal is exactly what
makes a mediocre fallback safe here. It would also make the one subsystem we can
prove deterministic stop being deterministic.

RUNG 3 vs RUNG 4, decided (docs/todo.md item 1). Held-out numbers, 168-item
paraphrase set:

    setup                         recovered  wrong  abstained  combined
    potion-8m / stripped @ 0.52   39         0      109        0.3333
    mpnet   / multi     @ 0.31   105        29       14        0.7262

mpnet recovers more but is wrong 29 times; potion-8m is wrong ZERO times. Under
this project's own asymmetry rule -- a wrong refusal/exhaustion read silently
loses a constraint bucket forever, an abstention costs nothing because Tier 1's
miss handling just runs -- zero-wrong is the property to buy even at a much
lower combined-recovery number. potion-8m wins.

PROVENANCE, STATED PLAINLY: there is NO reproducible harness for this run in
this repo. This table is its only trace. (An earlier version of this docstring
cited `potion-8m-evidence.png`, which is not in the tree either.) That is a
weaker footing than the cross-encoder's own comparison next door in
src/rerank.py, which at least has bakeoff/part4_checkpoint_comparison.py behind
it -- treated the same way docs/todo.md treats its +0.047 reconciliation debt,
because the difference between a recorded result and an unsourced number is
worth keeping visible. docs/todo.md:159-164 binds whoever builds that harness:
a self-authored paraphrase holdout is CIRCULAR unless the control is chosen
before the data is generated, not after.

WHY THIS IS JUDGED ON A DIFFERENT METRIC FROM THE CROSS-ENCODER. Both are
optional Layer 3 models and they sit at opposite ends of the same pass, but
picking either by the other's rule picks the wrong winner:

    aspect          this (centroid)          cross-encoder (src/rerank.py)
    task            classify customer intent rank products by relevance
    input           customer reply text      (query, product) pairs
    output          one of 8 intent frames   a relevance score
    failure cost    loses a constraint       falls back to BM25's own order
                    bucket, permanently
    can abstain?    yes -- returns None      no -- it always scores

So the centroid optimises for ZERO-WRONG and the cross-encoder optimises for
ACCURACY, and that asymmetry is the whole reason REFUSAL_BIAS_MARGIN below
exists at all.

WHY THE GUARDS BELOW ARE REAL
------------------------------------------------------------
`load_semantic_decoder` is called from Agent.__init__, and __init__ is NOT
wrapped by the evaluator (local_evaluator.py:306). An ImportError, a missing
checkpoint file, or a package that probes for CUDA at import time would kill all
200 sessions rather than one. `safe_decode` runs on the critical path inside
respond(); a throw there is swallowed into a silent zero for that turn.

So both are written to be total, so that no local-machine gap (an unvendored
model directory, a missing `model2vec` install) can take the graded path down
-- it falls back to `NullSemanticDecoder`, exactly today's behaviour.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from src.frames import normalise, split_disclosure
from src.optional_deps import try_import
from src.types import Decode, SemanticDecoder

# The master switch.
TIER2_ENABLED: bool = True

# Which rung fills the slot. See docs/todo.md item 1 for why rung 3 won.
SELECTED_RUNG: str | None = "rung3_centroid"

# Third-party modules each rung needs, checked through try_import before the
# rung is constructed. Declared here rather than imported at module top so that
# importing src.semantic never touches a third party -- this module is imported
# by the graded path.
RUNG_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    # Rung 3 -- embedding nearest-centroid over the eight known reply shapes.
    # model2vec is numpy-only inference -- no torch, no CUDA probe at import.
    "rung3_centroid": ("model2vec",),
    # Rung 4 -- the same frozen encoder with a small trained classifier head.
    # Not built: needs a training run this sandbox cannot do. See docs/todo.md.
    "rung4_head": ("sentence_transformers",),
}

# Where the vendored potion-8m weights live. Gitignored, distributed as a
# release asset -- same convention as data/catalog.jsonl. See
# docs/windows-dev-setup.md for the one-time fetch step.
POTION_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "potion-base-8m"

# The eight known reply shapes, each an interpolated slot STRIPPED to its fixed
# skeleton -- paraphrase-robustness-prompt.md hard constraint 2: embed the
# frame, never the payload. This is the "stripped" variant validated above.
# Mirrors src/frames.py's own docstring table; not derived from any corpus.
_ANCHOR_TEMPLATES: dict[str, str] = {
    "buying_open": "I'm looking for it. A key requirement is:",
    "override_open": "I'm looking for it.",
    "browsing_open": "I'm looking for it, but I'm still exploring.",
    "refusal": "I don't have a preference for it; please use your judgment.",
    "null_nudge": "Those options are not quite right yet. Ask me about one specific attribute.",
    "exhaustion": "I don't have an additional preference for it.",
    "disclosure": "For that, what matters is:",
    "override": "Actually, ignore my earlier preference. What I need is:",
}

# The scenario_signal each frame carries -- mirrors src/frames.py::decode()
# exactly so a rung-3 recovery behaves identically to a Tier 1 match on every
# field downstream consumers read.
_SCENARIO_SIGNAL: dict[str, str] = {
    "buying_open": "buying",
    "browsing_open": "browsing",
    "override_open": "intent_override",
    "refusal": "boundary",
    "override": "intent_override",
}

# Content-free frames carry payload="" -- mirrors src/types.py::CONTENT_FREE_FRAMES.
_CONTENT_FREE = frozenset(("refusal", "null_nudge", "exhaustion"))

# Below this cosine similarity to every anchor, abstain (return None) rather
# than guess. Measured value from the held-out run above -- potion-8m/stripped.
CENTROID_THRESHOLD: float = 0.52

# NOT a measured value -- paraphrase-robustness-prompt.md's non-negotiable
# ("abstain below threshold", "abstain toward refusal") still applies even
# though the evidence image only gives one global threshold, not a per-pair
# margin. Refusal/exhaustion differ by one token ("additional") and are the
# pair most likely to be confused. Deliberately wide and ASYMMETRIC: rung 3
# only commits to "exhaustion" when it beats refusal's own anchor similarity
# by at least this much; any closer call resolves to refusal, matching how
# conservative frames.py's own Tier 1.5 hedge is (it never guesses exhaustion
# at all on uncertain input -- only an exact regex match can). Ad hoc probing
# with light paraphrases ("I don't really have a preference on X" vs "I don't
# have any more preferences for X") found real confusion pairs under 0.05
# apart; 0.15 was chosen to clear that with margin.
#
# STATUS, corrected 1 Sep 2026: this used to say the value "should be
# re-measured before this ships to a graded run". It has already shipped -- the
# seam went live at cb5817e. So the honest statement is the other way round: an
# UNVALIDATED DESIGNED DEFAULT is carried live and disclosed, and the
# re-measurement against paraphrase-robustness-prompt.md's holdout is a debt
# owed, not a precondition anyone is waiting on. docs/todo.md tracks it beside
# the +0.047 reconciliation debt. Nobody may cite 0.15 as validated.
#
# Observed live, the first time this decoder was ever exercised (1 Sep 2026): a
# light exhaustion paraphrase scored exhaustion 0.649 / refusal 0.590 -- delta
# +0.059, inside the margin -- and was duly resolved to `refusal`. That is the
# intended behaviour, and it is worth being clear-eyed that it IS a trade: the
# frame read was wrong, and the price paid for it was one idle-turn re-ask
# instead of a bucket retired forever.
#
# ONE EDGE CASE, FIXED -- it lives in the guard below rather than in this
# constant. `scores_by_frame.get("refusal", 0.0)` used to default refusal's
# score to 0.0 when refusal was absent from `scored`, which happens if _cosine
# returns None for the refusal anchor specifically (a shape mismatch on that one
# vector). `best_score - 0.0` then cleared any sane margin, so the guard
# silently became a no-op exactly when the refusal signal was the thing that had
# gone missing: it failed OPEN, which is the wrong direction for this particular
# check. The default is now a `None` sentinel and that case takes the same
# branch as a near-tie -- resolve to refusal. Pinned by
# tests/test_src_semantic_rung3.py::test_missing_refusal_score_resolves_to_refusal_not_exhaustion.
REFUSAL_BIAS_MARGIN: float = 0.15


class CentroidSemanticDecoder:
    """Rung 3: nearest-centroid over the eight known reply shapes.

    `embed` and `anchors` are injected so this class is unit-testable without
    model2vec installed -- tests pass a stub embedder and hand-built anchor
    vectors. `_build_rung3_centroid()` below is the only place that wires in
    the real model.
    """

    name = "rung3_centroid"

    def __init__(
        self,
        embed: Callable[[str], object],
        anchors: dict[str, object],
        threshold: float = CENTROID_THRESHOLD,
        refusal_margin: float = REFUSAL_BIAS_MARGIN,
    ) -> None:
        self._embed = embed
        self._anchors = dict(anchors)
        self._threshold = threshold
        self._refusal_margin = refusal_margin

    def decode(self, message: str) -> Decode | None:
        try:
            return self._decode(message)
        except Exception:
            return None

    def _decode(self, message: str) -> Decode | None:
        text = normalise(message)
        if not text:
            return None
        vector = self._embed(text)

        scored: list[tuple[float, str]] = []
        for frame, anchor in self._anchors.items():
            similarity = _cosine(vector, anchor)
            if similarity is not None:
                scored.append((similarity, frame))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=True)

        best_score, best_frame = scored[0]
        if best_score < self._threshold:
            return None

        # Abstain-toward-refusal, asymmetric and unconditional: whenever
        # "exhaustion" would win, it must beat refusal's OWN anchor score --
        # wherever refusal ranks, not only when refusal happens to be runner-up
        # -- by the full margin, or we fall back to refusal. Misreading refusal
        # as exhaustion retires a live bucket forever; the reverse costs one
        # idle-turn re-ask. An early version of this check only compared
        # against second place, which missed exactly this case (a light
        # refusal paraphrase whose runner-up was "override", not "refusal").
        #
        # `refusal_score is None` -- refusal absent from `scored` because
        # _cosine could not score its anchor -- takes the SAME branch as "too
        # close to call". That is deliberate and it is a fix: the default used
        # to be 0.0, which made `best_score - 0.0` clear any sane margin, so the
        # guard silently no-opped and committed to exhaustion at exactly the
        # moment the refusal signal was the thing that had gone missing. It
        # failed open; the one check whose whole job is to prefer the reversible
        # answer must fail safe.
        if best_frame == "exhaustion":
            scores_by_frame = dict((frame, score) for score, frame in scored)
            refusal_score = scores_by_frame.get("refusal")
            if refusal_score is None or (best_score - refusal_score) < self._refusal_margin:
                best_frame = "refusal"

        original = message if isinstance(message, str) else text
        payload = "" if best_frame in _CONTENT_FREE else original
        if best_frame == "disclosure":
            segments = split_disclosure(original)
        elif best_frame in ("buying_open", "override_open", "override"):
            segments = (original.strip(),) if original.strip() else ()
        else:
            segments = ()

        return Decode(
            frame=best_frame,  # type: ignore[arg-type]
            payload=payload,
            segments=segments,
            decline="refusal" if best_frame == "refusal" else
                    "exhaustion" if best_frame == "exhaustion" else "none",
            scenario_signal=_SCENARIO_SIGNAL.get(best_frame, "unknown"),  # type: ignore[arg-type]
            attribute=None,  # rung 3 recovers WHICH FRAME, not which attribute;
                             # _ask_bookkeeping falls back to session.asks.last_ask
            source="tier2",
        )


def _cosine(a: object, b: object) -> float | None:
    """Cosine similarity between two same-length numeric sequences.

    Deliberately not a numpy dependency: this runs in unit tests with plain
    lists too. Returns None on any shape/type mismatch rather than raising --
    the caller treats that as "no match", not a crash.
    """
    try:
        pairs = list(zip(a, b))  # type: ignore[arg-type]
        if not pairs or len(pairs) != len(list(a)) or len(pairs) != len(list(b)):  # type: ignore[arg-type]
            return None
        dot = sum(float(x) * float(y) for x, y in pairs)
        norm_a = math.sqrt(sum(float(x) * float(x) for x in a))  # type: ignore[arg-type]
        norm_b = math.sqrt(sum(float(y) * float(y) for y in b))  # type: ignore[arg-type]
        if norm_a == 0.0 or norm_b == 0.0:
            return None
        return dot / (norm_a * norm_b)
    except Exception:
        return None


def _build_rung3_centroid() -> CentroidSemanticDecoder:
    """The real loader -- model2vec + vendored potion-8m weights.

    Raising here is fine and expected when the weights aren't vendored on this
    machine: load_semantic_decoder() below wraps this call and falls back to
    NullSemanticDecoder, which is exactly today's behaviour.
    """
    model2vec = try_import("model2vec")
    model = model2vec.StaticModel.from_pretrained(str(POTION_MODEL_PATH))  # type: ignore[union-attr]

    def _embed(text: str):
        return model.encode([text])[0]

    anchors = {frame: _embed(template) for frame, template in _ANCHOR_TEMPLATES.items()}
    return CentroidSemanticDecoder(embed=_embed, anchors=anchors)


# Constructors, registered by whoever builds a rung.
RUNG_BUILDERS: dict[str, Callable[[], object]] = {
    "rung3_centroid": _build_rung3_centroid,
}


class NullSemanticDecoder:
    """Abstains, always. The shipping implementation today."""

    name = "null"

    def decode(self, message: str) -> Decode | None:
        return None


def load_semantic_decoder(enabled: bool = TIER2_ENABLED) -> SemanticDecoder:
    """Returns NullSemanticDecoder unless a rung is chosen AND its deps import.

    Four independent gates, every one of which falls back to the null decoder:
    the flag, a chosen rung, that rung's dependencies importing through
    try_import, and the constructed object actually having a callable decode.
    The whole body is additionally wrapped, because a rung's constructor is
    third-party code we do not control and __init__ is not wrapped upstream.

    Never raises. Never returns None.
    """
    try:
        if not enabled:
            return NullSemanticDecoder()

        rung = SELECTED_RUNG
        if not isinstance(rung, str) or not rung:
            return NullSemanticDecoder()

        # Undeclared rung: refuse rather than construct something unreviewed.
        dependencies = RUNG_DEPENDENCIES.get(rung)
        if dependencies is None:
            return NullSemanticDecoder()
        for module_name in dependencies:
            if try_import(module_name) is None:
                return NullSemanticDecoder()

        builder = RUNG_BUILDERS.get(rung)
        if not callable(builder):
            return NullSemanticDecoder()

        decoder = builder()
        if decoder is None or not callable(getattr(decoder, "decode", None)):
            return NullSemanticDecoder()
        return decoder  # type: ignore[return-value]
    except Exception:
        return NullSemanticDecoder()


def safe_decode(decoder: SemanticDecoder | None, message: str) -> Decode | None:
    """Run Tier 2, returning None on ANY problem.

    A Tier 2 failure must fall back to Tier 1's miss handling UNCHANGED -- the
    worst outcome is exactly today's behaviour. Every branch here returns None,
    which is the abstention the caller already knows how to handle because it is
    what the null decoder returns on every call.

    None is returned for:
      - a None decoder (the layer is off, or loading fell through)
      - a decoder with no callable `decode`
      - any exception raised inside the decoder
      - any return value that is not a `Decode`
      - a `Decode` whose frame is `unknown` -- Tier 2 failed to recognise the
        reply too, so there is nothing to hand back that Tier 1 did not already
        have. Treated as an abstention, not as a successful decode.
    """
    if decoder is None:
        return None
    try:
        decode = getattr(decoder, "decode", None)
        if not callable(decode):
            return None
        result = decode(message)
    except Exception:
        return None
    if not isinstance(result, Decode):
        return None
    if result.frame == "unknown":
        return None
    return result
