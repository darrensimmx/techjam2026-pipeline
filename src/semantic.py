"""Layer 3 -- Tier 2 semantic fallback.  [WS-F OWNS]

INERT AND NOT APPROVED TO BUILD. This is the seam only.

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

Which implementation fills this one slot is UNDECIDED -- see docs/todo.md item 1
(rung 3 nearest-centroid vs rung 4 fine-tuned head) and item 2 (why rungs 4.5
and 5 are ruled out).

WHY THE GUARDS BELOW ARE REAL EVEN THOUGH THE LAYER IS INERT
------------------------------------------------------------
`load_semantic_decoder` is called from Agent.__init__, and __init__ is NOT
wrapped by the evaluator (local_evaluator.py:306). An ImportError, a missing
checkpoint file, or a package that probes for CUDA at import time would kill all
200 sessions rather than one. `safe_decode` runs on the critical path inside
respond(); a throw there is swallowed into a silent zero for that turn.

So both are written to be total NOW, while nothing is behind them, rather than
when a rung lands and the flag is flipped under time pressure.

SKELETON -- no rung is implemented. Signatures are frozen.
"""
from __future__ import annotations

from typing import Callable

from src.optional_deps import try_import
from src.types import Decode, SemanticDecoder

# The master switch. Stays False until a rung is chosen on held-out numbers.
TIER2_ENABLED: bool = False

# Which rung fills the slot. `None` means "undecided", which is today's state
# and the reason load_semantic_decoder() cannot return anything but the null
# implementation regardless of how `enabled` is passed.
SELECTED_RUNG: str | None = None

# Third-party modules each rung needs, checked through try_import before the
# rung is constructed. Declared here rather than imported at module top so that
# importing src.semantic never touches a third party -- this module is imported
# by the graded path.
RUNG_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    # Rung 3 -- embedding nearest-centroid over the eight known reply shapes.
    "rung3_centroid": ("sentence_transformers",),
    # Rung 4 -- the same frozen encoder with a small trained classifier head.
    "rung4_head": ("sentence_transformers",),
}

# Constructors, registered by whoever builds a rung. Empty today, on purpose:
# an empty registry is what makes "flipping the flag cannot raise" true rather
# than merely intended.
RUNG_BUILDERS: dict[str, Callable[[], object]] = {}


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
