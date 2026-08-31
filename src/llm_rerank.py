"""Layer 3 -- LLM escalation over the cross-encoder.  [WS-F OWNS]

INERT AND PROPOSED ONLY. This is the seam.

When the customer is NOT quoting the listing, keyword matching is blind, so hand
the shortlist to a language model to re-sort. It RE-ORDERS; it can never pull in
a product BM25 never found. That is a structural property of where it sits, not
a promise about the prompt: it is handed a shortlist and returns an ordering of
that shortlist, and `src.rerank.safe_rerank` discards any result that is not a
permutation of its input.

This is the ONLY language model anywhere in the system, and it sits in RANKING,
never in intent. If you are looking at the optional layers and wondering whether
the classifier falls back to an LLM: it does not. Intent is a regex frame decode
(Tier 1) with an ENCODER fallback (`src.semantic`, also inert) -- see that
module's docstring for why a generative model is excluded from that slot.

Against today's simulator it would barely ever fire -- 94.5% of the simulator's
disclosed constraint strings are verbatim substrings of the target's own
listing, so the case this layer exists for (the customer is not quoting the
listing) is 5.5% of the local set by construction. That is expected, and we say
so rather than hide it.

WHICH MODEL IS UNDECIDED. `submission_rules.md` requires disclosing model choice,
approximate cost, token usage, latency and any fallback behaviour, requires API
keys to live in environment variables and never be committed, and reserves the
right to run the submission with the network disabled at final scoring. A layer
behind live credentials must therefore declare itself and must have an offline
fallback -- which, today and until a model is chosen, is `NullLlmReranker`.
See docs/todo.md item 3.

SKELETON -- no model is implemented. Signatures are frozen.
"""
from __future__ import annotations

from typing import Callable, Sequence

from src.optional_deps import try_import
from src.types import Candidate, Reranker

# The master switch. Stays False until a model is chosen AND disclosed.
LLM_RERANK_ENABLED: bool = False

# Which model fills the slot. `None` means "undecided", which is today's state.
SELECTED_MODEL: str | None = None

# Third-party modules each candidate model needs, checked through try_import
# before anything is constructed. Declared as strings rather than imported at
# module top so that importing src.llm_rerank never touches a third party.
MODEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {}

# Constructors, registered by whoever wires a model up. Empty today, on purpose:
# an empty registry is what makes "flipping the flag cannot raise" true rather
# than merely intended.
MODEL_BUILDERS: dict[str, Callable[[], object]] = {}


class NullLlmReranker:
    """The identity pass-through, and it reports zero tokens."""

    name = "null-llm"

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        try:
            return list(candidates)
        except Exception:
            # Not reachable with a well-formed Sequence, and it still returns a
            # list rather than raising: this object's whole job is to be the
            # thing that cannot fail.
            return []

    def usage(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) since the last call.

        Zero today because no model runs. When a real model lands these become
        the counts reported on the wire as `usage.prompt_tokens` /
        `usage.completion_tokens`, which is the disclosure `submission_rules.md`
        asks for -- so the accessor exists now and returns honest zeros, rather
        than being retrofitted alongside the model.
        """
        return (0, 0)


def load_llm_reranker(enabled: bool = LLM_RERANK_ENABLED) -> Reranker:
    """Returns NullLlmReranker unless a model is chosen AND reachable.

    Four independent gates, every one of which falls back to the null reranker:
    the flag, a chosen model, that model's dependencies importing through
    try_import, and the constructed object actually having a callable rerank.
    The whole body is additionally wrapped, because a client library's
    constructor is third-party code we do not control -- and this is called from
    Agent.__init__, which the evaluator does NOT wrap (local_evaluator.py:306).
    A raise there kills every session in the run, not one turn.

    Reachability of a *network* endpoint is deliberately NOT probed here. A
    connection attempt at construction is exactly the behaviour that hangs on a
    rig with the network disabled; a model that needs credentials proves itself
    on its first real call, inside safe_rerank, where a failure costs BM25's
    order and nothing else.

    Never raises. Never returns None.
    """
    try:
        if not enabled:
            return NullLlmReranker()

        model = SELECTED_MODEL
        if not isinstance(model, str) or not model:
            return NullLlmReranker()

        # Undeclared model: refuse rather than construct something undisclosed.
        dependencies = MODEL_DEPENDENCIES.get(model)
        if dependencies is None:
            return NullLlmReranker()
        for module_name in dependencies:
            if try_import(module_name) is None:
                return NullLlmReranker()

        builder = MODEL_BUILDERS.get(model)
        if not callable(builder):
            return NullLlmReranker()

        reranker = builder()
        if reranker is None or not callable(getattr(reranker, "rerank", None)):
            return NullLlmReranker()
        return reranker  # type: ignore[return-value]
    except Exception:
        return NullLlmReranker()
