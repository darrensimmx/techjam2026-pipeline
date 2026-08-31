"""Layer 3 -- LLM escalation over the cross-encoder.  [WS-F OWNS]

LIVE as of 1 Sep 2026 (SMOKE-TESTED ONLY): Gemini 3.5 Flash, minimal reasoning.

DISCLOSURE (submission_rules.md, competition_specification.md "Model and API
Policy"): model = gemini-3.5-flash, provider = Google, thinking_budget=0
(minimum-reasoning mode -- report.md section 5: reasoning mode is
disqualifying on latency, ~15s TTFT at high effort). Cost ~$0.011/turn at the
top-50 window per report.md section 4/5 when it fires; it fires only on the
"vague" branch (src/pipeline.py::_llm_escalate), which is ~5.5% of turns
against today's simulator (94.5% verbatim-overlap rate, docs/todo.md). API key
via the GEMINI_API_KEY environment variable -- NEVER hardcoded, NEVER
committed. Offline fallback is `NullLlmReranker`, unconditionally, whenever
the key/package/network isn't available -- see load_llm_reranker() below.
**Only TWO live calls have ever been made against this integration (both
smoke tests, 1 Sep 2026); it has not been measured at scale. Treat any number
beyond those two calls as an estimate, not a result, until a real sweep runs**
(see bakeoff/followup_llmrr_esci.py for the measurement harness this should
eventually be run through).

**Call 1 found a real bug.** The client loaded and connected correctly
(proves the key, the google.genai wiring and Client() construction all work),
but the response's `ranking` length did not equal `min(RETURN_K, n)`, which
an earlier version of `_check_indices` enforced exactly -- rejecting a usable
response as a contract violation. safe_rerank caught it and degraded to
keeping the prior order, so nothing broke; the enhancement just did not fire
that turn. Fixed by relaxing the length check to "at most n, unique, in
range" (length was never load-bearing for correctness -- `_apply_indices`
appends whatever was not explicitly ranked regardless of count).

**Call 2, after the fix, succeeded.** 5 synthetic candidates (a scarf, boots,
a t-shirt, a rain jacket, a desk organizer), query "something that will keep
my feet dry hiking in Seattle in November": returned order put the boots and
rain jacket first -- both genuinely relevant, ahead of the scarf/shirt/desk
organizer. 2.087s, 212 prompt tokens, 9 completion tokens. This is a single
n=1 correctness check, not a benchmark -- it shows the integration works
end-to-end, nothing about aggregate quality or TechnicalScore impact.

When the customer is NOT quoting the listing, keyword matching is blind, so hand
the shortlist to a language model to re-sort. It RE-ORDERS; it can never pull in
a product BM25 never found. That is a structural property of where it sits, not
a promise about the prompt: it is handed a shortlist and returns an ordering of
that shortlist, and `src.rerank.safe_rerank` discards any result that is not a
permutation of its input.

This is the ONLY language model anywhere in the system, and it sits in RANKING,
never in intent. If you are looking at the optional layers and wondering whether
the classifier falls back to an LLM: it does not. Intent is a regex frame decode
(Tier 1) with an ENCODER fallback (`src.semantic`, rung 3 centroid, live) -- see
that module's docstring for why a generative model is excluded from that slot.

Against today's simulator it would barely ever fire -- 94.5% of the simulator's
disclosed constraint strings are verbatim substrings of the target's own
listing, so the case this layer exists for (the customer is not quoting the
listing) is 5.5% of the local set by construction. That is expected, and we say
so rather than hide it.

`submission_rules.md` requires disclosing model choice, approximate cost, token
usage, latency and any fallback behaviour, requires API keys to live in
environment variables and never be committed, and reserves the right to run the
submission with the network disabled at final scoring. That is why
_build_gemini() below reads GEMINI_API_KEY from the environment and why the
network is never probed at construction (see load_llm_reranker's docstring).
"""
from __future__ import annotations

import json
from typing import Callable, Sequence

from src.optional_deps import try_import
from src.types import Candidate, Reranker

# The master switch.
LLM_RERANK_ENABLED: bool = True

# Which model fills the slot -- docs/todo.md item 3.
SELECTED_MODEL: str | None = "gemini-3.5-flash"

# Third-party modules each candidate model needs, checked through try_import
# before anything is constructed. Declared as strings rather than imported at
# module top so that importing src.llm_rerank never touches a third party.
MODEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "gemini-3.5-flash": ("google.genai",),
}

RETURN_K = 10        # what the LLM selects; matches src.types.DEFAULT_TOP_K
LISTING_CHARS = 320  # per-candidate prompt truncation

# Output contract: 10 integer positions, no rationale -- report.md section 4's
# recommendation (~15x latency reduction over ASINs+rationale, same recall
# headroom, since local_evaluator.py:280 never reads message text).
_INSTRUCTION = """\
You are ranking products for a shopper's search query.

The shopper's words may not match the product text. Infer what the query
implies: "won't soak through in heavy rain" implies waterproof; "for hiking in
Seattle in November" implies waterproof, warm and breathable. Judge each
candidate on whether it satisfies what the shopper actually needs.

Return the positions of the 10 best candidates, best first, as integers -- or
every position, if fewer than 10 candidates are listed. Each position in range,
no repeats. No explanation, no other text."""

_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}


def _build_prompt(query: str, candidates: Sequence[Candidate]) -> str:
    lines = ["Query: " + query, "", "Candidates:"]
    for position, item in enumerate(candidates):
        text = (getattr(item, "text", "") or "")[:LISTING_CHARS]
        lines.append("[" + str(position) + "] " + text)
    return "\n".join(lines)


def _check_indices(ranking: object, n: int, k: int = RETURN_K) -> str | None:
    """Looser than bakeoff/llmrr_contract.py::check_indices on purpose: that
    version demands an EXACT length because it feeds a controlled recall@k
    measurement. This is the production path, where `_apply_indices` appends
    whatever was not explicitly ranked in its original order regardless of
    how many the model actually named -- so length is not load-bearing for
    correctness. Found live: gemini-3.5-flash returned a different count than
    min(k, n) on the very first real call, which an exact-length check turned
    into a rejected (but safely degraded) turn instead of a working rerank.
    Uniqueness and range are still enforced -- THOSE are load-bearing, because
    `_apply_indices` indexes directly into `head` with no bounds check of its
    own."""
    if not isinstance(ranking, list) or not ranking:
        return "wrong_type"
    for item in ranking:
        if isinstance(item, bool) or not isinstance(item, int):
            return "wrong_type"
    if len(ranking) > n:
        return "wrong_length"
    if any(item < 0 or item >= n for item in ranking):
        return "out_of_range"
    if len(set(ranking)) != len(ranking):
        return "duplicate"
    return None


def _apply_indices(ranking: list[int], head: list[Candidate]) -> list[Candidate]:
    chosen = [head[i] for i in ranking]
    picked = set(ranking)
    return chosen + [item for i, item in enumerate(head) if i not in picked]


class _GeminiReranker:
    """Wraps a constructed google.genai.Client. `usage()` reports the last
    call's real token counts -- (0, 0) whenever the LLM did not run this
    turn, which `rerank()` never claims on its own: the caller only asks for
    usage after a successful rerank."""

    name = "gemini-3.5-flash"

    def __init__(self, client: object, types_module: object) -> None:
        self._client = client
        self._types = types_module
        self._last_usage = (0, 0)

    def usage(self) -> tuple[int, int]:
        return self._last_usage

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        items = list(candidates)
        if not items:
            return items
        prompt = _build_prompt(query if isinstance(query, str) else "", items)
        response = self._client.models.generate_content(  # type: ignore[attr-defined]
            model=SELECTED_MODEL,
            contents=_INSTRUCTION + "\n\n" + prompt,
            config=self._types.GenerateContentConfig(  # type: ignore[attr-defined]
                response_mime_type="application/json",
                response_schema=_SCHEMA,
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),  # type: ignore[attr-defined]
            ),
        )
        payload = json.loads(response.text)
        ranking = payload.get("ranking")
        reason = _check_indices(ranking, len(items))
        if reason is not None:
            raise ValueError("gemini_contract_violation:" + reason)
        meta = response.usage_metadata
        self._last_usage = (
            int(getattr(meta, "prompt_token_count", 0) or 0),
            int(getattr(meta, "candidates_token_count", 0) or 0),
        )
        return _apply_indices(ranking, items)


def _build_gemini() -> _GeminiReranker:
    """The real loader. Raising here (missing package, missing/invalid
    GEMINI_API_KEY) is fine and expected -- load_llm_reranker() wraps this
    call and falls back to NullLlmReranker, exactly today's behaviour.

    Deliberately does NOT read the key itself: genai.Client() reads
    GEMINI_API_KEY from the environment on its own, which is the point --
    this file never touches the credential value.
    """
    genai = try_import("google.genai")
    types_module = try_import("google.genai.types")
    client = genai.Client()  # type: ignore[union-attr]
    return _GeminiReranker(client, types_module)


# Constructors, registered by whoever wires a model up.
MODEL_BUILDERS: dict[str, Callable[[], object]] = {
    "gemini-3.5-flash": _build_gemini,
}


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
