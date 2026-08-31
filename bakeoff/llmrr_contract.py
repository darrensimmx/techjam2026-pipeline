"""The LLMRR output contract, banding and cache key -- shared by both probes.

WHY THIS FILE EXISTS, and the one rule it must keep
---------------------------------------------------
**Standard library only. No numpy, no provider SDK, no `bakeoff.*` import.**

Two reasons, and the first is load-bearing:

1. `followup_llmrr_esci.py` imports `bakeoff.dense`, which does `import numpy` at
   module scope. CI installs only `requirements.txt`, which is comments-only by
   design. So a test that imports the probe BREAKS THE GREEN BUILD. Tests import
   this module instead, which is why it may not grow a dependency later either.
2. Arm A (ESCI, single-turn) and Arm B (the public set, multi-turn) both band by
   phrase overlap and both guard the model's output. If those live in two places
   they will drift, and the moment they do, "vague" stops meaning the same thing
   in the two arms and the results stop being readable together.

`normalise` is reimplemented here rather than imported from `bakeoff/overlap.py`
because that module imports `evaluator.local_evaluator` and `starter.retrieval`
at module scope -- pulling the whole superseded tree in behind a two-line regex.

THE TWO ENCODINGS
-----------------
`report.md` section 4 argues the output contract, not the model, is the latency
bottleneck: generation time is driven by OUTPUT length, and emitting 10 small
integers instead of 50 ASINs plus a rationale is a ~15x reduction. Section 6
lists the accuracy cost of that indirection as UNMEASURED. Both encodings
therefore ship here as arms, which is what section 6 asks for.

Their guards are not the same shape, and the difference is the point:

    permutation  the model returns every id, reordered. Guard: same multiset.
    indices      the model returns 10 positions. Guard: in range, unique, len 10.

`report.md` argues the second is arguably STRONGER -- an out-of-range integer is
trivially invalid, where a hallucinated ASIN could coincidentally be a real
product from elsewhere in the catalog. That claim is only testable if the
failure KINDS are counted separately, which is why every guard returns a reason
string rather than a bool.
"""
from __future__ import annotations

import hashlib
import re

RETURN_K = 10          # what the agent returns; the metric depth
LISTING_CHARS = 320    # per-candidate truncation

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Casefold and collapse whitespace. Mirrors `bakeoff/overlap.py::normalise`."""
    return _WHITESPACE.sub(" ", text).strip().lower()


# --- Banding ----------------------------------------------------------------

def phrase_overlap(query: str, listings: list[str]) -> float:
    """Longest run of consecutive query tokens appearing verbatim, / query length.

    PHRASE-level, deliberately. The token-level measure -- what fraction of query
    tokens appear anywhere in the listing -- is the one `overlap.py` reports
    second and `part5_realqueries.py` retracted: it is confounded by query
    length, saturates at 1.0 for short queries, and measures brevity as much as
    copying. Using it here collapsed the median split into a single slice.

    The property the gate is actually built on is the 94.5% string-level rate:
    does the customer's wording appear as a literal substring of the listing?
    This is that measure, graded rather than binary so the split has somewhere
    to cut.

    Moved verbatim from `followup_llmrr_esci.py`. Do not "improve" it -- the
    committed baseline artifact's band sizes are a regression test on this
    function, and a change here silently invalidates `report.md` section 1.
    """
    tokens = normalise(query).split()
    if not tokens:
        return 1.0
    haystack = " ".join(listings)
    best = 0
    for start in range(len(tokens)):
        # Only runs longer than the best so far can improve it.
        for end in range(len(tokens), start + best, -1):
            if " ".join(tokens[start:end]) in haystack:
                best = end - start
                break
    return best / len(tokens)


def tercile_cuts(values: list[float]) -> tuple[float, float]:
    """The two boundaries, from the SORTED gate values.

    Ties push LEFT (see `band`), so the bands come out uneven -- 274/151/175 on
    the 600-query ESCI set, not 200/200/200. That is recorded rather than fixed:
    re-cutting the terciles after seeing the data is re-deriving the
    segmentation, and it would invalidate the committed baseline artifact and
    `report.md` section 1 together.
    """
    if not values:
        return (1.0, 1.0)
    ordered = sorted(values)
    return (ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3])


def band(value: float, cuts: tuple[float, float]) -> str:
    """vague / mid / literal. `vague` is the headline -- the band the gate routes."""
    if value <= cuts[0]:
        return "vague"
    if value > cuts[1]:
        return "literal"
    return "mid"


# --- Encodings --------------------------------------------------------------

_SHARED_PREAMBLE = """\
You are ranking products for a shopper's search query.

The shopper's words may not match the product text. Infer what the query
implies: "won't soak through in heavy rain" implies waterproof; "for hiking in
Seattle in November" implies waterproof, warm and breathable. Judge each
candidate on whether it satisfies what the shopper actually needs."""

PERMUTATION_INSTRUCTION = _SHARED_PREAMBLE + """

Return every candidate id exactly once, best first. Returning a set that is not
a permutation of the input is a failure. Give one short reason for your top
choice, naming the implied requirement you inferred and the evidence for it."""

INDICES_INSTRUCTION = _SHARED_PREAMBLE + """

Return the positions of the 10 best candidates, best first, as integers -- or
every position, if fewer than 10 candidates are listed. Each position in range,
no repeats. No explanation, no other text."""

PERMUTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["ranking", "reason"],
    "additionalProperties": False,
}

INDICES_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


def build_prompt_permutation(query: str, candidates: list[tuple[str, str]]) -> str:
    """Candidates labelled by ASIN -- the model returns ids."""
    lines = ["Query: " + query, "", "Candidates:"]
    for cid, body in candidates:
        lines.append("[" + cid + "] " + _WHITESPACE.sub(" ", body)[:LISTING_CHARS])
    return "\n".join(lines)


def build_prompt_indices(query: str, candidates: list[tuple[str, str]]) -> str:
    """Candidates labelled by 0-based position -- the model returns integers.

    The label is the ONLY difference from the permutation prompt. That is
    deliberate: any other change would confound the encoding comparison with a
    prompt change, and the encoding's accuracy cost is the thing `report.md`
    section 6 says is unmeasured.
    """
    lines = ["Query: " + query, "", "Candidates:"]
    for position, (_cid, body) in enumerate(candidates):
        lines.append("[" + str(position) + "] "
                     + _WHITESPACE.sub(" ", body)[:LISTING_CHARS])
    return "\n".join(lines)


# --- Guards -----------------------------------------------------------------
# Each returns a REASON STRING on rejection, or None when the output is
# acceptable. Not a bool: `report.md` claims the index contract is stronger than
# the permutation contract, and that is only testable if the failure kinds are
# counted apart. The probe's own comment at lines 411-415 says the same thing
# about not collapsing contract failures into call failures.

def check_permutation(ranking: object, head: list[str]) -> str | None:
    """The shipped contract: an exact permutation of the input ids."""
    if not isinstance(ranking, list):
        return "wrong_type"
    if any(not isinstance(item, str) for item in ranking):
        return "wrong_type"
    if len(ranking) != len(head):
        return "wrong_length"
    if sorted(ranking) != sorted(head):
        return "permutation_mismatch"
    return None


def check_indices(ranking: object, n: int, k: int = RETURN_K) -> str | None:
    """Exactly `min(k, n)` distinct positions, every one inside [0, n).

    `isinstance(True, int)` is True in Python, so a JSON `true` would sail
    through a naive int check and then happily index a list. Rejected
    explicitly, before anything else touches the value.

    **`k` SCALES DOWN TO `n`, and that is not a convenience.** A fixed "exactly
    10" is unsatisfiable whenever the shortlist is shorter than 10, so it would
    reject a correct answer and silently fall back to the incoming order. On the
    600-query ESCI set 18 queries retrieve fewer than 10 BM25 candidates and
    five retrieve NONE, so a fixed rule books a ~3% contract-failure rate that
    is the harness's fault and reads as the model's. Found by the offline
    `reverse` fake before a single paid call -- which is what the fakes are for.
    """
    if not isinstance(ranking, list):
        return "wrong_type"
    for item in ranking:
        if isinstance(item, bool) or not isinstance(item, int):
            return "wrong_type"
    if len(ranking) != min(k, n):
        return "wrong_length"
    if any(item < 0 or item >= n for item in ranking):
        return "out_of_range"
    if len(set(ranking)) != len(ranking):
        return "duplicate"
    return None


def apply_indices(ranking: list[int], head: list[str]) -> list[str]:
    """The chosen `k` first, then everything else in its incoming order.

    The tail is appended rather than dropped so recall@10 stays comparable with
    the permutation arm -- and because only the top `RETURN_K` can affect the
    metric, the tail's order is irrelevant by construction. Selecting 10 of N
    makes "dropping" inherent to this encoding, which is exactly the shape
    change `report.md` section 6 says the safety contract has to absorb.
    """
    chosen = [head[i] for i in ranking]
    picked = set(ranking)
    return chosen + [a for i, a in enumerate(head) if i not in picked]


ENCODINGS = {
    "permutation": {
        "instruction": PERMUTATION_INSTRUCTION,
        "schema": PERMUTATION_SCHEMA,
        "build_prompt": build_prompt_permutation,
        "has_rationale": True,
    },
    "indices": {
        "instruction": INDICES_INSTRUCTION,
        "schema": INDICES_SCHEMA,
        "build_prompt": build_prompt_indices,
        "has_rationale": False,
    },
}


# --- Cache key --------------------------------------------------------------

def cache_key(model: str, encoding: str, system_text: str, prompt: str) -> str:
    """sha256 over the EXACT call inputs, including the fully rendered prompt.

    Hashing the rendered prompt rather than its ingredients (query, depth, arm)
    is the whole point: a change to LISTING_CHARS, the candidate order, the CE
    cache version, or the instruction text all change the prompt, and none of
    them would change an ingredient-based key. A stale answer replayed against a
    different shortlist is the failure this is built to make impossible.
    """
    digest = hashlib.sha256()
    for part in (model, encoding, system_text, prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def metrics(ranks: list, k: int = RETURN_K) -> dict:
    """recall@k and MRR@k. An empty slice reports None, never 0.0.

    A 0.0 from an empty slice is indistinguishable from a measured total miss,
    and this repo has a standing rule about plausible-looking numbers that are
    actually an absent measurement. Do not "simplify" this back to a bare
    division guarded by `or 1`.
    """
    if not ranks:
        return {"recall@10": None, "mrr@10": None, "n": 0}
    total = len(ranks)
    return {"recall@10": round(sum(1 for r in ranks if r and r <= k) / total, 4),
            "mrr@10": round(sum(1.0 / r for r in ranks if r and r <= k) / total, 4),
            "n": total}


__all__ = [
    "RETURN_K", "LISTING_CHARS", "normalise", "phrase_overlap", "tercile_cuts",
    "band", "ENCODINGS", "check_permutation", "check_indices", "apply_indices",
    "cache_key", "metrics",
]
