"""FROZEN. Every type, protocol and constant shared across modules.

This file is written once, before the parallel build fans out, and is the only
thing every workstream is allowed to import from another workstream's territory.
No workstream may edit it. If a signature here is wrong, it comes back to
assembly rather than being changed locally -- a divergent copy is how blind
parallel agents produce code that does not compose.

Sources for every constant below:
  - the ask enum and maxItems  ->  docs/agent_api_contract.json
  - MAX_TURNS, top_k           ->  evaluator/local_evaluator.py:15-16
  - ANSWERABLE                 ->  evaluator/local_evaluator.py classify_constraint()
  - FIXED_SCHEDULE, HEDGE      ->  "The Seven-Slot Ask Policy" (30 Aug 2026)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

# --------------------------------------------------------------------------
# Wire schema. docs/agent_api_contract.json is authoritative; a unit test
# asserts these three constants still equal what that file says.
# --------------------------------------------------------------------------

ALLOWED_ATTRIBUTES: frozenset[str] = frozenset((
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
))
MAX_RECOMMENDATIONS: int = 100
DEFAULT_TOP_K: int = 10
MAX_TURNS: int = 10

# --------------------------------------------------------------------------
# Ask policy.
# --------------------------------------------------------------------------

# Turns 1-7, in order. Seven because classify_constraint() can only ever return
# these seven labels -- `budget` is the one that was answerable all along and
# was simply never being asked.
FIXED_SCHEDULE: tuple[str, ...] = (
    "material", "feature", "color", "style", "size", "use_case", "budget",
)

# Askable but structurally unanswerable under this simulator: worth exactly zero
# here, kept as a hedge against a private-set simulator that can answer them.
HEDGE_ORDER: tuple[str, ...] = ("brand", "category")

# The labels classify_constraint() can return. Identical to FIXED_SCHEDULE as a
# set; kept separate because one is an ordering and the other is a fact about
# the evaluator.
ANSWERABLE: frozenset[str] = frozenset(FIXED_SCHEDULE)

# `other` bypasses the evaluator's constraint filter entirely and hands back any
# two undisclosed constraints. It is the highest-scoring option on the board
# (+0.004) and is DECLINED, permanently, on judging risk. Never send it.
#
# The trap: any value outside ALLOWED_ATTRIBUTES is silently rewritten to
# `other` by the evaluator, so a typo switches the declined exploit on without
# anyone choosing it. Always send a member of ALLOWED_ATTRIBUTES - FORBIDDEN_ASK.
FORBIDDEN_ASK: frozenset[str] = frozenset(("other",))

DISCLOSURE_CAP: int = 2   # evaluator's matches[:2] -- a 2-item reply may be truncated
CARD_CAPACITY: int = 4    # 2 hard_constraints + 2 soft_preferences

# --------------------------------------------------------------------------
# Retrieval.
# --------------------------------------------------------------------------

POOL_SIZE: int = 300       # candidate pool depth; the recall floor
RERANK_WINDOW: int = 50    # the top-N not-yet-shown handed to the reranker
MAX_QUERY_TERMS: int = 40  # unique stopword-filtered terms OR-joined into MATCH

# --------------------------------------------------------------------------
# Intent.
# --------------------------------------------------------------------------

FrameKind = Literal[
    "buying_open",    # 1  I'm looking for X. A key requirement is: C.
    "override_open",  # 2  I'm looking for X. {old_value}
    "browsing_open",  # 3  I'm looking for X, but I'm still exploring.
    "refusal",        # 4  I don't have A preference for a; please use your judgment.
    "null_nudge",     # 5  Those options are not quite right yet. ...
    "exhaustion",     # 6  I don't have AN ADDITIONAL preference for a.
    "disclosure",     # 7  For that, what matters is: A; B.
    "override",       # 8  Actually, ignore my earlier preference. What I need is: N.
    "unknown",        #    no frame matched -- the only case Tier 2 may look at
]
Scenario = Literal["buying", "browsing", "boundary", "intent_override", "unknown"]

# The decline split, on the single token `additional`. These mean opposite
# things: a refusal never opened the bucket (re-ask later), exhaustion proves it
# empty (never ask again). Both are dropped from the query.
DeclineKind = Literal["refusal", "exhaustion", "none"]

CONTENT_FREE_FRAMES: frozenset[str] = frozenset(("refusal", "null_nudge", "exhaustion"))


@dataclass(frozen=True)
class Decode:
    """What Tier 1 read off one customer reply. A decode, not an estimate.

    `payload` is what the ledger appends -- the ORIGINAL, unmodified message for
    every content-bearing frame, and "" for the three content-free ones.
    `segments` is what the slots and the overlap gate consume.
    """

    frame: FrameKind
    payload: str = ""
    segments: tuple[str, ...] = ()
    decline: DeclineKind = "none"
    scenario_signal: Scenario = "unknown"
    attribute: str | None = None          # populated only by frames 4 and 6
    source: Literal["tier1", "tier2", "none"] = "tier1"


@dataclass(frozen=True)
class Candidate:
    """One product out of retrieval.

    `text` is "" until Bm25Index.hydrate() fills it -- the two-phase split keeps
    us from materialising 300 full listings per turn when only 50 reach rerank.
    """

    parent_asin: str
    rowid: int = 0
    rank: int = 0        # 1-based, out of the stage that produced it
    score: float = 0.0   # raw bm25(); SQLite returns more-negative = better
    text: str = ""


@dataclass(frozen=True)
class TurnPlan:
    """What one turn decided, before it is coerced onto the wire."""

    message: str
    ask_attribute: str | None
    parent_asins: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class OverlapReport:
    """The verbatim-overlap instrument's readout for one turn.

    Observation only. This measures the 94.5%-verbatim claim and is the signal
    an LLM escalation would one day read. It must never remove a candidate.
    """

    segments: int = 0
    matched: int = 0
    rate: float = 0.0
    top_overlap: int = 0


class Reranker(Protocol):
    """Anything that re-orders candidates. Order-only: the return value must be
    a permutation of the input. Never retrieval, never a filter."""

    name: str

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]: ...


class SemanticDecoder(Protocol):
    """Tier 2. Fires ONLY on a Tier 1 `unknown`, and never overrides Tier 1.
    Returns None to abstain -- abstaining is what makes a mediocre fallback safe."""

    name: str

    def decode(self, message: str) -> Decode | None: ...
