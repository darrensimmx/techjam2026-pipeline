"""The seven-slot schedule, the free-turn ladder, and the two registers.
[WS-C OWNS]

Turns 1-7 walk a fixed order. Turns 8-10 are NOT a fixed order -- each free turn
independently runs the same fallthrough against the state at that moment.
Reading them as "turn 8 does this, turn 9 does that" is the single most common
misreading of this design.

The two registers are deliberately different shapes because the two facts they
hold are different:

  RETIREMENT (`retired`) is permanent and per-attribute, so it is a set. Its
  trigger is the customer's WORDS -- a segment count read straight off the
  reply, or an `exhaustion` decline -- and never a yield score. A ranking can be
  revised next turn; a retirement cannot, so it needs proof rather than an
  estimate.

  BURN (`burned`) is one-shot per session, so it is a single Optional[str], not
  a queue. A session carries one scenario_type, and both `boundary_used` and
  `override_applied` are one-shot latches in the evaluator, so no session can
  ever burn twice.

Nothing here raises. next_attribute() in particular is total: for any state,
including a malformed one, it returns a member of
ALLOWED_ATTRIBUTES - FORBIDDEN_ASK. A None ask draws the evaluator's
"Those options are not quite right yet" template (local_evaluator.py:171),
which the ledger drops -- the query is unchanged and the turn teaches nothing.
Asking costs nothing, so a null ask is weakly dominated and never emitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.types import (
    ALLOWED_ATTRIBUTES,
    CARD_CAPACITY,
    DISCLOSURE_CAP,
    FIXED_SCHEDULE,
    FORBIDDEN_ASK,
    HEDGE_ORDER,
)

# Every attribute we are willing to send, in the order we would try them as a
# last resort. `other` is absent by construction, not by a filter: it bypasses
# the evaluator's constraint filter entirely (`attribute == "other" or ...` at
# local_evaluator.py:180) and is DECLINED permanently on judging risk.
ASKABLE: tuple[str, ...] = FIXED_SCHEDULE + HEDGE_ORDER
_DEFAULT_ASK: str = FIXED_SCHEDULE[0]

# Retirement triggers. `refusal` is NOT among them: a boundary refusal means the
# customer declined to look in the bucket, not that the bucket is empty.
_RETIRING_DECLINES: frozenset[str] = frozenset(("exhaustion",))


def _is_valid_ask(attribute: object) -> bool:
    """True iff we may put this on the wire as ask_attribute.

    The trap this closes: any value outside ALLOWED_ATTRIBUTES is silently
    rewritten to `other` by the evaluator (local_evaluator.py:172-173), so a
    typo would switch the declined exploit on without anyone choosing it.
    """
    return (
        isinstance(attribute, str)
        and attribute in ALLOWED_ATTRIBUTES
        and attribute not in FORBIDDEN_ASK
    )


def _str_set(values: object) -> set[str]:
    """Defensive read of a list/set field that policy is about to branch on."""
    if isinstance(values, (set, frozenset, list, tuple)):
        return {item for item in values if isinstance(item, str)}
    return set()


def _int_or(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


@dataclass
class AskState:
    asked: list[str] = field(default_factory=list)
    retired: set[str] = field(default_factory=set)
    yield_counts: dict[str, int] = field(default_factory=dict)
    yield_order: list[str] = field(default_factory=list)
    burned: str | None = None
    burned_reasked: bool = False
    last_ask: str | None = None
    turn: int = 0
    disclosed_count: int = 0

    # -- writes ------------------------------------------------------------

    def mark_asked(self, attribute: str) -> None:
        """Record that we asked it. Idempotent."""
        if not isinstance(attribute, str):
            return
        name = attribute.strip()
        if not name:
            return
        if name not in self.asked:
            self.asked.append(name)
        self.last_ask = name
        # Consuming the burn happens HERE, not in next_attribute(), so that
        # selection stays free of side effects: askyield may call the fixed
        # policy after an adaptive attempt, and a selector that consumed state
        # would spend the re-ask on a turn that never sent it.
        if self.burned is not None and name == self.burned:
            self.burned_reasked = True

    def record_reply(self, attribute: str | None, segment_count: int, decline: str) -> None:
        """The retirement register.

        0 or 1 segments, or an `exhaustion` decline -> the bucket is provably
        drained: RETIRE, never ask again. Exactly 2 -> the evaluator's [:2] cap
        may have truncated it: keep live and re-askable. The trigger is the
        customer's words, not a yield number -- a ranking can be revised next
        turn, a retirement cannot.
        """
        if not isinstance(attribute, str):
            return
        name = attribute.strip()
        if not name:
            return

        kind = decline if isinstance(decline, str) else "none"

        # A refusal burns the ask but proves nothing about the bucket: the
        # customer declined to look, so the attribute stays live and re-askable.
        if kind == "refusal":
            return

        # A burned ask that has not been re-asked yet was never READ by the
        # customer -- on the override turn the evaluator skips customer_reply()
        # entirely (local_evaluator.py:259), so this turn's reply is not an
        # answer to it. Retiring on it would destroy an untouched bucket.
        if self.burned is not None and name == self.burned and not self.burned_reasked:
            return

        if kind in _RETIRING_DECLINES:
            self.retired.add(name)
            self.yield_counts[name] = 0
            return

        if isinstance(segment_count, bool) or not isinstance(segment_count, int):
            return  # unreadable count: do nothing rather than retire on noise

        count = max(0, segment_count)
        self.yield_counts[name] = count
        if name in self.yield_order:
            self.yield_order.remove(name)
        self.yield_order.append(name)
        if count < DISCLOSURE_CAP:
            # The filter ran and came back short. `disclosed` only ever grows,
            # so this bucket can never refill.
            self.retired.add(name)

    def burn(self, attribute: str | None) -> None:
        """One-shot. Two events burn an ask: a boundary refusal, and the override
        turn (where customer_reply is never called, so the ask is never read).
        A session is either boundary or intent_override, never both."""
        if self.burned is not None:
            return  # one-shot latch; a second burn is a no-op
        if not _is_valid_ask(attribute):
            return  # never latch something we could not legally re-ask
        self.burned = attribute
        self.burned_reasked = False

    # -- reads -------------------------------------------------------------

    def pending_reask(self) -> str | None:
        """The burned attribute, if it has not been re-asked yet."""
        if self.burned is None or self.burned_reasked:
            return None
        return self.burned

    def overflow_candidates(self) -> list[str]:
        """Attributes that yielded exactly 2 and are not retired, most recent first."""
        retired = _str_set(self.retired)
        counts = self.yield_counts if isinstance(self.yield_counts, dict) else {}
        order = self.yield_order if isinstance(self.yield_order, list) else []
        out: list[str] = []
        for name in reversed(order):
            if not isinstance(name, str) or name in retired or name in out:
                continue
            if counts.get(name) == DISCLOSURE_CAP:
                out.append(name)
        return out


def _first_valid(candidates: Iterable[object]) -> str | None:
    for candidate in candidates:
        if _is_valid_ask(candidate):
            return candidate  # type: ignore[return-value]
    return None


def _call(state: object, name: str) -> object:
    method = getattr(state, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _select(state: object) -> str | None:
    # Duck-typed rather than isinstance-checked so a Layer 2 state subclass
    # still works, but an object with none of the fields is unusable: fall all
    # the way back rather than letting the ladder read zeros off it.
    if not hasattr(state, "asked") and not hasattr(state, "turn"):
        return None
    asked = _str_set(getattr(state, "asked", ()))
    retired = _str_set(getattr(state, "retired", ()))

    # 1. The fixed schedule owns turns 1-7.
    turn = _int_or(getattr(state, "turn", 0), 0)
    if 1 <= turn <= len(FIXED_SCHEDULE):
        scheduled = FIXED_SCHEDULE[turn - 1]
        if scheduled not in retired and scheduled not in asked:
            return scheduled
        # else: fall through to the ladder, same as any free turn

    # 2. The free-turn fallthrough. Each free turn runs this against the state
    #    AT THAT MOMENT and stops at the first line that matches; it is not a
    #    per-turn assignment.

    # 2.i  A burned ask outranks everything, unconditionally. It was paid for
    #      and never read, so it is the only ask we know nothing about.
    pending = _call(state, "pending_reask")
    if _is_valid_ask(pending):
        return pending  # type: ignore[return-value]

    # 2.ii The card holds at most CARD_CAPACITY constraints. While fewer than
    #      that have been disclosed, an attribute that yielded a full pair of
    #      segments is the only one that can still be holding a string -- the
    #      evaluator's matches[:2] may have truncated it. Most recent first:
    #      its neighbours in the card are the least likely to have been reached.
    disclosed_count = _int_or(getattr(state, "disclosed_count", 0), 0)
    if disclosed_count < CARD_CAPACITY:
        overflow = _call(state, "overflow_candidates")
        if isinstance(overflow, (list, tuple)):
            choice = _first_valid(
                name for name in overflow if isinstance(name, str) and name not in retired
            )
            if choice is not None:
                return choice

    # 2.iii / 2.iv  The hedge. `brand` and `category` are structurally
    #      unanswerable under this simulator and worth exactly zero here. We ask
    #      them anyway: a wasted ask costs one turn and nothing else, the free
    #      turns have no other use, and the organizers hold 800 private sessions
    #      with real intent cards and the right to paraphrase. Priced honestly
    #      at zero -- do not "optimise" this away because it measures zero.
    hedge = _first_valid(name for name in HEDGE_ORDER if name not in asked)
    if hedge is not None:
        return hedge

    # 2.v  Everything is equally worthless by now and the turn is carried
    #      entirely by the return list -- but we still may never send null.
    live = _first_valid(name for name in ASKABLE if name not in retired)
    if live is not None:
        return live
    return _first_valid(ASKABLE)


def next_attribute(state: AskState) -> str:
    """NEVER None, NEVER "other". See src/types.py FORBIDDEN_ASK for why."""
    try:
        choice = _select(state)
    except Exception:
        choice = None
    return choice if _is_valid_ask(choice) else _DEFAULT_ASK
