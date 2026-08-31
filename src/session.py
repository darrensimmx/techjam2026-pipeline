"""The per-session aggregate.  [WS-E OWNS]

Everything the agent holds for one shopper lives here, and reset() wipes all of
it. Nothing carries over and nothing is supposed to: session ids are random
UUIDs with no user identity behind them, so "remembering" across sessions would
be mixing up DIFFERENT SHOPPERS. The wipe is structural rather than procedural:
Agent.reset() replaces the whole Session object via new_session(), so there is
no partial-clear path to get wrong and no field that can be forgotten.

Two labels read the wrong way if you skim them: the ledger's "never erased" and
"exhaustion is permanent" both mean WITHIN ONE SESSION.

new_session() is called from Agent.reset(), which the evaluator does NOT wrap
(local_evaluator.py:228) -- a raise there kills all 200 sessions, not one. So it
never raises, for any argument, including an id whose __str__ throws.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from src.askpolicy import AskState
from src.ledger import ConstraintLedger
from src.shown import ShownRegistry
from src.slots import SlotState
from src.types import Scenario


@dataclass
class Session:
    session_id: str
    profile: dict = field(default_factory=dict)
    ledger: ConstraintLedger = field(default_factory=ConstraintLedger)
    slots: SlotState = field(default_factory=SlotState)
    asks: AskState = field(default_factory=AskState)
    shown: ShownRegistry = field(default_factory=ShownRegistry)
    scenario: Scenario = "unknown"
    override_applied: bool = False
    turn: int = 0
    # Diagnostics only. Never read by policy. `null_nudge` arriving at all is a
    # canary: we never send a null ask, so a non-zero count is an upstream bug.
    frame_counts: dict[str, int] = field(default_factory=dict)

    def note_frame(self, frame: object) -> None:
        """Bump the diagnostic counter for one decoded frame. Never raises.

        Diagnostics only -- nothing downstream branches on these counts.
        """
        try:
            name = frame if isinstance(frame, str) and frame else "unknown"
            if isinstance(self.frame_counts, dict):
                self.frame_counts[name] = self.frame_counts.get(name, 0) + 1
        except Exception:
            pass


def _coerce_id(session_id: object) -> str:
    """str() survives an unhashable session id; this also survives a __str__
    that raises, which str() on its own does not."""
    try:
        return str(session_id)
    except Exception:
        return ""


def new_session(session_id: object, user_profile: object) -> Session:
    """Never raises. str() survives an unhashable session id."""
    identifier = _coerce_id(session_id)
    profile = user_profile if isinstance(user_profile, dict) else {}
    try:
        return Session(session_id=identifier, profile=profile)
    except Exception:
        # A default_factory in another workstream's module raised at
        # construction. Build the shell anyway and let each collaborator fail
        # independently -- run_turn degrades per step, so a session missing one
        # collaborator still answers every turn with a valid response.
        return _salvage_session(identifier, profile)


def _salvage_session(identifier: str, profile: dict) -> Session:
    """Last resort: a Session whose collaborators are built one at a time, so a
    single raising constructor does not cost the whole run."""
    session = Session.__new__(Session)
    defaults: dict[str, object] = {
        "session_id": identifier,
        "profile": profile,
        "ledger": None,
        "slots": None,
        "asks": None,
        "shown": None,
        "scenario": "unknown",
        "override_applied": False,
        "turn": 0,
        "frame_counts": {},
    }
    factories = {
        "ledger": ConstraintLedger,
        "slots": SlotState,
        "asks": AskState,
        "shown": ShownRegistry,
    }
    for name, factory in factories.items():
        try:
            defaults[name] = factory()
        except Exception:
            defaults[name] = None
    for spec in fields(Session):
        try:
            object.__setattr__(session, spec.name, defaults.get(spec.name))
        except Exception:
            pass
    return session
