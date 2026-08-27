"""Per-session state: the accumulated constraint string and what's been asked.

The one lever the planning repo's review found to matter: concatenate every
disclosed constraint into the retrieval query, unconditionally, every turn.
No structured slot parsing — a raw string measured identically for retrieval.
"""
from __future__ import annotations

import re

_DECLINE_RE = re.compile(
    r"no preference|don'?t have a preference|do not have a preference|use your judgment",
    re.IGNORECASE,
)


class SessionState:
    def __init__(self) -> None:
        self.disclosed_constraints: str = ""
        self.asked_attributes: list[str] = []

    def record_message(self, message: str) -> None:
        """Append the customer's message to the ledger, unless it's a boundary
        decline (no preference stated) — nothing to accumulate in that case."""
        if not message or _DECLINE_RE.search(message):
            return
        cleaned = re.sub(r"\s+", " ", message).strip()
        if cleaned:
            self.disclosed_constraints = f"{self.disclosed_constraints} {cleaned}".strip()

    def mark_asked(self, attribute: str) -> None:
        if attribute not in self.asked_attributes:
            self.asked_attributes.append(attribute)
