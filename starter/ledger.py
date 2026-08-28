"""Per-session state: the accumulated constraint string and what's been asked.

The one lever the planning repo's review found to matter: concatenate every
disclosed constraint into the retrieval query, every turn. No structured slot
parsing — a raw string measured identically for retrieval.

"Unconditionally" holds for every reply that discloses something. The single
documented exception is a content-free reply, which states no constraint and
only adds noise tokens — see `_CONTENT_FREE_PATTERNS` below.
"""
from __future__ import annotations

import re

# Documented exception to "append unconditionally": the simulator has three reply
# shapes that carry no disclosed constraint at all, and appending them injects noise
# tokens ("preference", "options", "attribute") into the BM25 query for the rest of
# the session. Everything else — including the override sentence, which carries the
# new value — is appended verbatim.
#
# Anchored at ^ on purpose: these are whole-message templates, and a message that
# merely *contains* a decline phrase after real content must still be appended.
_CONTENT_FREE_PATTERNS = (
    # "I don't have a preference for color; please use your judgment."  (boundary)
    # "I don't have an additional preference for material."  (asked attribute, no match left)
    re.compile(r"^i do(?:\s+not|n'?t)\s+have\s+(?:a|an\s+additional)\s+preference\b", re.IGNORECASE),
    # "Those options are not quite right yet. Ask me about one specific attribute."
    # Fires on every turn where ask_attribute is None — i.e. turns 7-10 once the
    # fixed six-attribute schedule is exhausted, so it lands repeatedly, not once.
    re.compile(r"^those\s+options\s+are\s+not\s+quite\s+right\s+yet\b", re.IGNORECASE),
)


def _is_content_free(message: str) -> bool:
    return any(pattern.search(message) for pattern in _CONTENT_FREE_PATTERNS)


class SessionState:
    def __init__(self) -> None:
        self.disclosed_constraints: str = ""
        self.asked_attributes: list[str] = []

    def record_message(self, message: str) -> None:
        """Append the customer's message to the ledger verbatim.

        The one exception is a content-free reply (a boundary decline, an
        attribute with nothing left to disclose, or the no-ask prompt) — those
        state no constraint, so appending them only adds noise tokens to the
        query. See `_CONTENT_FREE_PATTERNS`.
        """
        cleaned = re.sub(r"\s+", " ", message or "").strip()
        if not cleaned or _is_content_free(cleaned):
            return
        self.disclosed_constraints = f"{self.disclosed_constraints} {cleaned}".strip()

    def mark_asked(self, attribute: str) -> None:
        if attribute not in self.asked_attributes:
            self.asked_attributes.append(attribute)
