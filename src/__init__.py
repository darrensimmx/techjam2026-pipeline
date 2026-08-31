"""The Statement 4 conversational-retrieval agent.

A clean-room rebuild against the two source-of-truth design documents
(Statement 4 Architecture v5 and The Seven-Slot Ask Policy, both 30 Aug 2026).

This package has NO dependency on `starter/`, which is the superseded Phase-1
system and is left untouched as the historical record.

Entry point for the organizer's harness is the repo-root `agent.py`, which
re-exports `src.agent.Agent`.
"""

__version__ = "2.0.0-sol"
