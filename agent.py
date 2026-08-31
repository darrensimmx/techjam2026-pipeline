"""Submission entry point.

The organizer's harness imports `Agent` from this file. Everything real lives in
`src/`; this module exists so the bundle matches the layout in
docs/submission_rules.md ("Recommended File Layout": agent.py + src/).

The sys.path self-heal below is deliberate. `src` is a common package name, and
the harness may run us from a directory where another `src` shadows ours or
where the repo root is not on the path. A raise here happens inside
Agent.__init__, which the evaluator does not wrap -- so it would kill all 200
sessions rather than one turn. Failing over to an explicit repo-root path is
cheaper than debugging that.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.agent import Agent
except ImportError:  # pragma: no cover - environment-dependent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.agent import Agent

__all__ = ["Agent"]
