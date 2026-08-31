"""Demo tooling. NOT the submission, and never on the graded path.

Two CLIs that make the agent watchable, driven by the evaluator's own simulated
customer over a session from ``data/public_set.jsonl``:

    python -m demo.backend                     # terminal 2: the reasoning
    python -m demo.frontend --bracket leaky    # terminal 1: the conversation

The frontend runs the real ``Agent`` and appends a JSONL trace; the backend
tails that file and renders each turn's nineteen pipeline stages.

THREE PROPERTIES THIS PACKAGE MUST KEEP, each enforced by a test in
``tests/test_demo_tracer_targets.py`` rather than by good intentions:

  - It never edits ``src/``. Observation is by monkeypatch (see demo/tracer.py),
    so the submission stays byte-identical and the graded path pays nothing.
  - It never sets a seam flag. Enabling ``rerank`` / ``semantic`` / ``llm_rerank``
    / ``askyield`` is a submission-level decision with a disclosure attached
    (CLAUDE.md), not something a demo may do behind your back.
  - It never imports ``starter/`` -- the superseded first-generation system that
    ``cli/agent_server.py`` still drives.

Standard library only, and 3.11-compatible: CI pins 3.11.
"""

SCHEMA_VERSION = 1
