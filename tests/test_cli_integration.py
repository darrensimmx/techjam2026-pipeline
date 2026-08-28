"""Integration-level proof, distinct from the in-process Agent tests: spawns
the real cli/agent_server.py subprocess -- the actual "endpoint" cli/client.py
talks to -- and drives a full scripted session over the same stdio JSON
protocol.

Scope: this is a smoke test of the *baseline offline spine* only -- the process
boundary is reachable, and the agent survives a full 10-turn drive still
returning schema-shaped, non-empty results. That is all it claims.

It deliberately does NOT claim to prove the competition's termination rule
("the session ends when the target appears in the scored Top 10 or after turn
10"). The agent has no turn-limit logic at all, by design: the caller owns the
turn counter and the stop condition (see cli/client.py's module docstring), and
under grading that caller is the organizer's evaluator, which applies both
halves of the rule itself at local_evaluator.py:238-256. A test that loops
`range(1, 11)` and then asserts it looped ten times is asserting Python's
`range`, not the agent's behaviour, so the loop bound here is a fixture, not
the thing under test. Early termination on a hit is covered where it actually
lives, through the real evaluator, in tests/test_evaluator_smoke.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CATALOG = REPO_ROOT / "tests" / "fixtures" / "catalog.jsonl"
MAX_TURNS = 10


class TestCliIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cli.agent_server", "--catalog", str(CATALOG)],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def tearDown(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                stream.close()

    def _send(self, request: dict) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        self.assertTrue(line, "agent_server produced no response -- endpoint did not connect")
        return json.loads(line)

    def test_endpoint_connects_and_spine_survives_a_full_ten_turn_drive(self) -> None:
        session_id = uuid.uuid4().hex
        reset_response = self._send({
            "op": "reset",
            "session_id": session_id,
            "user_profile": {
                "purchase_frequency": "3-4 prior purchases",
                "average_prior_rating": 4.5,
                "rating_style": "usually positive",
                "preference_tags": ["comfort"],
                "summary": "CI smoke session",
            },
        })
        self.assertTrue(reset_response.get("ok"), "reset did not succeed -- endpoint did not connect")

        # MAX_TURNS is the drive length, not an assertion target: nothing here
        # stops at 10 on its own, so counting the iterations would only re-assert
        # `range`. What each turn is actually checked for is that the endpoint
        # answered at all, that the payload is shaped per the contract, and that
        # the retrieval spine still produced results this deep into a session.
        for turn in range(1, MAX_TURNS + 1):
            message = "Looking for waterproof leather boots." if turn == 1 else "No strong preference."
            response = self._send({
                "op": "respond",
                "session_id": session_id,
                "user_message": message,
                "turn": turn,
                "top_k": 10,
            })
            self.assertIn("message", response)
            self.assertIn("ask_attribute", response)
            self.assertIn("recommendations", response)
            self.assertLessEqual(len(response["recommendations"]), 10, f"turn {turn} exceeded top_k")
            self.assertTrue(
                response["recommendations"],
                f"turn {turn} returned 0 recommendations -- the spine went dead mid-session",
            )

        self.assertIsNone(
            self.proc.poll(),
            "agent_server died during the 10-turn drive",
        )


if __name__ == "__main__":
    unittest.main()
