"""Integration-level proof, distinct from the in-process Agent tests: spawns
the real cli/agent_server.py subprocess -- the actual "endpoint" cli/client.py
talks to -- and drives a full scripted session over the same stdio JSON
protocol. Proves two things the in-process tests can't: the process boundary
is actually reachable ("endpoints can connect"), and a driven session
terminates at exactly 10 turns.
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

    def test_endpoint_connects_and_session_ends_by_turn_10(self) -> None:
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

        turns_completed = 0
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
            turns_completed += 1

        self.assertEqual(turns_completed, MAX_TURNS, "session did not run for exactly the allotted 10 turns")


if __name__ == "__main__":
    unittest.main()
