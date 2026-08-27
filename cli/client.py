"""Dev-only REPL: manually chat with the agent for up to 10 turns.

Spawns cli/agent_server.py as a subprocess and talks to it over stdio.
The client owns the turn counter and the stop condition — the agent itself
knows nothing about turn limits beyond the `turn` number it's handed.

Run via: python3 -m cli.client --catalog data/catalog.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid

MAX_TURNS = 10
DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Manual CLI testing session.",
}


def _send(proc: subprocess.Popen, request: dict) -> dict:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("agent_server exited unexpectedly")
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()

    proc = subprocess.Popen(
        [sys.executable, "-m", "cli.agent_server", "--catalog", args.catalog],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    session_id = uuid.uuid4().hex
    try:
        _send(proc, {"op": "reset", "session_id": session_id, "user_profile": DEFAULT_PROFILE})

        print(f"Session {session_id} -- up to {MAX_TURNS} turns. Type /quit to stop early.\n")
        for turn in range(1, MAX_TURNS + 1):
            user_message = input(f"[turn {turn}] you: ").strip()
            if user_message == "/quit":
                break
            response = _send(proc, {
                "op": "respond",
                "session_id": session_id,
                "user_message": user_message,
                "turn": turn,
                "top_k": 10,
            })
            print(f"[turn {turn}] agent: {response['message']}")
            if response.get("ask_attribute"):
                print(f"           asking about: {response['ask_attribute']}")
            recs = response.get("recommendations", [])
            print(f"           top-{len(recs)}: {[r['parent_asin'] for r in recs]}\n")
        else:
            print("Session ended -- turn 10 reached.")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
