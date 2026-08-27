"""Dev-only: hosts one Agent instance behind newline-delimited JSON on stdio.

Not the graded path — the organizer's evaluator imports and calls Agent
directly, in-process. This exists purely so a human can drive that same
Agent turn-by-turn via cli/client.py, over stdio rather than a network
socket (nothing to leave open, nothing to contradict "offline core").

Run via: python3 -m cli.agent_server --catalog data/catalog.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()

    agent = Agent(args.catalog)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        op = request.get("op")
        if op == "reset":
            agent.reset(request["session_id"], request["user_profile"])
            response = {"ok": True}
        elif op == "respond":
            response = agent.respond(
                request["session_id"],
                request["user_message"],
                request["turn"],
                request["top_k"],
            )
        else:
            response = {"error": f"unknown op: {op!r}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
