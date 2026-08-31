"""Dev-only: drive the SUBMISSION agent (src/, via agent.py) turn by turn.

cli/agent_server.py is hardwired to `starter.agent` -- the superseded system --
so it cannot answer "does the latest src work". This script is the src/
equivalent, and it prints the internal state the transcript alone does not show:
the ledger (which IS the query), the typed slots, the ask schedule, and the
override/shown bookkeeping.

It replays a FIXED list of customer messages rather than reading stdin, so a
session is reproducible from a file and a human (or an agent) can extend the
list one turn at a time -- the index builds in ~1.3s, so a full replay per turn
is cheaper than holding a REPL open.

    python3 scripts/manual_src_session.py --catalog data/catalog.jsonl \
        --messages runs/buying.json [--top 5] [--target B0XXXXXXXX]

The messages file is either a JSON list of strings or
{"label": str, "profile": dict, "target": str, "messages": [str, ...]}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Agent  # noqa: E402  (after the sys.path self-heal)

DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Manual src/ session.",
}


def load_titles(catalog_path: str) -> dict[str, str]:
    """parent_asin -> a short human label, for reading the top-10 by eye."""
    titles: dict[str, str] = {}
    with open(catalog_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                product = json.loads(line)
            except Exception:
                continue
            asin = str(product.get("parent_asin", ""))
            if not asin:
                continue
            title = str(product.get("title") or "")[:90]
            price = product.get("price")
            titles[asin] = f"{title}  [${price}]" if price not in (None, "") else title
    return titles


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"messages": data}
    return data


def state_lines(agent: Agent, session_id: str) -> list[str]:
    """Read the private session aggregate. Dev tooling only -- nothing in src/
    exposes this, and nothing in src/ should have to."""
    session = getattr(agent, "_sessions", {}).get(session_id)
    if session is None:
        return ["  state: <no session>"]
    ledger = getattr(session, "ledger", None)
    entries = list(getattr(ledger, "entries", ()) or ()) if ledger is not None else []
    query = getattr(ledger, "query", "") if ledger is not None else ""
    slots = getattr(session, "slots", None)
    asks = getattr(session, "asks", None)
    shown = getattr(session, "shown", None)
    return [
        f"  scenario={session.scenario!r} override_applied={session.override_applied} turn={session.turn}",
        f"  ledger({len(entries)}): {entries}",
        f"  query: {query[:220]!r}",
        f"  slots: {slots.as_dict() if slots is not None else None}",
        f"  asked={getattr(asks, 'asked', None)} retired={sorted(getattr(asks, 'retired', set()) or set())}"
        f" last_ask={getattr(asks, 'last_ask', None)!r} disclosed={getattr(asks, 'disclosed_count', None)}",
        f"  shown={len(getattr(shown, '_shown', ()) or ())} suppressed={getattr(shown, 'suppressed', None)}"
        f" frames={session.frame_counts}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--messages", required=True)
    parser.add_argument("--top", type=int, default=5, help="titles to print per turn")
    parser.add_argument("--top-k", type=int, default=10, help="top_k handed to respond()")
    parser.add_argument("--target", default=None, help="ground-truth parent_asin to flag")
    parser.add_argument("--session-id", default="manual")
    parser.add_argument("--no-titles", action="store_true", help="skip the catalog title map")
    args = parser.parse_args()

    spec = load_spec(args.messages)
    messages = [str(item) for item in spec.get("messages", [])]
    target = args.target or spec.get("target")
    profile = spec.get("profile") or DEFAULT_PROFILE
    label = spec.get("label") or Path(args.messages).stem

    agent = Agent(args.catalog)
    titles = {} if args.no_titles else load_titles(args.catalog)

    print(f"### {label}   degraded={agent.degraded}   target={target}")
    agent.reset(args.session_id, profile)

    first_hit_turn = None
    for turn, message in enumerate(messages, start=1):
        response = agent.respond(args.session_id, message, turn, args.top_k)
        recs = response.get("recommendations") or []
        asins = [item.get("parent_asin") if isinstance(item, dict) else item for item in recs]
        hit_rank = asins.index(target) + 1 if target and target in asins else None
        if hit_rank and first_hit_turn is None:
            first_hit_turn = turn

        print(f"\n--- turn {turn} ---")
        print(f"  user: {message}")
        print(f"  ask_attribute: {response.get('ask_attribute')!r}")
        print(f"  message: {response.get('message')!r}")
        print(f"  usage: {response.get('usage')}")
        print(f"  recs: n={len(recs)} hit_rank={hit_rank}")
        for rank, asin in enumerate(asins[: args.top], start=1):
            mark = " <== TARGET" if target and asin == target else ""
            print(f"    {rank}. {asin}  {titles.get(asin, '')}{mark}")
        for line in state_lines(agent, args.session_id):
            print(line)

    print(f"\n### end {label}: turns={len(messages)} first_hit_turn={first_hit_turn}")


if __name__ == "__main__":
    main()
