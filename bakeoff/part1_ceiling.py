"""Part 1 -- the rerank ceiling.

A reranker reorders a shortlist; it cannot promote a document the shortlist
does not contain. So the entire headroom available to *any* reranker over
BM25's top-N is the gap between "target is somewhere in the top N" and "target
is in the top 10, at the rank BM25 gave it".

The oracle arm is a perfect reranker: whenever the target appears anywhere in
BM25's top N, it goes to rank 1. No real reranker can beat it. If the gap
between R1 and the oracle is small, Part 4 is not worth building.

The oracle changes the *trajectory* too, not just the ranks -- a session that
BM25 hits at turn 5 the oracle hits at turn 2, which moves MTTC and therefore
Efficiency. Replaying through simulate.play() captures that; scoring MRR alone
would not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.simulate import (  # noqa: E402
    MAX_TURNS, bm25_ranker, bootstrap_ci, bootstrap_delta_ci, by_scenario,
    load_trajectories, paired, play, score,
)

DEPTHS = (10, 50, 100)


def oracle_ranker(depth: int):
    """Perfect rerank of BM25's top `depth`: target to rank 1 if present."""
    def rank(record: dict, turn: int) -> list[str]:
        shortlist = record["bm25"][turn - 1][:depth]
        target = record["target"]
        if target in shortlist:
            return [target] + [a for a in shortlist if a != target]
        return shortlist
    return rank


def recall_by_turn(records: list[dict]) -> list[dict]:
    """Recall@k of the target per turn index, over all 200 sessions.

    Computed on the full 10-turn trajectory, so it is retrieval-quality
    independent -- a real run truncates the moment it hits, which would bias
    later turns toward the sessions that are hardest.
    """
    rows = []
    for turn in range(1, MAX_TURNS + 1):
        row = {"turn": turn}
        for depth in DEPTHS:
            hits = sum(1 for r in records if r["target"] in r["bm25"][turn - 1][:depth])
            row[f"recall@{depth}"] = round(hits / len(records), 4)
        ranks = []
        for r in records:
            shortlist = r["bm25"][turn - 1][:10]
            ranks.append(1.0 / (shortlist.index(r["target"]) + 1)
                         if r["target"] in shortlist else 0.0)
        row["turn_mrr@10"] = round(sum(ranks) / len(ranks), 4)
        rows.append(row)
    return rows


def report(ledger: str) -> dict:
    records = load_trajectories(ledger)
    baseline = play(records, bm25_ranker())
    base_score = score(baseline)

    print(f"\n{'=' * 78}\nPART 1 -- RERANK CEILING   (ledger: {ledger})\n{'=' * 78}")
    print(f"\nR1  BM25 only: {json.dumps(base_score)}")

    print("\n-- Per-turn-index recall of the target, BM25, all 200 sessions --")
    rows = recall_by_turn(records)
    print(f"{'turn':>4} {'R@10':>8} {'R@50':>8} {'R@100':>8} {'MRR@10':>8} "
          f"{'headroom(50-10)':>16}")
    for row in rows:
        print(f"{row['turn']:>4} {row['recall@10']:>8} {row['recall@50']:>8} "
              f"{row['recall@100']:>8} {row['turn_mrr@10']:>8} "
              f"{round(row['recall@50'] - row['recall@10'], 4):>16}")

    arms = {}
    for depth in DEPTHS:
        sessions = play(records, oracle_ranker(depth))
        arms[depth] = {
            "score": score(sessions),
            "paired": paired(baseline, sessions),
            "delta_ci": bootstrap_delta_ci(baseline, sessions),
            "by_scenario": by_scenario(sessions),
        }

    print("\n-- Oracle rerank (no real reranker can beat these) --")
    print(f"{'arm':<22} {'Tech':>9} {'delta':>9} {'Hit@10':>8} {'MRR':>9} "
          f"{'MTTC':>7} {'Eff':>8} {'win/loss/tie':>16}")
    print(f"{'R1 BM25 only':<22} {base_score['technical_score']:>9} {'--':>9} "
          f"{base_score['hit_rate_at_10']:>8} {base_score['mrr']:>9} "
          f"{base_score['mttc']:>7} {base_score['efficiency']:>8} {'--':>16}")
    for depth, arm in arms.items():
        s, p = arm["score"], arm["paired"]
        print(f"{'oracle rerank top-' + str(depth):<22} {s['technical_score']:>9} "
              f"{round(s['technical_score'] - base_score['technical_score'], 6):>9} "
              f"{s['hit_rate_at_10']:>8} {s['mrr']:>9} {s['mttc']:>7} "
              f"{s['efficiency']:>8} "
              f"{str(p['wins']) + '/' + str(p['losses']) + '/' + str(p['ties']):>16}")

    print("\n-- Baseline sampling spread (200 sessions, 2000 bootstrap draws) --")
    print(f"R1 TechnicalScore bootstrap: {json.dumps(bootstrap_ci(baseline))}")
    print("\n-- Paired bootstrap on the oracle deltas --")
    for depth, arm in arms.items():
        print(f"  oracle top-{depth:<4} {json.dumps(arm['delta_ci'])}")

    hit_sessions = [s for s in baseline if s["hit"]]
    at_rank_1 = sum(1 for s in hit_sessions if s["best_rank"] == 1)
    print(f"\n-- Where the MRR headroom physically is --")
    print(f"sessions that hit at all: {len(hit_sessions)}/{len(baseline)}")
    print(f"  already at rank 1     : {at_rank_1} "
          f"({at_rank_1 / len(hit_sessions):.1%} of hits) -- a reranker cannot improve these")
    print(f"  at rank 2-10          : {len(hit_sessions) - at_rank_1} "
          f"-- the only sessions any top-10 reranker can move")
    from collections import Counter
    dist = Counter(s["best_rank"] for s in hit_sessions)
    print("  rank histogram        : "
          + ", ".join(f"{k}:{dist[k]}" for k in sorted(dist)))

    return {"ledger": ledger, "r1": base_score, "recall_by_turn": rows,
            "r1_bootstrap": bootstrap_ci(baseline),
            "oracle": {str(d): a for d, a in arms.items()}}


if __name__ == "__main__":
    out = {ledger: report(ledger) for ledger in ("current", "legacy")}
    path = ROOT / "bakeoff" / "results-part1.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
