"""Replay a captured trajectory under an arbitrary ranking function.

Reproduces `evaluator/local_evaluator.py`'s scoring exactly, including its
rounding order (:188-201, :278-280): hit_rate / mrr / mttc are each rounded to
6dp *before* efficiency and TechnicalScore are computed from them.

`validate` proves the replay against a real `evaluate()` run. Nothing
downstream is trustworthy unless that check passes.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_TURNS = 10
TOP_K = 10

Ranker = Callable[[dict, int], list[str]]


def load_trajectories(ledger: str = "current") -> list[dict]:
    path = ROOT / "bakeoff" / f"trajectories-{ledger}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def bm25_ranker(depth: int = TOP_K) -> Ranker:
    return lambda record, turn: record["bm25"][turn - 1][:depth]


def play(records: Iterable[dict], ranker: Ranker) -> list[dict]:
    """One session outcome per record, in evaluator field names."""
    sessions = []
    for record in records:
        hit_turn = None
        best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            ranked = ranker(record, turn)[:TOP_K]
            if turn >= record["scoreable_from"] and record["target"] in ranked:
                best_rank = ranked.index(record["target"]) + 1
                hit_turn = turn
                break
        sessions.append({
            "sample_id": record["sample_id"],
            "scenario_type": record["scenario_type"],
            "difficulty_bucket": record.get("difficulty_bucket"),
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    return sessions


def score(sessions: list[dict]) -> dict:
    """local_evaluator.metric_summary + :279-280, rounding order preserved."""
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None,
                "efficiency": 0.0, "technical_score": 0.0}
    hit_rate = round(sum(int(s["hit"]) for s in sessions) / len(sessions), 6)
    mrr = round(statistics.fmean(s["reciprocal_rank"] for s in sessions), 6)
    mttc = round(statistics.fmean(
        s["first_hit_turn"] if s["first_hit_turn"] is not None else MAX_TURNS + 1
        for s in sessions), 6)
    efficiency = round(max(0.0, min(1.0, (11.0 - mttc) / 10.0)), 6)
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": efficiency,
        "technical_score": round(0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency, 6),
    }


def by_scenario(sessions: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {name: score(grouped[name]) for name in sorted(grouped)}


def paired(baseline: list[dict], arm: list[dict]) -> dict:
    """Per-session win/loss/tie of `arm` against `baseline`, on reciprocal rank.

    A mean gain built from many wins and nearly as many losses is a different
    finding from a small consistent gain, and the mean cannot tell them apart
    (bakeoff-prompt.md, Part 3).
    """
    index = {s["sample_id"]: s for s in baseline}
    wins = losses = ties = 0
    deltas = []
    for session in arm:
        before = index[session["sample_id"]]["reciprocal_rank"]
        after = session["reciprocal_rank"]
        deltas.append(after - before)
        if after > before + 1e-12:
            wins += 1
        elif after < before - 1e-12:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties,
            "mean_rr_delta": round(statistics.fmean(deltas), 6),
            "regressed_fraction": round(losses / len(arm), 4)}


def bootstrap_ci(sessions: list[dict], draws: int = 2000, seed: int = 20260829) -> dict:
    """Resample sessions with replacement; the spread of TechnicalScore.

    The 200 public sessions are the only sampling variation this harness has --
    see part0_variance.py for why seed variance is not available here.
    """
    rng = random.Random(seed)
    n = len(sessions)
    scores = []
    for _ in range(draws):
        sample = [sessions[rng.randrange(n)] for _ in range(n)]
        scores.append(score(sample)["technical_score"])
    scores.sort()
    return {
        "mean": round(statistics.fmean(scores), 6),
        "sd": round(statistics.pstdev(scores), 6),
        "p2_5": round(scores[int(0.025 * draws)], 6),
        "p97_5": round(scores[int(0.975 * draws)], 6),
    }


def bootstrap_delta_ci(baseline: list[dict], arm: list[dict],
                       draws: int = 2000, seed: int = 20260829) -> dict:
    """Paired bootstrap on the TechnicalScore *difference*.

    Paired: the same resampled session indices are scored under both arms, so
    session-difficulty variance cancels and the CI is on the delta itself.
    """
    rng = random.Random(seed)
    order = {s["sample_id"]: i for i, s in enumerate(baseline)}
    arm_ordered: list = [None] * len(baseline)
    for session in arm:
        arm_ordered[order[session["sample_id"]]] = session
    n = len(baseline)
    deltas = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(score([arm_ordered[i] for i in idx])["technical_score"]
                      - score([baseline[i] for i in idx])["technical_score"])
    deltas.sort()
    return {
        "mean_delta": round(statistics.fmean(deltas), 6),
        "sd": round(statistics.pstdev(deltas), 6),
        "p2_5": round(deltas[int(0.025 * draws)], 6),
        "p97_5": round(deltas[int(0.975 * draws)], 6),
        "excludes_zero": bool(deltas[int(0.025 * draws)] > 0 or deltas[int(0.975 * draws)] < 0),
    }


def validate() -> None:
    """Replayed R1 must equal a real evaluate() run of the shipped Agent."""
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from starter.agent import Agent

    catalog_path = ROOT / "data" / "catalog.jsonl"
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    catalog_ids, categories, products = catalog_index(catalog_path)
    real = evaluate(Agent(catalog_path), samples, catalog_ids, categories, products)

    records = load_trajectories()
    replayed = play(records, bm25_ranker())
    mine = score(replayed)

    print("field                     real         replayed")
    ok = True
    pairs = [("hit_rate_at_10", "hit_rate_at_10"), ("mrr", "mrr"), ("mttc", "mttc"),
             ("efficiency", "efficiency"), ("recommended_technical_score", "technical_score")]
    for real_key, mine_key in pairs:
        same = abs(real[real_key] - mine[mine_key]) < 1e-9
        ok = ok and same
        print(f"{real_key:<25} {real[real_key]:<12} {mine[mine_key]:<12} "
              f"{'OK' if same else 'MISMATCH'}")

    real_sessions = {s["sample_id"]: s for s in real["sessions"]}
    diffs = [s["sample_id"] for s in replayed
             if (real_sessions[s["sample_id"]]["best_rank"],
                 real_sessions[s["sample_id"]]["first_hit_turn"])
             != (s["best_rank"], s["first_hit_turn"])]
    print(f"per-session mismatches: {len(diffs)}/{len(replayed)}"
          + (f" -> {diffs[:5]}" if diffs else ""))
    ok = ok and not diffs
    print("\nVALIDATION " + ("PASSED" if ok else "FAILED"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    validate()
