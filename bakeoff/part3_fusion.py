"""Part 3 -- retrieval arms. R1 BM25, R2 dense, R3 RRF, R4 weighted sweep.

One variable at a time, no reranking in any of these.

R4's weight sweep is the arm worth reading carefully. If its optimum sits at or
near w=0 -- "ignore the dense half" -- that is a clean negative result and is
more useful than a tuned near-tie, so it is reported as the finding rather than
buried under a best-weight number.

Every arm also gets a paired per-session comparison against R1. A mean gain
built from 40 wins and 38 losses is a different finding from a small consistent
gain across 78 sessions, and the mean cannot tell them apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.part2_dense import dense_topk  # noqa: E402
from bakeoff.simulate import (  # noqa: E402
    bm25_ranker, bootstrap_delta_ci, load_trajectories, paired, play, score,
)

CACHE = ROOT / "bakeoff" / "cache"
RRF_K = 60  # the r5 round's setting, kept so the arms are comparable


def _minmax(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    values = [v for _, v in pairs]
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return {k: 1.0 for k, _ in pairs}
    return {k: (v - low) / (high - low) for k, v in pairs}


def dense_ranker(dense: dict, depth: int):
    return lambda record, turn: dense[record["queries"][turn - 1]][:depth]


def rrf_ranker(dense: dict, depth: int, k: int = RRF_K):
    def rank(record: dict, turn: int) -> list[str]:
        lists = [record["bm25"][turn - 1][:depth], dense[record["queries"][turn - 1]][:depth]]
        fused: dict[str, float] = {}
        for candidates in lists:
            for position, asin in enumerate(candidates, 1):
                fused[asin] = fused.get(asin, 0.0) + 1.0 / (k + position)
        return sorted(fused, key=lambda a: -fused[a])
    return rank


def weighted_ranker(dense_scores: dict, bm25_scored: dict, weight: float, depth: int):
    """combined = w * dense_norm + (1 - w) * bm25_norm over the union."""
    def rank(record: dict, turn: int) -> list[str]:
        query = record["queries"][turn - 1]
        bm = _minmax([(a, s) for a, s in bm25_scored.get(query, [])][:depth])
        dn = _minmax(dense_scores[query][:depth])
        combined = {a: weight * dn.get(a, 0.0) + (1 - weight) * bm.get(a, 0.0)
                    for a in set(bm) | set(dn)}
        return sorted(combined, key=lambda a: -combined[a])
    return rank


def dense_scored(model: str, depth: int) -> dict[str, list[tuple[str, float]]]:
    """Dense top-`depth` with cosine scores attached (needed by R4)."""
    import numpy as np
    cached = CACHE / f"dense-top{depth}-scored-{model}.json"
    if cached.exists():
        return {q: [(a, s) for a, s in v]
                for q, v in json.loads(cached.read_text(encoding="utf-8")).items()}
    asins = np.array(json.loads((CACHE / "asins.json").read_text(encoding="utf-8")))
    queries = json.loads((CACHE / "queries.json").read_text(encoding="utf-8"))
    docs = np.load(CACHE / f"docs-{model}.npy")
    qvecs = np.load(CACHE / f"queries-{model}.npy")
    result: dict[str, list] = {}
    for start in range(0, len(queries), 256):
        block = qvecs[start:start + 256] @ docs.T
        top = np.argpartition(-block, depth, axis=1)[:, :depth]
        for row in range(block.shape[0]):
            idx = top[row][np.argsort(-block[row, top[row]])]
            result[queries[start + row]] = [[a, round(float(s), 5)]
                                            for a, s in zip(asins[idx].tolist(), block[row, idx])]
    cached.write_text(json.dumps(result), encoding="utf-8")
    return {q: [(a, s) for a, s in v] for q, v in result.items()}


def report(model: str, ledger: str, depth: int) -> dict:
    records = load_trajectories(ledger)
    dense_ids = dense_topk(model)
    dense_sc = dense_scored(model, depth)
    bm25_sc = json.loads((CACHE / "bm25-top100-scored.json").read_text(encoding="utf-8"))

    baseline = play(records, bm25_ranker())
    base = score(baseline)

    print(f"\n{'=' * 92}\nPART 3 -- RETRIEVAL ARMS  "
          f"(model: {model}, ledger: {ledger}, fusion depth: top-{depth})\n{'=' * 92}")
    print(f"{'arm':<34} {'Tech':>9} {'delta':>10} {'Hit@10':>8} {'MRR':>9} "
          f"{'MTTC':>7} {'Eff':>8} {'win/loss/tie':>15}")

    def line(label: str, sessions: list[dict]) -> dict:
        s = score(sessions)
        p = paired(baseline, sessions)
        print(f"{label:<34} {s['technical_score']:>9} "
              f"{round(s['technical_score'] - base['technical_score'], 6):>10} "
              f"{s['hit_rate_at_10']:>8} {s['mrr']:>9} {s['mttc']:>7} {s['efficiency']:>8} "
              f"{str(p['wins']) + '/' + str(p['losses']) + '/' + str(p['ties']):>15}")
        return {"score": s, "paired": p}

    arms = {}
    print(f"{'R1  BM25 only (baseline)':<34} {base['technical_score']:>9} {'--':>10} "
          f"{base['hit_rate_at_10']:>8} {base['mrr']:>9} {base['mttc']:>7} "
          f"{base['efficiency']:>8} {'--':>15}")
    arms["R1"] = {"score": base}

    r2 = play(records, dense_ranker(dense_ids, depth))
    arms["R2"] = line(f"R2  dense only ({model})", r2)
    arms["R2"]["delta_ci"] = bootstrap_delta_ci(baseline, r2)

    r3 = play(records, rrf_ranker(dense_ids, depth))
    arms["R3"] = line(f"R3  RRF k={RRF_K}", r3)
    arms["R3"]["delta_ci"] = bootstrap_delta_ci(baseline, r3)

    print()
    sweep = {}
    for step in range(11):
        weight = step / 10
        sessions = play(records, weighted_ranker(dense_sc, bm25_sc, weight, depth))
        sweep[f"{weight:.1f}"] = line(f"R4  weighted w(dense)={weight:.1f}", sessions)
    arms["R4"] = sweep
    peak = max(sweep, key=lambda w: sweep[w]["score"]["technical_score"])
    print(f"\nR4 peak at w(dense)={peak} "
          f"-> {sweep[peak]['score']['technical_score']} "
          f"(delta {round(sweep[peak]['score']['technical_score'] - base['technical_score'], 6)})")
    if float(peak) <= 0.1:
        print("  ==> the optimum is at or adjacent to pure BM25: the dense half "
              "earns nothing under fusion.")
    return {"model": model, "ledger": ledger, "depth": depth,
            "arms": arms, "r4_peak_weight": peak}


if __name__ == "__main__":
    out = {}
    for model_name in ("minilm", "bge"):
        for depth_value in (50, 100):
            out[f"{model_name}/current/top{depth_value}"] = report(model_name, "current", depth_value)
    out["minilm/legacy/top50"] = report("minilm", "legacy", 50)
    path = ROOT / "bakeoff" / "results-part3.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
