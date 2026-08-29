"""Part 2 -- the dense ceiling.

Same logic as Part 1, on the retrieval side instead of the rerank side. If the
union of BM25 and dense finds the target barely more often than BM25 alone,
then dense has nothing to contribute and no fusion scheme can conjure it --
fusion can only reorder what one of its inputs already found.

Segmented by turn index because turn 1 (one vague opening line) and turn 6 (a
fully accumulated ledger of strings copied out of the target listing) are
different retrieval problems, and an average across them hides the answer. The
standing prior is that dense helps at turn 1 and decays as the ledger fills;
this confirms or refutes that shape explicitly rather than assuming it.

Writes dense top-100 lists to cache so Part 3's fusion arms are free.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.simulate import MAX_TURNS, load_trajectories  # noqa: E402

CACHE = ROOT / "bakeoff" / "cache"
DEPTH = 100


def dense_topk(model: str, depth: int = DEPTH) -> dict[str, list[str]]:
    """query text -> top-`depth` asins, exact cosine search over the catalog."""
    cached = CACHE / f"dense-top{depth}-{model}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    asins = json.loads((CACHE / "asins.json").read_text(encoding="utf-8"))
    queries = json.loads((CACHE / "queries.json").read_text(encoding="utf-8"))
    docs = np.load(CACHE / f"docs-{model}.npy")
    qvecs = np.load(CACHE / f"queries-{model}.npy")
    asin_array = np.array(asins)

    result: dict[str, list[str]] = {}
    chunk = 256
    for start in range(0, len(queries), chunk):
        block = qvecs[start:start + chunk] @ docs.T          # cosine, both L2-normalised
        top = np.argpartition(-block, depth, axis=1)[:, :depth]
        for row in range(block.shape[0]):
            idx = top[row]
            idx = idx[np.argsort(-block[row, idx])]
            result[queries[start + row]] = asin_array[idx].tolist()
    cached.write_text(json.dumps(result), encoding="utf-8")
    return result


def report(model: str, ledger: str) -> dict:
    records = load_trajectories(ledger)
    dense = dense_topk(model)

    print(f"\n{'=' * 86}\nPART 2 -- DENSE CEILING   (model: {model}, ledger: {ledger})\n{'=' * 86}")
    print(f"\n{'turn':>4} {'BM25@10':>9} {'BM25@50':>9} {'dense@10':>9} {'dense@50':>9} "
          f"{'union@10':>9} {'union@50':>9} {'d-only@50':>10} {'lift@50':>9}")
    rows = []
    for turn in range(1, MAX_TURNS + 1):
        stats = {"turn": turn}
        for depth in (10, 50):
            bm = [r["target"] in r["bm25"][turn - 1][:depth] for r in records]
            dn = [r["target"] in dense[r["queries"][turn - 1]][:depth] for r in records]
            union = [b or d for b, d in zip(bm, dn)]
            stats[f"bm25@{depth}"] = round(sum(bm) / len(bm), 4)
            stats[f"dense@{depth}"] = round(sum(dn) / len(dn), 4)
            stats[f"union@{depth}"] = round(sum(union) / len(union), 4)
            stats[f"dense_only@{depth}"] = sum(1 for b, d in zip(bm, dn) if d and not b)
            stats[f"lift@{depth}"] = round(stats[f"union@{depth}"] - stats[f"bm25@{depth}"], 4)
        rows.append(stats)
        print(f"{turn:>4} {stats['bm25@10']:>9} {stats['bm25@50']:>9} "
              f"{stats['dense@10']:>9} {stats['dense@50']:>9} "
              f"{stats['union@10']:>9} {stats['union@50']:>9} "
              f"{stats['dense_only@50']:>10} {stats['lift@50']:>9}")

    # Session-level: does dense EVER rescue a session BM25 never finds?
    bm_any = {r["sample_id"]: any(r["target"] in r["bm25"][t][:50] for t in range(MAX_TURNS))
              for r in records}
    dn_any = {r["sample_id"]: any(r["target"] in dense[r["queries"][t]][:50]
                                  for t in range(MAX_TURNS)) for r in records}
    rescues = [r for r in records if dn_any[r["sample_id"]] and not bm_any[r["sample_id"]]]
    losses = [r for r in records if bm_any[r["sample_id"]] and not dn_any[r["sample_id"]]]
    print(f"\nsession level, top-50, any turn:")
    print(f"  BM25 finds target            : {sum(bm_any.values())}/{len(records)}")
    print(f"  dense finds target           : {sum(dn_any.values())}/{len(records)}")
    print(f"  union                        : "
          f"{sum(1 for r in records if bm_any[r['sample_id']] or dn_any[r['sample_id']])}"
          f"/{len(records)}")
    print(f"  dense-only rescues           : {len(rescues)}")
    print(f"  BM25-only (dense would lose) : {len(losses)}")

    # Median dense rank of the target where BM25 already has it at rank 1 --
    # the r5 round's stated mechanism for why RRF dilutes.
    dense_ranks, bm25_ranks = [], []
    for r in records:
        final = MAX_TURNS - 1
        dlist, blist = dense[r["queries"][final]], r["bm25"][final]
        if r["target"] in dlist:
            dense_ranks.append(dlist.index(r["target"]) + 1)
        if r["target"] in blist:
            bm25_ranks.append(blist.index(r["target"]) + 1)
    for label, ranks in (("BM25", bm25_ranks), ("dense", dense_ranks)):
        ranks = sorted(ranks)
        if ranks:
            print(f"  final-turn {label} rank of target when found in top-100: "
                  f"median {ranks[len(ranks) // 2]}, n={len(ranks)}")

    print(f"\n-- up to 10 real dense-only rescues (turn 1, top-50) --")
    shown = 0
    for r in records:
        if shown >= 10:
            break
        query = r["queries"][0]
        if r["target"] in dense[query][:50] and r["target"] not in r["bm25"][0][:50]:
            print(f"  [{r['sample_id']} / {r['scenario_type']}] target={r['target']}")
            print(f"    query: {query[:160]}")
            shown += 1
    if shown == 0:
        print("  none at turn 1")

    return {"model": model, "ledger": ledger, "by_turn": rows,
            "session_bm25_any": sum(bm_any.values()),
            "session_dense_any": sum(dn_any.values()),
            "dense_only_rescues": [r["sample_id"] for r in rescues],
            "bm25_only": [r["sample_id"] for r in losses]}


if __name__ == "__main__":
    models = sys.argv[1:] or ["minilm", "bge"]
    out = {}
    for model_name in models:
        for ledger_name in ("current", "legacy"):
            out[f"{model_name}/{ledger_name}"] = report(model_name, ledger_name)
    path = ROOT / "bakeoff" / "results-part2.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
