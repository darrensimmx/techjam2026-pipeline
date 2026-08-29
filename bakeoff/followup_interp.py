"""Follow-up B -- score interpolation, and the two robust levers combined.

Two questions Part 4 left open.

**Does interpolation beat pure rerank?** Part 4 replaced BM25's ordering
outright, so the cross-encoder could demote a correct rank-1 hit -- and the
losses show it did: 47/200 at top-20, 59/200 at top-50, with *zero* extra wins
between those depths. Keeping BM25 as a prior and letting the reranker perturb
it is the standard fix, and it is exactly what made weighted fusion beat RRF in
Part 3.

    final = alpha * BM25_norm + (1 - alpha) * CE_norm

alpha = 1.0 is pure BM25 (no rerank), alpha = 0.0 is Part 4's pure rerank, so
the sweep contains both endpoints and the arms stay comparable.

**Do phrase retrieval and reranking compound?** A better shortlist is a better
input to a reranker, so the two should add rather than compete. Both retrieval
variants are run under every rerank arm to check that, rather than assuming it.

Cross-encoder scores are cached to disk here, unlike Part 4 which held them in
memory only -- that is what made Part 4 unabandonable once started.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.bm25_scores import ScoringIndex  # noqa: E402
from bakeoff.dense import catalog_documents  # noqa: E402
from bakeoff.followup_phrase import build_query, disclosed_per_turn  # noqa: E402
from bakeoff.part3_fusion import _minmax  # noqa: E402
from bakeoff.simulate import (  # noqa: E402
    MAX_TURNS, bootstrap_delta_ci, load_trajectories, paired, play, score,
)

CACHE = ROOT / "bakeoff" / "cache"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def retrieval_lists(records: list[dict], variant: str, ledger: str) -> dict[str, list]:
    """key 'sample|turn' -> [(asin, bm25_score), ...] top-100."""
    path = CACHE / f"retrieval-{variant}-{ledger}.json"
    if path.exists():
        return {k: [(a, s) for a, s in v]
                for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    index = ScoringIndex(ROOT / "data" / "catalog.jsonl")
    disclosed = {r["sample_id"]: disclosed_per_turn(r) for r in records}
    out: dict[str, list] = {}
    for record in records:
        for turn in range(1, MAX_TURNS + 1):
            key = f"{record['sample_id']}|{turn}"
            mode = "phrase_plus" if variant == "phrase" else "unigram"
            expression = build_query(disclosed[record["sample_id"]][turn - 1],
                                     record["queries"][turn - 1], mode)
            try:
                out[key] = index.search_expression(expression, 100) if expression else []
            except Exception:
                out[key] = []
    path.write_text(json.dumps({k: [[a, round(s, 5)] for a, s in v]
                                for k, v in out.items()}), encoding="utf-8")
    return out


def ce_scores(records: list[dict], lists: dict, variant: str, ledger: str,
              depth: int) -> dict[str, float]:
    """(query, asin) -> cross-encoder score, cached on disk."""
    path = CACHE / f"ce-{variant}-{ledger}-top{depth}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CE_MODEL, max_length=256, device="cpu")
    asins, documents = catalog_documents(ROOT / "data" / "catalog.jsonl")
    text = dict(zip(asins, documents))
    cache: dict[str, float] = {}
    t0 = time.time()
    for position, record in enumerate(records, 1):
        for turn in range(1, MAX_TURNS + 1):
            query = record["queries"][turn - 1]
            shortlist = [a for a, _ in lists[f"{record['sample_id']}|{turn}"][:depth]]
            missing = [a for a in shortlist if f"{query}{a}" not in cache]
            if not missing:
                continue
            values = model.predict([[query, text.get(a, "")] for a in missing],
                                   batch_size=64, show_progress_bar=False)
            for asin, value in zip(missing, values):
                cache[f"{query}{asin}"] = float(value)
        if position % 50 == 0:
            print(f"    ce {variant}/top{depth}: {position}/{len(records)} "
                  f"({time.time() - t0:.0f}s, {len(cache)} pairs)", flush=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
    print(f"    ce {variant}/top{depth}: {len(cache)} pairs in {time.time() - t0:.0f}s",
          flush=True)
    return cache


def interp_ranker(lists: dict, ce: dict, depth: int, alpha: float):
    def rank(record: dict, turn: int) -> list[str]:
        key = f"{record['sample_id']}|{turn}"
        head = lists[key][:depth]
        if not head:
            return []
        query = record["queries"][turn - 1]
        bm = _minmax(head)
        ce_pairs = [(a, ce.get(f"{query}{a}", 0.0)) for a, _ in head]
        cn = _minmax(ce_pairs)
        combined = {a: alpha * bm.get(a, 0.0) + (1 - alpha) * cn.get(a, 0.0)
                    for a, _ in head}
        ordered = sorted(combined, key=lambda a: -combined[a])
        return ordered + [a for a, _ in lists[key][depth:]]
    return rank


def main() -> None:
    ledger = sys.argv[1] if len(sys.argv) > 1 else "current"
    records = load_trajectories(ledger)
    results = {}

    for variant in ("unigram", "phrase"):
        lists = retrieval_lists(records, variant, ledger)
        no_rerank = play(records, lambda r, t: [a for a, _ in lists[f"{r['sample_id']}|{t}"][:10]])
        base = score(no_rerank)
        print(f"\n{'=' * 96}\nFOLLOW-UP B -- {variant.upper()} retrieval  (ledger: {ledger})\n{'=' * 96}")
        print(f"{'arm':<34} {'Tech':>9} {'delta':>10} {'Hit@10':>8} {'MRR':>9} "
              f"{'MTTC':>7} {'win/loss/tie':>15}")
        print(f"{'no rerank':<34} {base['technical_score']:>9} {'--':>10} "
              f"{base['hit_rate_at_10']:>8} {base['mrr']:>9} {base['mttc']:>7} {'--':>15}")

        for depth in (10, 20):
            ce = ce_scores(records, lists, variant, ledger, depth)
            for alpha in ALPHAS:
                arm = play(records, interp_ranker(lists, ce, depth, alpha))
                s, p = score(arm), paired(no_rerank, arm)
                tag = ("pure CE" if alpha == 0.0 else
                       "pure BM25" if alpha == 1.0 else f"a={alpha}")
                label = f"top-{depth}  {tag}"
                results[f"{variant}/top{depth}/a{alpha}"] = {
                    "score": s, "paired": p,
                    "delta_ci": bootstrap_delta_ci(no_rerank, arm)}
                print(f"{label:<34} {s['technical_score']:>9} "
                      f"{round(s['technical_score'] - base['technical_score'], 6):>10} "
                      f"{s['hit_rate_at_10']:>8} {s['mrr']:>9} {s['mttc']:>7} "
                      f"{str(p['wins']) + '/' + str(p['losses']) + '/' + str(p['ties']):>15}")
        results[f"{variant}/no_rerank"] = {"score": base}

    out = ROOT / "bakeoff" / f"results-followup-interp-{ledger}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
