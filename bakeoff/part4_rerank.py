"""Part 4 -- rerank arms. K0 none, K1 local cross-encoder over BM25's top-N.

Part 1 showed the ceiling is large, so this is worth building. It is applied to
R1 (BM25) because Part 3 found nothing that beats R1 to apply it to.

K2 (LLM rerank over top-50, networking enabled) is NOT built. That is a
deliberate refusal, reported as a finding rather than skipped silently:
`submission_rules.md:59` says organizer policy may disable network access for
official final scoring, and the evaluator swallows any exception from
`respond()` into an empty response (`local_evaluator.py:239-242`) -- so a
network rerank that fails under offline scoring scores zero for that turn
without raising anything a log would show. The number K2 would produce with
networking on is not the number it would score, and the honest version of that
arm is the offline one, which is K1.

N is swept (10 / 20 / 50) because depth is the cost lever: the cross-encoder is
linear in candidates, and the planning repo records top-20 as untested.

Wall-clock is measured on the *uncached* path and reported per turn, because a
reranker that improves MRR while degrading Efficiency can be net-negative and
the aggregate TechnicalScore hides that trade.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.dense import catalog_documents  # noqa: E402
from bakeoff.simulate import (  # noqa: E402
    MAX_TURNS, bm25_ranker, bootstrap_delta_ci, by_scenario, load_trajectories,
    paired, play, score,
)

CACHE = ROOT / "bakeoff" / "cache"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, depth: int) -> None:
        from sentence_transformers import CrossEncoder
        self.depth = depth
        self.model = CrossEncoder(MODEL, max_length=256, device="cpu")
        asins, documents = catalog_documents(ROOT / "data" / "catalog.jsonl")
        self.text = dict(zip(asins, documents))
        self.cache: dict[tuple[str, str], float] = {}
        self.pairs_scored = 0
        self.seconds = 0.0
        self.turns_reranked = 0

    def rank(self, record: dict, turn: int) -> list[str]:
        query = record["queries"][turn - 1]
        shortlist = record["bm25"][turn - 1][:self.depth]
        if not shortlist:
            return []
        self.turns_reranked += 1
        missing = [a for a in shortlist if (query, a) not in self.cache]
        if missing:
            t0 = time.perf_counter()
            scores = self.model.predict([[query, self.text.get(a, "")] for a in missing],
                                        batch_size=64, show_progress_bar=False)
            self.seconds += time.perf_counter() - t0
            self.pairs_scored += len(missing)
            for asin, value in zip(missing, scores):
                self.cache[(query, asin)] = float(value)
        return sorted(shortlist, key=lambda a: -self.cache[(query, a)])


def model_size_mb() -> float:
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(MODEL))
    weights = [p for p in root.rglob("*")
               if p.is_file() and p.suffix in {".bin", ".safetensors", ".onnx"}]
    # A submission bundles one weight format, not all of them.
    return max((p.stat().st_size for p in weights), default=0) / 1e6


def report(ledger: str) -> dict:
    records = load_trajectories(ledger)
    baseline = play(records, bm25_ranker())
    base = score(baseline)

    print(f"\n{'=' * 96}\nPART 4 -- RERANK ARMS   (ledger: {ledger}, model: {MODEL})\n{'=' * 96}")
    print(f"{'arm':<30} {'Tech':>9} {'delta':>10} {'Hit@10':>8} {'MRR':>9} {'MTTC':>7} "
          f"{'Eff':>8} {'win/loss/tie':>15} {'s/turn':>8}")
    print(f"{'K0  none (R1 BM25)':<30} {base['technical_score']:>9} {'--':>10} "
          f"{base['hit_rate_at_10']:>8} {base['mrr']:>9} {base['mttc']:>7} "
          f"{base['efficiency']:>8} {'--':>15} {0.0:>8}")

    arms = {}
    for depth in (10, 20, 50):
        reranker = CrossEncoderReranker(depth)
        sessions = play(records, reranker.rank)
        s, p = score(sessions), paired(baseline, sessions)
        per_turn = reranker.seconds / max(reranker.turns_reranked, 1)
        print(f"{'K1  cross-encoder top-' + str(depth):<30} {s['technical_score']:>9} "
              f"{round(s['technical_score'] - base['technical_score'], 6):>10} "
              f"{s['hit_rate_at_10']:>8} {s['mrr']:>9} {s['mttc']:>7} {s['efficiency']:>8} "
              f"{str(p['wins']) + '/' + str(p['losses']) + '/' + str(p['ties']):>15} "
              f"{per_turn:>8.3f}")
        arms[f"top{depth}"] = {
            "score": s, "paired": p,
            "delta_ci": bootstrap_delta_ci(baseline, sessions),
            "by_scenario": by_scenario(sessions),
            "pairs_scored": reranker.pairs_scored,
            "seconds_total": round(reranker.seconds, 2),
            "seconds_per_reranked_turn": round(per_turn, 4),
            "turns_reranked": reranker.turns_reranked,
        }

    print("\n-- cost --")
    size = model_size_mb()
    print(f"  largest single weight file: {size:.1f} MB")
    print(f"  submission_rules.md states NO numeric size limit; the governing phrase is "
          f"'lightweight local assets\\nrequired by your agent' (:41), and :100-101 "
          f"reserves CPU / memory / timeout restrictions.")
    for depth, arm in arms.items():
        print(f"  {depth:<6} {arm['pairs_scored']:>7} pairs, "
              f"{arm['seconds_total']:>7.1f}s total, "
              f"{arm['seconds_per_reranked_turn']:.3f}s per reranked turn")

    print("\n-- paired bootstrap on the deltas --")
    for depth, arm in arms.items():
        print(f"  {depth:<6} {json.dumps(arm['delta_ci'])}")

    return {"ledger": ledger, "k0": base, "k1": arms, "weights_mb": round(size, 1)}


if __name__ == "__main__":
    out = {ledger: report(ledger) for ledger in (sys.argv[1:] or ["current", "legacy"])}
    path = ROOT / "bakeoff" / "results-part4.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
