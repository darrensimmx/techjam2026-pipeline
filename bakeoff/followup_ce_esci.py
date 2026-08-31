"""Does the cross-encoder earn its place where the leak is absent?

Follow-up C showed the phrase gain is ~entirely a property of the simulator
copying constraint strings out of the target listing. That cuts both ways, and
the second way is the one this measures.

If BM25's dominance on the public set is a harness artifact, then the component
that matters for "does the architecture hold under real-world conditions"
(problem-statement.md:136 -- Feasibility) is whichever one still works when the
copying stops. Part 5 already showed dense retrieval wins there. The
cross-encoder was never run there, and it is the component actually under
consideration.

Cleaner than perturbing the public set's strings ourselves: perturbation means
inventing the distribution that decides the answer, which is the trap Part 5
caught. ESCI's queries were written by Amazon customers.

This produces no TechnicalScore. It answers one question: on real human queries,
does reranking BM25's shortlist recover what BM25 misses?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.dense import catalog_documents
from bakeoff.part3_fusion import _minmax
from bakeoff.part5_realqueries import first_rank

CACHE = ROOT / "bakeoff" / "cache"
CORPUS = CACHE / "esci_catalog.jsonl"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEPTH = 20      # default; --depth overrides. See the note in main().


def main() -> None:
    from sentence_transformers import CrossEncoder

    # DEPTH and the output path are flags rather than constants because the LLM
    # probe's depth-50 arm is only honest against a CE that actually scored 50.
    # With a top-20 cache the probe leaves BM25 ranks 21-50 unranked, its CE
    # baseline at depth 20 and 50 come out bit-identical, and the LLM gets
    # credited with recovery work the cross-encoder would have done. Editing a
    # module constant to produce that second cache would leave the number
    # unreproducible from committed source, which is how report.md section 3
    # ended up with a table nobody can regenerate.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--out", default=None,
                        help="score cache path (default: cache/ce-esci-top<depth>.json)")
    args = parser.parse_args()
    depth = args.depth

    blob = json.loads((CACHE / "esci_queries.json").read_text(encoding="utf-8"))
    entries = blob["queries"]
    bm25 = json.loads((CACHE / "esci-bm25.json").read_text(encoding="utf-8"))
    asins, documents = catalog_documents(CORPUS)
    text = dict(zip(asins, documents))

    # Persisted, unlike the first run -- the same mistake Part 4 made, and it
    # cost a re-run when a confidence interval was wanted afterwards.
    score_path = Path(args.out) if args.out else CACHE / f"ce-esci-top{depth}.json"
    # Seed from any narrower cache so only the new ranks are scored -- the merge
    # below is what makes a top-20 -> top-50 widening cost 30 pairs per query
    # rather than 50.
    if not score_path.exists():
        candidates = [(int(f.stem.rsplit("top", 1)[1]), f)
                      for f in CACHE.glob("ce-esci-top*.json")
                      if f.stem.rsplit("top", 1)[1].isdigit()]
        for _d, narrower in sorted(c for c in candidates if c[0] < depth):
            score_path.write_text(narrower.read_text(encoding="utf-8"),
                                  encoding="utf-8")
            print(f"seeded {score_path.name} from {narrower.name}", flush=True)
            break
    model = CrossEncoder(CE_MODEL, max_length=256, device="cpu")
    scores: dict[str, float] = (json.loads(score_path.read_text(encoding="utf-8"))
                                if score_path.exists() else {})
    t0 = time.time()
    for position, entry in enumerate(entries, 1):
        query = entry["query"]
        head = [a for a, _ in bm25.get(query, [])][:depth]
        head = [a for a in head if f"{query}\x1f{a}" not in scores]
        if head:
            values = model.predict([[query, text.get(a, "")] for a in head],
                                   batch_size=64, show_progress_bar=False)
            for asin, value in zip(head, values):
                scores[f"{query}\x1f{asin}"] = float(value)
        if position % 150 == 0:
            print(f"  {position}/{len(entries)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'arm':<26}{'R@10':>8}{'MRR@10':>9}   vs BM25")
    score_path.write_text(json.dumps(scores), encoding="utf-8")
    rows = {}
    base = None
    for label, alpha in (("BM25 (no rerank)", None), ("+ CE pure (a=0.0)", 0.0),
                         ("+ CE blended a=0.25", 0.25), ("+ CE blended a=0.5", 0.5)):
        ranks = []
        for entry in entries:
            query, targets = entry["query"], set(entry["targets"])
            full = [a for a, _ in bm25.get(query, [])]
            if alpha is None or not full:
                ranks.append(first_rank(full, targets, 100)); continue
            head = full[:depth]
            bn = _minmax([(a, s) for a, s in bm25[query]][:depth])
            cn = _minmax([(a, scores.get(f"{query}\x1f{a}", 0.0)) for a in head])
            combined = {a: alpha * bn.get(a, 0.0) + (1 - alpha) * cn.get(a, 0.0) for a in head}
            ranks.append(first_rank(sorted(combined, key=lambda a: -combined[a]) + full[depth:],
                                    targets, 100))
        row = {"recall@10": round(sum(1 for r in ranks if r and r <= 10)/len(ranks), 4),
               "mrr@10": round(sum(1.0/r for r in ranks if r and r <= 10)/len(ranks), 4)}
        rows[label] = row
        d = "" if base is None else (f"  R@10 {row['recall@10']-base['recall@10']:+.4f}"
                                     f"  MRR {row['mrr@10']-base['mrr@10']:+.4f}")
        base = base or row
        print(f"{label:<26}{row['recall@10']:>8}{row['mrr@10']:>9}{d}")

    suffix = "" if depth == DEPTH else f"-top{depth}"
    out = ROOT / "bakeoff" / f"results-followup-ce-esci{suffix}.json"
    out.write_text(json.dumps({"source": blob["source"], "queries": len(entries),
                               "depth": depth, "arms": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
