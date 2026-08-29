"""Part 5 -- does the BM25-over-dense ordering survive real human queries?

Part 5 of the prompt asks whether an arm's wins concentrate where the disclosed
strings are NOT verbatim substrings of the target listing, and warns that a gain
concentrated there is less bankable because we cannot know how many such
sessions the private set holds.

The public set can barely be asked: 94.5% of the simulator's disclosed
constraint strings are verbatim substrings of the target's own listing
(`overlap.py`), because `local_evaluator.py:52-71` copies them out of it. So a
BM25 win on the public set is consistent with two very different worlds -- BM25
is genuinely the right retriever for this task, or BM25 is being handed the
answer key by a simulator that copy-pastes. Distinguishing them needs a query
distribution nobody on this project authored.

This runs the identical arms over `bakeoff/cache/esci_catalog.jsonl` -- 50,000
real Amazon products in our catalog's schema, indexed by the shipped
`Bm25Index` with the shipped column weights -- against 1,500 real human Amazon
search queries with human Exact-relevance labels (ESCI, Reddy et al. 2022,
arXiv:2206.06588). See `esci.py` for how it is built and why it is a standalone
corpus rather than our own.

This produces no TechnicalScore and is not the competition task. It answers one
question: does the ordering hold, or invert, when the queries stop being copied
out of the target?

A correction to this script's own premise, kept because it was wrong
--------------------------------------------------------------------
This was built expecting ESCI to be the *low-overlap* condition, and the
query-token-coverage measure below was added to demonstrate that. **It does not.
ESCI scores higher, not lower: mean 0.815 / median 1.000, against the public
set's 0.724 / 0.714.**

That measure is confounded by query length, and the confound runs the wrong way.
A three-token human query ("$1 stuffed toy") has all of its tokens somewhere in a
long product listing, so coverage saturates at 1.0. A ten-turn accumulated ledger
is long, so its coverage is diluted no matter how verbatim its source. Token
coverage therefore measures brevity here at least as much as copying, and it
should not be quoted as evidence for either side.

The property that actually differs is *phrase-level* copying, which
`overlap.py` measures and this cannot: 94.5% of the simulator's disclosed
constraint strings are verbatim substrings of the target's own listing, because
the evaluator builds them by copying out of it. Nobody typing "wireless earbuds
for running" is copying a phrase out of the document they are looking for.

So the claim this script supports is the narrower one, and it is still the claim
that matters: these are queries nobody on this project authored, and which were
not generated from the target document. The coverage line is printed anyway,
against the result it was meant to support, rather than dropped.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.bm25_scores import ScoringIndex  # noqa: E402
from bakeoff.dense import MODELS, catalog_documents  # noqa: E402
from bakeoff.part3_fusion import RRF_K, _minmax  # noqa: E402
from starter.retrieval import _terms  # noqa: E402

CACHE = ROOT / "bakeoff" / "cache"
CORPUS = CACHE / "esci_catalog.jsonl"
DEPTH = 100


def first_rank(ranked: list[str], targets: set[str], depth: int) -> int | None:
    for position, asin in enumerate(ranked[:depth], 1):
        if asin in targets:
            return position
    return None


def encode_corpus(model: str) -> tuple[np.ndarray, list[str]]:
    from sentence_transformers import SentenceTransformer

    vector_path = CACHE / f"esci-docs-{model}.npy"
    asins, documents = catalog_documents(CORPUS)
    if vector_path.exists():
        return np.load(vector_path), asins
    encoder = SentenceTransformer(MODELS[model]["hf"], device="cpu")
    encoder.max_seq_length = 256
    t0 = time.time()
    vectors = encoder.encode(documents, batch_size=128, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=True)
    np.save(vector_path, vectors.astype(np.float32))
    print(f"[{model}] ESCI corpus encoded in {time.time() - t0:.0f}s", flush=True)
    return vectors.astype(np.float32), asins


def main() -> None:
    blob = json.loads((CACHE / "esci_queries.json").read_text(encoding="utf-8"))
    entries = blob["queries"]
    queries = [e["query"] for e in entries]
    targets = [set(e["targets"]) for e in entries]
    print(f"ESCI: {len(entries)} human queries over "
          f"{sum(1 for _ in CORPUS.open(encoding='utf-8'))} products")

    # BM25 over the same index implementation the agent ships.
    bm25_path = CACHE / "esci-bm25.json"
    if bm25_path.exists():
        bm25 = {q: [(a, s) for a, s in v]
                for q, v in json.loads(bm25_path.read_text(encoding="utf-8")).items()}
    else:
        index = ScoringIndex(CORPUS)
        bm25 = {}
        t0 = time.time()
        for position, query in enumerate(queries, 1):
            bm25[query] = index.search_scored(query, DEPTH)
            if position % 300 == 0:
                print(f"  bm25 {position}/{len(queries)} ({time.time() - t0:.0f}s)", flush=True)
        bm25_path.write_text(json.dumps({q: [[a, round(s, 5)] for a, s in v]
                                         for q, v in bm25.items()}), encoding="utf-8")

    # How verbatim are these queries, on overlap.py's own measure?
    corpus_text: dict[str, str] = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            corpus_text[product["parent_asin"]] = " ".join([
                product["title"], " ".join(product["features"]),
                " ".join(product["description"]), product["store"],
                product["details"].get("color", "")])
    coverages = []
    for query, target_set in zip(queries, targets):
        tokens = set(_terms(query))
        if not tokens:
            continue
        best = max((len(tokens & set(_terms(corpus_text.get(a, "")))) / len(tokens))
                   for a in target_set)
        coverages.append(best)
    coverages.sort()
    n = len(coverages)
    print(f"\nquery-token coverage of the target listing (overlap.py's measure):")
    print(f"  ESCI human queries   mean {sum(coverages) / n:.3f}  "
          f"median {coverages[n // 2]:.3f}  p10 {coverages[n // 10]:.3f}")
    print(f"  public-set sessions  mean 0.724  median 0.714  p10 0.600  <- for comparison")

    rows = []
    for model in ("minilm", "bge"):
        vectors, asins = encode_corpus(model)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(MODELS[model]["hf"], device="cpu")
        encoder.max_seq_length = 256
        qvecs = encoder.encode([MODELS[model]["query_prefix"] + q for q in queries],
                               batch_size=128, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        asin_array = np.array(asins)
        dense: dict[str, list[tuple[str, float]]] = {}
        for start in range(0, len(queries), 256):
            block = qvecs[start:start + 256] @ vectors.T
            top = np.argpartition(-block, DEPTH, axis=1)[:, :DEPTH]
            for row in range(block.shape[0]):
                idx = top[row][np.argsort(-block[row, top[row]])]
                dense[queries[start + row]] = list(zip(asin_array[idx].tolist(),
                                                       block[row, idx].tolist()))

        arms: dict[str, list[list[str]]] = {n: [] for n in ("BM25", "dense", "RRF",
                                                           "w0.3", "w0.5")}
        blists, dlists = [], []
        for query in queries:
            blist = [a for a, _ in bm25.get(query, [])]
            dlist = [a for a, _ in dense[query]]
            blists.append(blist)
            dlists.append(dlist)
            arms["BM25"].append(blist)
            arms["dense"].append(dlist)
            fused: dict[str, float] = {}
            for candidates in (blist, dlist):
                for position, asin in enumerate(candidates, 1):
                    fused[asin] = fused.get(asin, 0.0) + 1.0 / (RRF_K + position)
            arms["RRF"].append(sorted(fused, key=lambda a: -fused[a]))
            bn, dn = _minmax(bm25.get(query, [])), _minmax(dense[query])
            for weight, key in ((0.3, "w0.3"), (0.5, "w0.5")):
                combined = {a: weight * dn.get(a, 0.0) + (1 - weight) * bn.get(a, 0.0)
                            for a in set(bn) | set(dn)}
                arms[key].append(sorted(combined, key=lambda a: -combined[a]))

        print(f"\n{'=' * 72}\nESCI real human queries -- encoder: {model}\n{'=' * 72}")
        print(f"{'arm':<18} {'R@10':>8} {'R@50':>8} {'R@100':>8} {'MRR@10':>8}")
        for name, ranked_lists in arms.items():
            ranks = [first_rank(r, t, DEPTH) for r, t in zip(ranked_lists, targets)]
            row = {
                "model": model, "arm": name,
                "recall@10": round(sum(1 for r in ranks if r and r <= 10) / len(ranks), 4),
                "recall@50": round(sum(1 for r in ranks if r and r <= 50) / len(ranks), 4),
                "recall@100": round(sum(1 for r in ranks if r) / len(ranks), 4),
                "mrr@10": round(sum(1.0 / r for r in ranks if r and r <= 10) / len(ranks), 4),
            }
            rows.append(row)
            print(f"{name:<18} {row['recall@10']:>8} {row['recall@50']:>8} "
                  f"{row['recall@100']:>8} {row['mrr@10']:>8}")

        # Union recall -- a set union of the two top-k lists, matching part2_dense.py.
        # NOT a concatenated list: concatenating puts all 100 BM25 ids first, so
        # "union@10" would just be BM25@10 and would print below dense's own recall,
        # which is impossible for a real union. That bug shipped in the first run of
        # this script and is why the row is recomputed here rather than as an "arm".
        union_row = {"model": model, "arm": "union (ceiling)"}
        for depth in (10, 50, 100):
            found = sum(1 for b, d, t in zip(blists, dlists, targets)
                        if (set(b[:depth]) | set(d[:depth])) & t)
            union_row[f"recall@{depth}"] = round(found / len(targets), 4)
        rows.append(union_row)
        print(f"{'union (ceiling)':<18} {union_row['recall@10']:>8} "
              f"{union_row['recall@50']:>8} {union_row['recall@100']:>8} {'--':>8}")

    out = ROOT / "bakeoff" / "results-part5.json"
    out.write_text(json.dumps({
        "source": blob["source"], "filter": blob["filter"], "note": blob["note"],
        "queries": len(entries),
        "token_coverage": {"mean": round(sum(coverages) / n, 4),
                           "median": round(coverages[n // 2], 4),
                           "p10": round(coverages[n // 10], 4)},
        "public_set_token_coverage": {"mean": 0.724, "median": 0.714, "p10": 0.600},
        "arms": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
