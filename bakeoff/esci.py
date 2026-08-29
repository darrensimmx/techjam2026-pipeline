"""Build a real-human-query retrieval benchmark, in our own catalog's schema.

Why this exists
---------------
Part 5 of the bake-off asks whether an arm's wins concentrate on sessions where
the disclosed strings are NOT verbatim substrings of the target listing. The
public set can barely answer it: `overlap.py` measures 94.5% of the simulator's
disclosed constraint strings as verbatim substrings of the target's own listing,
because `local_evaluator.py:52-71` builds them by copying out of it. So "BM25
wins" could be a fact about retrieval, or just a fact about a simulator that
hands BM25 the answer key -- and the public set cannot separate those two.

Writing our own paraphrases would not separate them either. We would be
inventing the distribution that decides the answer and then reporting what we
invented.

So use real human queries. The Shopping Queries Dataset (ESCI):

    Reddy, Marquez, Valero, Rao, Zhang, Sanz, Nag, Nagaraj, Karim, Rowe, Nio,
    Zhu. "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
    Product Search." arXiv:2206.06588 (2022).
    https://github.com/amazon-science/esci-data  -- Apache-2.0.

130k real Amazon customer queries, 2.6M human relevance judgements (Exact /
Substitute / Complement / Irrelevant) over real ASINs.

First attempt, and why it was abandoned
---------------------------------------
The plan was to run ESCI queries against OUR index, by intersecting ESCI's
`product_id` with our catalog's `parent_asin`. Measured: 71 shared ASINs out of
599,151 ESCI products x 50,000 of ours, and **zero** survived the
(locale == us AND esci_label == E) filter. The two are near-disjoint samples of
Amazon. That route is dead; it is recorded here so nobody spends the download
again.

What this builds instead
------------------------
A standalone benchmark from ESCI's own product fields, written in **our
catalog's exact JSONL schema** so `starter/retrieval.py`'s `Bm25Index` indexes
it unmodified -- same FTS5 tokenizer, same column weights -- and `dense.py`
encodes it through the same path. The corpus is 20,000 documents. Recall@k depends on corpus size, so these
numbers are NOT comparable to the public-set recalls in Part 2 -- only the
BM25-vs-dense ordering *within* this benchmark is. 20k rather than our 50k
purely because every document has to be encoded twice on CPU.

Field mapping (ESCI has no category field; that column is left empty):
    parent_asin <- product_id        features <- product_bullet_point
    title       <- product_title     store    <- product_brand
    description <- product_description   details <- {"color": product_color}
"""
from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "bakeoff" / "cache"
HF_GLOB = str(Path.home() / ".cache" / "huggingface" / "hub"
              / "datasets--tasksource--esci" / "snapshots" / "*" / "data" / "*.parquet")

CORPUS_SIZE = 20_000   # a real haystack; smaller than our 50k only for CPU budget
N_QUERIES = 600      # keeps targets a minority of the corpus (~23%)
SEED = 20260829


def main() -> None:
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(HF_GLOB))
    if not shards:
        raise SystemExit("no ESCI shards cached; see bakeoff/README.md")
    print(f"{len(shards)} ESCI shards on disk")

    columns = ["query", "product_id", "product_locale", "esci_label", "product_title",
               "product_description", "product_bullet_point", "product_brand",
               "product_color"]
    frames = []
    for path in shards:
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame[frame["product_locale"] == "us"])
        print(f"  {Path(path).name}: {len(frames[-1])} us rows", flush=True)
    data = pd.concat(frames, ignore_index=True)
    del frames
    print(f"us judgements: {len(data)}, distinct queries: {data['query'].nunique()}, "
          f"distinct products: {data['product_id'].nunique()}")

    exact = data[data["esci_label"] == "Exact"]  # this mirror spells labels out
    per_query = exact.groupby("query")["product_id"].apply(lambda s: sorted(set(s)))
    eligible = sorted(per_query.index)
    print(f"queries with >=1 Exact-labelled product: {len(eligible)}")

    rng = random.Random(SEED)
    chosen = sorted(rng.sample(eligible, min(N_QUERIES, len(eligible))))
    targets = {q: per_query[q] for q in chosen}
    must_keep = {asin for asins in targets.values() for asin in asins}
    print(f"sampled {len(chosen)} queries, {len(must_keep)} distinct target products")

    # Corpus: every target, then distractors up to CORPUS_SIZE, drawn from the
    # same pool so the index is a realistic haystack rather than only answers.
    catalogue = data.drop_duplicates(subset="product_id").set_index("product_id")
    pool = [a for a in catalogue.index.tolist() if a not in must_keep]
    rng.shuffle(pool)
    corpus_ids = sorted(must_keep) + pool[:max(0, CORPUS_SIZE - len(must_keep))]
    print(f"corpus: {len(corpus_ids)} products "
          f"({len(must_keep)} targets + {len(corpus_ids) - len(must_keep)} distractors)")

    def clean(value: object) -> str:
        return "" if value is None or (isinstance(value, float)) else str(value)

    corpus_path = CACHE / "esci_catalog.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for asin in corpus_ids:
            row = catalogue.loc[asin]
            handle.write(json.dumps({
                "parent_asin": asin,
                "title": clean(row["product_title"]),
                "categories": [],
                "features": [clean(row["product_bullet_point"])],
                "details": {"color": clean(row["product_color"])},
                "store": clean(row["product_brand"]),
                "description": [clean(row["product_description"])],
            }) + "\n")

    queries_path = CACHE / "esci_queries.json"
    queries_path.write_text(json.dumps({
        "source": "Reddy et al. 2022, arXiv:2206.06588; HF tasksource/esci",
        "filter": f"product_locale == 'us'; targets are esci_label == 'Exact'; "
                  f"seed {SEED}; corpus {len(corpus_ids)} products",
        "note": "ESCI product_ids and our catalog's parent_asins are near-disjoint "
                "(71 shared of 599,151 x 50,000, 0 after the us+Exact filter), so this "
                "is a standalone corpus in our catalog's schema, not our catalog.",
        "queries": [{"query": q, "targets": targets[q]} for q in chosen],
    }, indent=2), encoding="utf-8")

    print(f"wrote {corpus_path} ({corpus_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {queries_path}")


if __name__ == "__main__":
    main()
