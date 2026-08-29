"""Build bi-encoder indexes over the catalog and encode every captured query.

Part 2 of features/retrieval-rerank/bakeoff-prompt.md needs dense Recall@k
alongside BM25 Recall@k. This does the expensive half once: encode 50,000
products and the ~4,000 distinct queries the two captured trajectories contain,
and cache them. part2_dense.py then does pure linear algebra.

Model choice is itself a variable (the prompt says so explicitly), so two are
built rather than one:

  all-MiniLM-L6-v2      22.7M params, 384d -- the prompt's suggestion, and the
                        model behind the r5 round's rejected hybrid arm.
  BAAI/bge-small-en-v1.5  33.4M params, 384d -- a retrieval-tuned model that
                        outranks MiniLM on BEIR. Included so a negative result
                        cannot be dismissed as "you picked a weak encoder".

bge-* wants an asymmetric query prefix; MiniLM does not. Getting that wrong
silently costs several recall points, so it is encoded per-model here rather
than left to the caller.

Document text mirrors the evaluator's own `searchable_text` field set
(local_evaluator.py:22) so BM25 and dense see the same source fields. They do
not see the same *amount*: BM25 indexes all of it, the encoders truncate at 256
word-pieces. That asymmetry is inherent to the comparison and is reported, not
hidden -- see the truncation stats this script prints.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "bakeoff" / "cache"

MODELS = {
    "minilm": {"hf": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": ""},
    "bge": {"hf": "BAAI/bge-small-en-v1.5",
            "query_prefix": "Represent this sentence for searching relevant passages: "},
}

DOC_FIELDS = ("title", "categories", "features", "details", "store", "description")
DOC_CHAR_CAP = 2000  # well past 256 word-pieces; keeps tokenization cheap


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def catalog_documents(catalog_path: Path) -> tuple[list[str], list[str]]:
    asins: list[str] = []
    documents: list[str] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asins.append(str(product["parent_asin"]))
            joined = " ".join(_text(product.get(field)) for field in DOC_FIELDS)
            documents.append(" ".join(joined.split())[:DOC_CHAR_CAP])
    return asins, documents


def all_queries() -> list[str]:
    seen: dict[str, None] = {}
    for ledger in ("current", "legacy"):
        path = ROOT / "bakeoff" / f"trajectories-{ledger}.json"
        if not path.exists():
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            for query in record["queries"]:
                seen.setdefault(query, None)
    return list(seen)


def build(name: str) -> None:
    from sentence_transformers import SentenceTransformer

    spec = MODELS[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    doc_path = CACHE / f"docs-{name}.npy"
    query_path = CACHE / f"queries-{name}.npy"

    asins, documents = catalog_documents(ROOT / "data" / "catalog.jsonl")
    (CACHE / "asins.json").write_text(json.dumps(asins), encoding="utf-8")
    queries = all_queries()
    (CACHE / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    print(f"[{name}] {len(asins)} docs, {len(queries)} distinct queries", flush=True)

    model = SentenceTransformer(spec["hf"], device="cpu")
    model.max_seq_length = 256

    # Sampled, and batched: tokenizing all 50k one at a time costs more than
    # the encode itself.
    probe = documents[::50]
    lengths = [len(ids) for ids in model.tokenizer(probe)["input_ids"]]
    truncated = sum(1 for n in lengths if n > 256)
    print(f"[{name}] docs truncated at 256 word-pieces: {truncated}/{len(probe)} "
          f"({truncated / len(probe):.1%} of a 1-in-50 sample)", flush=True)

    if not doc_path.exists():
        t0 = time.time()
        vectors = model.encode(documents, batch_size=128, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=True)
        np.save(doc_path, vectors.astype(np.float32))
        print(f"[{name}] docs encoded in {time.time() - t0:.0f}s -> {doc_path.name}", flush=True)

    if not query_path.exists():
        t0 = time.time()
        prefixed = [spec["query_prefix"] + q for q in queries]
        vectors = model.encode(prefixed, batch_size=128, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=True)
        np.save(query_path, vectors.astype(np.float32))
        print(f"[{name}] queries encoded in {time.time() - t0:.0f}s -> {query_path.name}",
              flush=True)


if __name__ == "__main__":
    for model_name in (sys.argv[1:] or sorted(MODELS)):
        build(model_name)
