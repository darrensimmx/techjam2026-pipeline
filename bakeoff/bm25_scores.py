"""Capture BM25 *scores* (not just rank order) for every captured query.

Part 3's R4 arm is weighted score fusion, which needs a magnitude on each side,
not an ordering. `starter/retrieval.py` deliberately returns only ids -- that is
the shipped submission surface and is not widened for a measurement -- so this
re-issues the identical FTS5 query with `bm25()` selected alongside.

SQLite's `bm25()` returns a *negative* number, more negative = better match, and
`ORDER BY bm25(...)` is therefore ascending. Scores here are negated so that
larger = better, which is what every fusion formula below assumes. Getting that
sign wrong produces a fusion whose optimum sits at "ignore BM25", which is
exactly the kind of plausible-looking artefact this bake-off is supposed to
catch rather than report.

The column weights are copied verbatim from `starter/retrieval.py:76` so the
scores correspond to the shipped ranking, not a re-tuned one.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from starter.retrieval import Bm25Index, _terms  # noqa: E402

CACHE = ROOT / "bakeoff" / "cache"
DEPTH = 100
WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"  # starter/retrieval.py:76


class ScoringIndex(Bm25Index):
    def search_expression(self, expression: str, top_k: int) -> list[tuple[str, float]]:
        """Run a caller-built FTS5 MATCH expression, same weights and ordering.

        Needed by followup_phrase.py, which builds phrase clauses rather than the
        unigram OR `search_scored` assembles. Kept here so both share one index
        build and one weight vector.
        """
        rows = self._connection.execute(
            f"SELECT parent_asin, bm25(products, {WEIGHTS}) AS s FROM products "
            "WHERE products MATCH ? ORDER BY s LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [(str(asin), -float(raw)) for asin, raw in rows]

    def search_scored(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self._connection.execute(
            f"SELECT parent_asin, bm25(products, {WEIGHTS}) AS s FROM products "
            "WHERE products MATCH ? ORDER BY s LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [(str(asin), -float(raw)) for asin, raw in rows]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    queries = json.loads((CACHE / "queries.json").read_text(encoding="utf-8"))
    index = ScoringIndex(ROOT / "data" / "catalog.jsonl")
    out: dict[str, list] = {}
    t0 = time.time()
    for position, query in enumerate(queries, 1):
        out[query] = [[asin, round(score, 5)] for asin, score in index.search_scored(query, DEPTH)]
        if position % 500 == 0:
            print(f"  {position}/{len(queries)}  ({time.time() - t0:.0f}s)", flush=True)
    path = CACHE / f"bm25-top{DEPTH}-scored.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
