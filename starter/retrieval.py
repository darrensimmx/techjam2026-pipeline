"""BM25 spine over the product catalog — local, offline, no network calls.

Index is built once at construction (analogous to precomputing embeddings at
startup): retrieval itself never touches the network or does per-turn setup.
Adapted from the competition starter's SQLite FTS5 approach.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
_CATALOG_FIELDS = ("title", "categories", "features", "details", "store", "description")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


class Bm25Index:
    def __init__(self, catalog_path: str | Path) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._build(Path(catalog_path))

    def _build(self, catalog_path: Path) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        placeholders = ", ".join(["?"] * (1 + len(_CATALOG_FIELDS)))
        batch: list[tuple[str, ...]] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                row = (str(product["parent_asin"]), *(_text(product.get(field)) for field in _CATALOG_FIELDS))
                batch.append(row)
                if len(batch) >= 1000:
                    cursor.executemany(f"INSERT INTO products VALUES ({placeholders})", batch)
                    batch.clear()
        if batch:
            cursor.executemany(f"INSERT INTO products VALUES ({placeholders})", batch)
        self._connection.commit()

    def search(self, query_text: str, top_k: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self._connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [str(row[0]) for row in rows]
