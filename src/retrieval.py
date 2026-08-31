"""BM25 over the local catalog -- the sole retrieval route.  [WS-D OWNS]

Dense fusion was measured twice and REJECTED (-0.206 at top-100, -0.065 at
top-50). The mechanism is understood: the target is BM25's rank 1 in 87 of 176
hit sessions but sits around dense rank 72, so blending drags a good list down
with a bad one. Do not re-propose it without new evidence.

Two-phase on purpose. search() pulls POOL_SIZE ids cheaply; hydrate() pulls the
concatenated text only for the RERANK_WINDOW that survive to reranking. Never
query `WHERE parent_asin = ?` -- parent_asin is UNINDEXED in the FTS5 table and
that is a full scan of ~48k rows. Use rowid.

NEVER RAISES. __init__ is not wrapped by the evaluator (local_evaluator.py:306),
so a throw here kills the whole 200-session run rather than one turn. A missing,
empty, or corrupt catalog therefore builds an EMPTY index and reports it through
is_empty(); it does not propagate. search() and hydrate() degrade to an empty /
unchanged result on any SQLite or type error.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Sequence

from src.types import MAX_QUERY_TERMS, POOL_SIZE, Candidate

CATALOG_FIELDS: tuple[str, ...] = (
    "title", "categories", "features", "details", "store", "description",
)
# The organizer's own weight vector, carried over unchanged. Nothing about
# retrieval has ever been tuned; changing this is a measurement, not an edit.
FIELD_WEIGHTS: tuple[float, ...] = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
})

# The FTS5 table's columns, in order. parent_asin is column 0 and UNINDEXED,
# which is why FIELD_WEIGHTS carries one more entry than CATALOG_FIELDS.
_COLUMNS: tuple[str, ...] = ("parent_asin", *CATALOG_FIELDS)

# unicode61 with remove_diacritics splits on everything that is not a letter or
# a digit, and folds case. Mirroring that here means terms() produces exactly
# the tokens the index holds -- an underscore is a separator, not a word char,
# so `\w` is wrong and `[^\W_]` is right.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_INSERT_BATCH = 1000        # rows per executemany
_HYDRATE_CHUNK = 500        # rowids per IN (...) -- well under SQLITE_MAX_VARIABLE_NUMBER


def terms(text: str) -> list[str]:
    """Lowercased tokens, length > 1, stopwords removed. Never raises."""
    if not isinstance(text, str) or not text:
        return []
    try:
        return [
            token for token in _TOKEN_RE.findall(text.lower())
            if len(token) > 1 and token not in STOPWORDS
        ]
    except Exception:
        return []


def _flatten(value: object) -> str:
    """One catalog field -> one text blob.

    Mirrors the evaluator's own searchable_text() (local_evaluator.py:27): a
    dict flattens to "key value" pairs -- NOT "key: value". That matters
    downstream: intent_card() emits "color: brown" and the overlap gate strips
    the prefix to find "brown" here.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items() if item not in (None, ""))
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def _row_values(product: dict) -> list[str] | None:
    """The seven column values for one catalog row, or None if unusable."""
    parent_asin = product.get("parent_asin")
    if parent_asin in (None, ""):
        return None
    row = [str(parent_asin)]
    for field in CATALOG_FIELDS:
        try:
            row.append(_flatten(product.get(field)))
        except Exception:
            row.append("")
    return row


def _weights_sql() -> str:
    """The bm25() weight arguments, one per column, as SQL literals.

    Built from FIELD_WEIGHTS but padded/truncated to the real column count: a
    length mismatch is an OperationalError from SQLite on every single query,
    which would look exactly like "retrieval is broken" with no traceback.
    """
    values = [float(weight) for weight in FIELD_WEIGHTS[:len(_COLUMNS)]]
    values.extend(1.0 for _ in range(len(_COLUMNS) - len(values)))
    return ", ".join(repr(value) for value in values)


class Bm25Index:
    def __init__(self, catalog_path: str | Path) -> None:
        self._size = 0
        self._conn: sqlite3.Connection | None = None
        self._weights = _weights_sql()
        try:
            self._build(catalog_path)
        except Exception:
            # A half-built index is worse than none: the caller checks
            # is_empty() and degrades gracefully, but cannot detect a partial.
            self._drop()

    # -- construction -------------------------------------------------------

    def _drop(self) -> None:
        conn, self._conn, self._size = self._conn, None, 0
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    def _build(self, catalog_path: str | Path) -> None:
        if catalog_path is None:
            return
        try:
            path = Path(catalog_path)
        except Exception:
            return
        if not path.is_file():
            return

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        columns = ", ".join(
            f"{name} UNINDEXED" if name == "parent_asin" else name for name in _COLUMNS
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE products USING fts5("
            f"{columns}, tokenize='unicode61 remove_diacritics 2')"
        )
        placeholders = ", ".join("?" for _ in _COLUMNS)
        statement = f"INSERT INTO products ({', '.join(_COLUMNS)}) VALUES ({placeholders})"

        size = 0
        batch: list[list[str]] = []
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    product = json.loads(line)
                except Exception:
                    continue  # skip the malformed line, keep the catalog
                if not isinstance(product, dict):
                    continue
                row = _row_values(product)
                if row is None:
                    continue
                batch.append(row)
                if len(batch) >= _INSERT_BATCH:
                    conn.executemany(statement, batch)
                    size += len(batch)
                    batch.clear()
        if batch:
            conn.executemany(statement, batch)
            size += len(batch)
        conn.commit()

        self._conn = conn
        self._size = size

    # -- public surface -----------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def close(self) -> None:
        """Release the in-memory database. Never raises.

        The graded path never calls this -- one index lives for the whole run.
        It exists because a test suite builds dozens of indexes, and an
        unclosed sqlite3.Connection is a ResourceWarning each time.
        """
        self._drop()

    def __del__(self) -> None:  # pragma: no cover -- interpreter teardown
        try:
            self._drop()
        except Exception:
            pass

    def search(self, query_text: str, limit: int) -> list[Candidate]:
        """The candidate pool -- the recall floor. `text` is "" on every result.

        Must survive FTS5 metacharacters in the query: a customer reply can
        legitimately contain quotes, and NEAR/AND/OR/NOT are MATCH operators.
        Every term is emitted as a quoted PHRASE, so an operator word is matched
        as a word and a stray quote cannot break out of its phrase.
        """
        if self._conn is None or self._size == 0:
            return []
        limit = _clamp_limit(limit)
        if limit == 0:
            return []
        match = self._match_expression(query_text)
        if not match:
            return []
        try:
            rows = self._conn.execute(
                f"SELECT rowid, parent_asin, bm25(products, {self._weights}) AS score "
                f"FROM products WHERE products MATCH ? "
                f"ORDER BY score LIMIT ?",
                (match, limit),
            ).fetchall()
        except Exception:
            return []
        return [
            Candidate(
                parent_asin=str(row[1]),
                rowid=int(row[0]),
                rank=position,
                score=float(row[2]) if row[2] is not None else 0.0,
                text="",
            )
            for position, row in enumerate(rows, start=1)
        ]

    def hydrate(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Fill `.text` for these candidates via rowid. Order preserved.

        rowid, never `WHERE parent_asin = ?`: parent_asin is UNINDEXED, so
        matching on it is a full scan of the whole catalog per candidate.
        """
        try:
            items = list(candidates)
        except Exception:
            return []
        if self._conn is None or not items:
            return items

        wanted: list[int] = []
        seen: set[int] = set()
        for candidate in items:
            rowid = getattr(candidate, "rowid", 0)
            if isinstance(rowid, int) and rowid > 0 and rowid not in seen:
                seen.add(rowid)
                wanted.append(rowid)
        if not wanted:
            return items

        texts: dict[int, str] = {}
        selection = ", ".join(CATALOG_FIELDS)
        try:
            for start in range(0, len(wanted), _HYDRATE_CHUNK):
                chunk = wanted[start:start + _HYDRATE_CHUNK]
                placeholders = ", ".join("?" for _ in chunk)
                for row in self._conn.execute(
                    f"SELECT rowid, {selection} FROM products WHERE rowid IN ({placeholders})",
                    chunk,
                ):
                    texts[int(row[0])] = " ".join(str(value) for value in row[1:] if value)
        except Exception:
            return items

        hydrated: list[Candidate] = []
        for candidate in items:
            try:
                text = texts.get(getattr(candidate, "rowid", 0))
                if text is None or getattr(candidate, "text", ""):
                    hydrated.append(candidate)
                else:
                    hydrated.append(
                        Candidate(
                            parent_asin=candidate.parent_asin,
                            rowid=candidate.rowid,
                            rank=candidate.rank,
                            score=candidate.score,
                            text=text,
                        )
                    )
            except Exception:
                hydrated.append(candidate)
        return hydrated

    # -- internals ----------------------------------------------------------

    def _match_expression(self, query_text: str) -> str:
        """`"term" OR "term" ...` over <= MAX_QUERY_TERMS unique terms.

        Deduped preserving order so the cap keeps the FIRST 40 distinct terms --
        the ledger front-loads the customer's own words, and truncating the tail
        loses less than truncating the head.
        """
        unique: list[str] = []
        seen: set[str] = set()
        for token in terms(query_text):
            if token in seen:
                continue
            seen.add(token)
            unique.append(token)
            if len(unique) >= MAX_QUERY_TERMS:
                break
        if not unique:
            return ""
        # Doubling an embedded quote is belt-and-braces: _TOKEN_RE cannot emit
        # one. It is here so that a future change to tokenisation cannot turn
        # a customer's apostrophe into an FTS5 syntax error.
        return " OR ".join('"' + token.replace('"', '""') + '"' for token in unique)


def _clamp_limit(limit: object) -> int:
    """A LIMIT SQLite will accept. Negative means "no limit" to SQLite, which
    would hand back the whole catalog and blow maxItems; a non-int raises."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        return POOL_SIZE
    return max(0, limit)
