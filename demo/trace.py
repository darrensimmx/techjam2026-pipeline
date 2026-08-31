"""The trace event stream: schema constants, writer, and tailing reader.

The two CLIs are coupled by an append-only JSON Lines file under
``demo/runs/``. That was chosen over a socket or a FIFO for four reasons
specific to this demo:

  - A file gives sequential replay for free. One renderer serves both live
    ``--follow`` and post-hoc ``--replay``.
  - ``os.mkfifo`` does not exist on Windows, and CLAUDE.md makes Windows a
    first-class dev platform.
  - The backend may lag arbitrarily -- or attach late, or step turn by turn --
    without applying backpressure to the frontend. Over a socket a stepping
    reader would stall the agent mid-turn and the demo would look broken.
  - Start order does not matter. A late backend replays from ``seq`` 0.

The writer emits one ``write()`` per record and flushes; the reader parses only
up to the last newline it can see and buffers any trailing fragment. Those two
halves are what make a partially-written line unobservable rather than fatal.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_RUN_DIR = Path("demo") / "runs"
POINTER_NAME = "latest.json"

# -- record types ----------------------------------------------------------

RUN_OPEN = "run_open"
SESSION_OPEN = "session_open"
TURN = "turn"
SESSION_CLOSE = "session_close"
RUN_CLOSE = "run_close"
NOTE = "note"

RECORD_TYPES = (RUN_OPEN, SESSION_OPEN, TURN, SESSION_CLOSE, RUN_CLOSE, NOTE)

# -- note codes ------------------------------------------------------------

DEGRADED_AGENT = "degraded_agent"
POOL_EMPTY = "pool_empty"
SMALL_CATALOG = "small_catalog"
PATCH_TARGET_MISSING = "patch_target_missing"
RUNG_MISMATCH = "rung_mismatch"
SPLIT_INCONSISTENT = "split_inconsistent"
GUARD_REJECTED = "guard_rejected"
DEGRADED_PLAN_FIRED = "degraded_plan_fired"
TRACER_RECORD_ERROR = "tracer_record_error"
EVALUATOR_PATCH_LEFT_INSTALLED = "evaluator_patch_left_installed"

NOTE_CODES = (
    DEGRADED_AGENT, POOL_EMPTY, SMALL_CATALOG, PATCH_TARGET_MISSING,
    RUNG_MISMATCH, SPLIT_INCONSISTENT, GUARD_REJECTED, DEGRADED_PLAN_FIRED,
    TRACER_RECORD_ERROR, EVALUATOR_PATCH_LEFT_INSTALLED,
)

# -- brackets --------------------------------------------------------------

LEAKY = "leaky"
SCRUBBED = "scrubbed"

CARD_SOURCE = {
    LEAKY: "leaky-vendored-intent_card",
    SCRUBBED: "scrubbed-intent_card_scrubbed",
}

BRACKET_NOTE = {
    LEAKY: (
        "UPPER BOUND -- the simulated customer's hidden card is built from the "
        "target product's own listing, so it recites text that is already "
        "indexed. Not a score."
    ),
    SCRUBBED: (
        "LOWER BOUND -- the hidden card discloses only atomic attribute values, "
        "never a multi-word span lifted from the target's listing."
    ),
}

SCORE_FORMULA = "0.50*hit@10 + 0.30*mrr + 0.20*(11-mttc)/10"

# A catalog smaller than this is the tests/fixtures trap: 6 products against
# top_k=10 means any single-term match returns everything, so a query-blind
# ranker looks perfect (CLAUDE.md, "A green test run proves less than it looks
# like").
SMALL_CATALOG_ROWS = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


class TraceWriter:
    """Appends records to a JSONL file and maintains the ``latest.json`` pointer.

    Every record is stamped with ``v``/``type``/``ts``/``run_id``/``seq`` here,
    so no caller can forget one and no two records can share a ``seq``.
    """

    def __init__(self, run_dir: object = DEFAULT_RUN_DIR, run_id: str = "") -> None:
        self.run_id = run_id or new_run_id()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / (self.run_id + ".jsonl")
        self.seq = 0
        self.errors = 0
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        self._write_pointer()

    # -- pointer ----------------------------------------------------------

    def _write_pointer(self) -> None:
        """Publish the current run atomically, so a reader never sees a stub.

        Written to a temp name in the same directory then ``os.replace``d --
        atomic on POSIX and on same-volume Windows.
        """
        pointer = self.run_dir / POINTER_NAME
        tmp = self.run_dir / (POINTER_NAME + ".tmp-" + uuid.uuid4().hex[:8])
        payload = {
            "run_id": self.run_id,
            "path": str(self.path),
            "started": utc_now(),
            "pid": os.getpid(),
        }
        try:
            tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            os.replace(str(tmp), str(pointer))
        except Exception:
            self.errors += 1
            try:
                tmp.unlink()
            except Exception:
                pass

    # -- records ----------------------------------------------------------

    def emit(self, record_type: str, **fields: object) -> dict:
        """Stamp and append one record. Returns what was written."""
        record: dict = {
            "v": SCHEMA_VERSION,
            "type": record_type,
            "ts": utc_now(),
            "run_id": self.run_id,
            "seq": self.seq,
        }
        record.update(fields)
        self.seq += 1
        try:
            line = json.dumps(record, ensure_ascii=False, default=_fallback) + "\n"
        except Exception:
            self.errors += 1
            line = json.dumps({
                "v": SCHEMA_VERSION, "type": NOTE, "ts": utc_now(),
                "run_id": self.run_id, "seq": record["seq"], "level": "error",
                "code": TRACER_RECORD_ERROR,
                "text": "record of type %r could not be serialised" % (record_type,),
            }) + "\n"
        try:
            self._fh.write(line)
            self._fh.flush()
        except Exception:
            self.errors += 1
        return record

    def note(self, level: str, code: str, text: str) -> dict:
        return self.emit(NOTE, level=level, code=code, text=text)

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except Exception:
            return 0

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False


def _fallback(value: object) -> object:
    """Last-resort JSON coercion, so one odd value cannot cost a whole record."""
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class TraceReader:
    """Tails a JSONL trace by byte offset, tolerating partial trailing lines.

    ``malformed`` counts lines that failed to parse. They are skipped and
    surfaced in the backend header rather than killing the render -- a demo
    that dies on one bad line is worse than one that says "1 malformed".
    """

    def __init__(self, path: object) -> None:
        self.path = Path(path)
        self.offset = 0
        self.malformed = 0
        self._buffer = ""

    def exists(self) -> bool:
        try:
            return self.path.is_file()
        except Exception:
            return False

    def read_new(self) -> list[dict]:
        """Return every complete record appended since the last call."""
        if not self.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
                self.offset = fh.tell()
        except Exception:
            return []

        if not chunk:
            return []

        self._buffer += chunk
        cut = self._buffer.rfind("\n")
        if cut == -1:
            # Nothing complete yet -- hold the fragment for the next poll.
            return []
        complete, self._buffer = self._buffer[:cut], self._buffer[cut + 1:]

        records: list[dict] = []
        for line in complete.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                self.malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                self.malformed += 1
        return records

    def read_all(self) -> list[dict]:
        """Drain the whole file. Used by ``--replay`` and by the tests."""
        self.offset = 0
        self._buffer = ""
        return self.read_new()

    def follow(self, poll: float = 0.05, idle_timeout: float = 0.0):
        """Yield records as they land. Stops after ``run_close``.

        ``idle_timeout`` of 0 means wait forever, which is what a demo wants:
        the backend is normally started first and should sit patiently.
        """
        last_seen = time.monotonic()
        while True:
            batch = self.read_new()
            if batch:
                last_seen = time.monotonic()
                for record in batch:
                    yield record
                    if record.get("type") == RUN_CLOSE:
                        return
            elif idle_timeout and (time.monotonic() - last_seen) > idle_timeout:
                return
            else:
                time.sleep(poll)


def discover_run(run_dir: object = DEFAULT_RUN_DIR, explicit: object = None) -> "Path | None":
    """Find the trace to render: explicit path, then pointer, then newest file."""
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None

    directory = Path(run_dir)
    pointer = directory / POINTER_NAME
    try:
        if pointer.is_file():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            candidate = Path(str(payload.get("path", "")))
            if candidate.is_file():
                return candidate
    except Exception:
        pass

    try:
        runs = sorted(
            (p for p in directory.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        return runs[-1] if runs else None
    except Exception:
        return None
