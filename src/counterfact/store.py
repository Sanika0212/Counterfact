"""SQLite storage with content-addressed, de-duplicated payloads.

Payloads are stored once per distinct content (SHA-256 addressed, zlib-compressed).
Chat requests resend the whole conversation on every turn, so lists of messages are
stored as hash chains: each node holds one message and a pointer to the previous
node. Turn ``t`` then adds only its new messages, and a 30-turn conversation stores
each message once instead of up to 30 times, with constant-size references.

Writes happen in one transaction per run, or in a background thread when
streaming spans, never on the agent's hot path.
"""

from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from counterfact.jsonable import canonical, to_jsonable
from counterfact.model import Boundary, Run, Span, SpanKind
from counterfact.redact import Redactor

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY, data BLOB NOT NULL, raw_size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT NOT NULL, seed INTEGER NOT NULL,
    started_ns INTEGER NOT NULL, ended_ns INTEGER, status TEXT NOT NULL, error TEXT,
    passed INTEGER, parent_run_id TEXT, meta TEXT NOT NULL, outcome_ref TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL,
    start_ns INTEGER NOT NULL, end_ns INTEGER, status TEXT NOT NULL, error TEXT,
    attributes TEXT NOT NULL, input_ref TEXT, output_ref TEXT,
    b_index INTEGER, b_key TEXT, b_fingerprint TEXT, b_seed INTEGER, b_source TEXT
);
CREATE INDEX IF NOT EXISTS spans_by_run ON spans(run_id, start_ns);
CREATE INDEX IF NOT EXISTS runs_by_parent ON runs(parent_run_id);
CREATE INDEX IF NOT EXISTS runs_by_start ON runs(started_ns DESC);
"""

_SPLIT_KEYS = ("messages", "input", "contents")
_CHAIN_MARK = "__cf_chain__"


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    name: str
    status: str
    passed: bool | None
    started_ns: int
    parent_run_id: str | None
    spans: int


class Store:
    def __init__(self, path: str | Path = ":memory:", *, redactor: Redactor | None = None) -> None:
        self.path = str(path)
        self.redactor = redactor if redactor is not None else Redactor()
        self._local = threading.local()
        self._write_lock = threading.Lock()
        if self.path == ":memory:":
            self._shared: sqlite3.Connection | None = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
        else:
            self._shared = None
        self._conn().executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # ---------------------------------------------------------------- blobs
    def _put_raw(self, conn: sqlite3.Connection, value: Any) -> str:
        text = canonical(value)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO blobs(hash, data, raw_size) VALUES (?, ?, ?)",
            (digest, zlib.compress(text.encode("utf-8"), 6), len(text)),
        )
        return digest

    def _put(self, conn: sqlite3.Connection, value: Any) -> str:
        return self._put_raw(conn, self._split(conn, to_jsonable(value)))

    def _split(self, conn: sqlite3.Connection, value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key in _SPLIT_KEYS and isinstance(item, list) and len(item) > 1:
                    out[key] = {_CHAIN_MARK: self._put_chain(conn, item)}
                else:
                    out[key] = self._split(conn, item)
            return out
        if isinstance(value, list):
            return [self._split(conn, item) for item in value]
        return value

    def _put_chain(self, conn: sqlite3.Connection, items: list[Any]) -> str:
        node: str | None = None
        for item in items:
            node = self._put_raw(conn, {"p": node, "m": self._put_raw(conn, item)})
        assert node is not None
        return node

    def _get_chain(self, node: str | None) -> list[Any]:
        items: list[Any] = []
        while node is not None:
            link = self._get_raw(node)
            items.append(self._get_raw(link["m"]))
            node = link["p"]
        return items[::-1]

    def _get_raw(self, digest: str) -> Any:
        row = self._conn().execute("SELECT data FROM blobs WHERE hash = ?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        return json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def _join(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {_CHAIN_MARK}:
                return self._get_chain(value[_CHAIN_MARK])
            return {k: self._join(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._join(v) for v in value]
        return value

    def get(self, digest: str | None) -> Any:
        return None if digest is None else self._join(self._get_raw(digest))

    # ----------------------------------------------------------------- runs
    def _write_span(self, conn: sqlite3.Connection, span: Span) -> None:
        b = span.boundary
        conn.execute(
            "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                span.span_id,
                span.run_id,
                span.parent_id,
                span.kind.value,
                span.name,
                span.start_ns,
                span.end_ns,
                span.status,
                self.redactor(span.error),
                canonical(self.redactor(to_jsonable(span.attributes))),
                self._put(conn, self.redactor(to_jsonable(span.input))),
                self._put(conn, self.redactor(to_jsonable(span.output))),
                b.index if b else None,
                b.key if b else None,
                b.fingerprint if b else None,
                b.seed if b else None,
                b.source if b else None,
            ),
        )

    def save_run(self, run: Run) -> None:
        with self._write_lock:
            conn = self._conn()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run.run_id,
                        run.name,
                        run.seed,
                        run.started_ns,
                        run.ended_ns,
                        run.status,
                        self.redactor(run.error),
                        None if run.passed is None else int(run.passed),
                        run.parent_run_id,
                        canonical(self.redactor(to_jsonable(run.meta))),
                        self._put(conn, self.redactor(to_jsonable(run.outcome))),
                    ),
                )
                for span in run.spans:
                    self._write_span(conn, span)

    def load_run(self, run_id: str) -> Run:
        conn = self._conn()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        spans = []
        for s in conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY start_ns", (run_id,)
        ).fetchall():
            boundary = (
                Boundary(index=s[12], key=s[13], fingerprint=s[14], seed=s[15], source=s[16])
                if s[13] is not None
                else None
            )
            spans.append(
                Span(
                    span_id=s[0],
                    run_id=s[1],
                    parent_id=s[2],
                    kind=SpanKind(s[3]),
                    name=s[4],
                    start_ns=s[5],
                    end_ns=s[6],
                    status=s[7],
                    error=s[8],
                    attributes=json.loads(s[9]),
                    input=self.get(s[10]),
                    output=self.get(s[11]),
                    boundary=boundary,
                )
            )
        return Run(
            run_id=row[0],
            name=row[1],
            seed=row[2],
            started_ns=row[3],
            ended_ns=row[4],
            status=row[5],
            error=row[6],
            passed=None if row[7] is None else bool(row[7]),
            parent_run_id=row[8],
            meta=json.loads(row[9]),
            outcome=self.get(row[10]),
            spans=spans,
        )

    def list_runs(self, limit: int = 50, *, parent: str | None = None) -> list[RunSummary]:
        sql = (
            "SELECT r.run_id, r.name, r.status, r.passed, r.started_ns, r.parent_run_id,"
            " (SELECT COUNT(*) FROM spans s WHERE s.run_id = r.run_id) FROM runs r"
        )
        args: tuple[Any, ...] = ()
        if parent is not None:
            sql += " WHERE r.parent_run_id = ?"
            args = (parent,)
        sql += " ORDER BY r.started_ns DESC LIMIT ?"
        rows = self._conn().execute(sql, (*args, min(limit, 10_000))).fetchall()
        return [
            RunSummary(r[0], r[1], r[2], None if r[3] is None else bool(r[3]), r[4], r[5], r[6])
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        conn = self._conn()
        blobs, stored, raw = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(data)),0), COALESCE(SUM(raw_size),0) FROM blobs"
        ).fetchone()
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        return {
            "runs": runs,
            "spans": spans,
            "blobs": blobs,
            "stored_bytes": stored,
            "unique_raw_bytes": raw,
        }

    # -------------------------------------------------------------- streaming
    def span_writer(self) -> tuple[Callable[[Span], None], Callable[[], None]]:
        """``(on_span, close)``: write spans from a background thread as they finish."""
        pending: queue.Queue[Span | None] = queue.Queue()

        def worker() -> None:
            while (item := pending.get()) is not None:
                with self._write_lock:
                    conn = self._conn()
                    with conn:
                        self._write_span(conn, item)

        thread = threading.Thread(target=worker, name="counterfact-writer", daemon=True)
        thread.start()

        def close() -> None:
            pending.put(None)
            thread.join()

        return pending.put, close
