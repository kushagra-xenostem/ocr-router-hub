import json
import time
import uuid
import sqlite3
import threading
import numpy as np
from .pii import redact
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL, mode TEXT NOT NULL,
    doc_type TEXT, type_confidence REAL, type_method TEXT, status TEXT NOT NULL,
    n_pages INTEGER DEFAULT 0, quality REAL, routes TEXT, processing_ms INTEGER,
    batch_id TEXT, error TEXT, created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sha ON documents(sha256, mode);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, page_no INTEGER NOT NULL,
    route TEXT NOT NULL, text TEXT NOT NULL, ocr_conf REAL, lines TEXT, image_path TEXT,
    width INTEGER, height INTEGER, PRIMARY KEY (doc_id, page_no)
);
CREATE TABLE IF NOT EXISTS fields (
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, name TEXT NOT NULL,
    value TEXT, confidence REAL, method TEXT, grounded INTEGER, page INTEGER, snippet TEXT, bbox TEXT,
    valid INTEGER, issue TEXT, pii_kind TEXT, corrected_value TEXT, position INTEGER,
    PRIMARY KEY (doc_id, name)
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, doc_type TEXT, field TEXT NOT NULL,
    original TEXT, corrected TEXT, confidence REAL, snippet TEXT, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no INTEGER NOT NULL, position INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, batch_id TEXT, doc_id TEXT, filename TEXT NOT NULL, status TEXT NOT NULL,
    stage TEXT, progress REAL DEFAULT 0, eta_s REAL, error TEXT, options TEXT, cache_hit INTEGER DEFAULT 0,
    created REAL NOT NULL, started REAL, finished REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);
CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, ts REAL NOT NULL,
    level TEXT NOT NULL, msg TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_job ON job_logs(job_id, id);
CREATE TABLE IF NOT EXISTS qa_messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
    sources TEXT, meta TEXT, created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_messages(session_id, created);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
    doc_id TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS ocr_cache (hash TEXT PRIMARY KEY, result TEXT NOT NULL, created REAL NOT NULL);
"""


def _j(v):
    return None if v is None else json.dumps(v)


def _blob(vec) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        current = self._db.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self._db.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        if has_tables and current != SCHEMA_VERSION:
            raise RuntimeError(f"Database {path} uses another schema version; delete it to start fresh.")
        self._db.executescript(SCHEMA)
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._lock = threading.RLock()
        self._chunk_version = 0

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args)]

    def _one(self, sql: str, args=()) -> dict | None:
        with self._lock:
            r = self._db.execute(sql, args).fetchone()
        return dict(r) if r else None

    def _exec(self, sql: str, args=()) -> int:
        with self._lock, self._db:
            return self._db.execute(sql, args).rowcount

    # ---------- jobs ----------
    def create_job(self, filename: str, options: dict, batch_id: str | None = None) -> str:
        jid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO jobs (id, batch_id, filename, status, stage, options, created) VALUES (?,?,?,?,?,?,?)",
                   (jid, batch_id, filename, "queued", "Queued", _j(options), time.time()))
        return jid

    def update_job(self, job_id: str, **kw) -> None:
        if not kw:
            return
        cols = ", ".join(f"{k}=?" for k in kw)
        self._exec(f"UPDATE jobs SET {cols} WHERE id=?", (*kw.values(), job_id))

    def get_job(self, job_id: str) -> dict | None:
        j = self._one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if j:
            j["options"] = json.loads(j["options"]) if j["options"] else {}
        return j

    def list_jobs(self, batch_id: str | None = None, limit: int = 200) -> list[dict]:
        if batch_id:
            return self._rows("SELECT * FROM jobs WHERE batch_id=? ORDER BY created", (batch_id,))
        return self._rows("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))

    def log(self, job_id: str, level: str, msg: str) -> None:
        self._exec("INSERT INTO job_logs (job_id, ts, level, msg) VALUES (?,?,?,?)",
                   (job_id, time.time(), level, redact(msg)))

    def logs(self, job_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
        return self._rows("SELECT * FROM job_logs WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
                          (job_id, after_id, limit))

    def fail_interrupted_jobs(self) -> int:
        return self._exec("UPDATE jobs SET status='failed', error='Interrupted by restart', finished=? "
                          "WHERE status IN ('queued','running')", (time.time(),))

    # ---------- documents ----------
    def create_document(self, filename: str, sha256: str, mode: str, batch_id: str | None) -> str:
        did = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO documents (id, filename, sha256, mode, status, batch_id, created) "
                   "VALUES (?,?,?,?,?,?,?)", (did, filename, sha256, mode, "processing", batch_id, time.time()))
        return did

    def update_document(self, doc_id: str, **kw) -> None:
        if "routes" in kw:
            kw["routes"] = _j(kw["routes"])
        cols = ", ".join(f"{k}=?" for k in kw)
        self._exec(f"UPDATE documents SET {cols} WHERE id=?", (*kw.values(), doc_id))

    def find_processed(self, sha256: str, mode: str) -> dict | None:
        return self._one("SELECT * FROM documents WHERE sha256=? AND mode=? AND status='done' "
                         "ORDER BY created DESC LIMIT 1", (sha256, mode))

    def get_document(self, doc_id: str) -> dict | None:
        d = self._one("SELECT * FROM documents WHERE id=?", (doc_id,))
        if d:
            d["routes"] = json.loads(d["routes"]) if d["routes"] else []
        return d

    def list_documents(self, doc_type: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM documents", ()
        if doc_type:
            sql, args = sql + " WHERE doc_type=?", (doc_type,)
        out = self._rows(sql + " ORDER BY created DESC", args)
        for d in out:
            d["routes"] = json.loads(d["routes"]) if d["routes"] else []
        return out

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._db:
            n = self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
            self._db.execute("DELETE FROM corrections WHERE doc_id=?", (doc_id,))
            if n:
                self._chunk_version += 1
        return n > 0

    def save_pages(self, doc_id: str, pages: list[dict]) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
            self._db.executemany(
                "INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)",
                [(doc_id, p["page_no"], p["route"], p["text"], p.get("ocr_conf"), _j(p.get("lines")),
                  p.get("image_path"), p.get("width"), p.get("height")) for p in pages],
            )

    def get_pages(self, doc_id: str) -> list[dict]:
        out = self._rows("SELECT * FROM pages WHERE doc_id=? ORDER BY page_no", (doc_id,))
        for p in out:
            p["lines"] = json.loads(p["lines"]) if p["lines"] else []
        return out

    def save_fields(self, doc_id: str, fields: list[dict]) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM fields WHERE doc_id=?", (doc_id,))
            self._db.executemany(
                "INSERT INTO fields VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(doc_id, f["name"], f.get("value"), f.get("confidence"), f.get("method"),
                  int(bool(f.get("grounded"))), f.get("page"), f.get("snippet"), _j(f.get("bbox")),
                  int(bool(f.get("valid", True))), f.get("issue"), f.get("pii_kind"), None, i)
                 for i, f in enumerate(fields)],
            )

    def get_fields(self, doc_id: str) -> list[dict]:
        out = self._rows("SELECT * FROM fields WHERE doc_id=? ORDER BY position", (doc_id,))
        for f in out:
            f["bbox"] = json.loads(f["bbox"]) if f["bbox"] else None
            f["grounded"], f["valid"] = bool(f["grounded"]), bool(f["valid"])
            f["final_value"] = f["corrected_value"] if f["corrected_value"] is not None else f["value"]
        return out

    def correct_field(self, doc_id: str, name: str, value: str) -> dict | None:
        doc = self.get_document(doc_id)
        f = self._one("SELECT * FROM fields WHERE doc_id=? AND name=?", (doc_id, name))
        if not doc or not f:
            return None
        with self._lock, self._db:
            self._db.execute("UPDATE fields SET corrected_value=? WHERE doc_id=? AND name=?", (value, doc_id, name))
            self._db.execute(
                "INSERT INTO corrections (doc_id, doc_type, field, original, corrected, confidence, snippet, created) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, doc["doc_type"], name, f["value"], value, f["confidence"], f["snippet"], time.time()))
        return {"doc_id": doc_id, "field": name, "original": f["value"], "corrected": value}

    def corrections(self, doc_type: str | None = None, limit: int = 1000) -> list[dict]:
        if doc_type:
            return self._rows("SELECT * FROM corrections WHERE doc_type=? ORDER BY id DESC LIMIT ?", (doc_type, limit))
        return self._rows("SELECT * FROM corrections ORDER BY id DESC LIMIT ?", (limit,))

    def all_fields(self) -> list[dict]:
        return self._rows("SELECT f.*, d.doc_type FROM fields f JOIN documents d ON d.id=f.doc_id "
                          "WHERE d.status='done'")

    # ---------- chunks ----------
    @property
    def chunk_version(self) -> int:
        return self._chunk_version

    def save_chunks(self, doc_id: str, chunks: list[dict], embeddings: list | None) -> None:
        embs = embeddings if embeddings is not None else [None] * len(chunks)
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._db.executemany(
                "INSERT INTO chunks (doc_id, page_no, position, text, embedding) VALUES (?,?,?,?,?)",
                [(doc_id, c["page_no"], i, c["text"], _blob(e)) for i, (c, e) in enumerate(zip(chunks, embs))],
            )
            self._chunk_version += 1

    def all_chunks(self) -> tuple[list[dict], np.ndarray | None]:
        with self._lock:
            rows = [dict(r) for r in self._db.execute(
                "SELECT c.id, c.doc_id, c.page_no, c.text, c.embedding, d.filename, d.doc_type FROM chunks c "
                "JOIN documents d ON d.id=c.doc_id WHERE d.status='done' ORDER BY d.created, c.position")]
        blobs = [r.pop("embedding") for r in rows]
        if rows and all(b is not None for b in blobs):
            return rows, np.vstack([np.frombuffer(b, dtype=np.float32) for b in blobs])
        return rows, None

    def chunks_missing_embeddings(self) -> list[dict]:
        return self._rows("SELECT id, text FROM chunks WHERE embedding IS NULL")

    def set_embeddings(self, pairs: list[tuple[int, object]]) -> None:
        with self._lock, self._db:
            self._db.executemany("UPDATE chunks SET embedding=? WHERE id=?", [(_blob(v), i) for i, v in pairs])
            self._chunk_version += 1

    def bump_chunks(self) -> None:
        with self._lock:
            self._chunk_version += 1

    # ---------- OCR cache ----------
    def ocr_cache_get(self, h: str) -> dict | None:
        r = self._one("SELECT result FROM ocr_cache WHERE hash=?", (h,))
        return json.loads(r["result"]) if r else None

    def ocr_cache_put(self, h: str, result: dict) -> None:
        self._exec("INSERT OR REPLACE INTO ocr_cache VALUES (?,?,?)", (h, json.dumps(result), time.time()))

    # ---------- Q&A ----------
    def add_qa(self, session_id: str, role: str, content: str, sources=None, meta=None) -> str:
        mid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO qa_messages VALUES (?,?,?,?,?,?,?)",
                   (mid, session_id, role, content, _j(sources), _j(meta), time.time()))
        return mid

    def qa_history(self, session_id: str) -> list[dict]:
        out = self._rows("SELECT * FROM qa_messages WHERE session_id=? ORDER BY created", (session_id,))
        for m in out:
            m["sources"] = json.loads(m["sources"]) if m["sources"] else []
            m["meta"] = json.loads(m["meta"]) if m["meta"] else {}
        return out

    def clear_qa(self, session_id: str) -> None:
        self._exec("DELETE FROM qa_messages WHERE session_id=?", (session_id,))

    # ---------- audit ----------
    def audit(self, actor: str, action: str, doc_id: str | None = None, detail: str | None = None) -> None:
        self._exec("INSERT INTO audit (ts, actor, action, doc_id, detail) VALUES (?,?,?,?,?)",
                   (time.time(), actor, action, doc_id, redact(detail) if detail else None))

    def audit_log(self, limit: int = 200) -> list[dict]:
        return self._rows("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
