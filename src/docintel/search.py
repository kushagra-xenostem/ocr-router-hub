"""Chunking, local embeddings, and hybrid (BM25 + vector, RRF) retrieval with optional reranking."""
import re
import math
import threading
import numpy as np
from .config import ROOT
from typing import Protocol
from collections import Counter
from dataclasses import dataclass

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODEL_DIR = ROOT / "models"
_TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by for from has have how i in is it its of on or that the this to was what "
                 "when where which who why will with do does did can you your".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def chunk_pages(pages: list[dict], size: int = 120, overlap: int = 25) -> list[dict]:
    chunks = []
    for p in pages:
        words = p["text"].split()
        step = size - overlap
        for start in range(0, max(1, len(words)), step):
            piece = words[start:start + size]
            if piece:
                chunks.append({"page_no": p["page_no"], "text": " ".join(piece)})
            if start + size >= len(words):
                break
    return chunks


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _normalize(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class FastEmbedder:
    def __init__(self):
        self._m, self._lock = None, threading.Lock()

    def _load(self):
        with self._lock:
            if self._m is None:
                from fastembed import TextEmbedding

                self._m = TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_DIR))
        return self._m

    def embed(self, texts):
        if not texts:
            return np.zeros((0, 384), np.float32)
        return _normalize(np.vstack(list(self._load().embed(texts, batch_size=32))))

    def embed_query(self, text):
        return _normalize(np.asarray(next(iter(self._load().query_embed(text)))))


class FastReranker:
    def __init__(self):
        self._m, self._lock = None, threading.Lock()

    def score(self, query, texts):
        with self._lock:
            if self._m is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._m = TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODEL_DIR))
            return [float(s) for s in self._m.rerank(query, texts)] if texts else []


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(t.values()) for t in self.tfs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0
        df = Counter()
        for t in self.tfs:
            df.update(t.keys())
        self.idf = {t: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1)))
            out.append(s)
        return out


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 5
    rerank: bool = True
    min_rerank_score: float = -6.0
    min_similarity: float = 0.5
    candidates: int = 30


class Index:
    """Hybrid retrieval over every processed document; optionally restricted to some documents."""

    def __init__(self, store, embedder: Embedder | None, reranker: Reranker | None):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self._lock = threading.Lock()
        self._version = -1
        self._chunks, self._matrix, self._bm25 = [], None, BM25([])

    def _refresh(self):
        with self._lock:
            if self._version != self.store.chunk_version:
                self._chunks, self._matrix = self.store.all_chunks()
                self._bm25 = BM25([c["text"] for c in self._chunks])
                self._version = self.store.chunk_version

    @property
    def semantic_ready(self) -> bool:
        self._refresh()
        return self.embedder is not None and (self._matrix is not None or not self._chunks)

    def search(self, query: str, cfg: SearchConfig = SearchConfig(), doc_ids: list[str] | None = None) -> list[dict]:
        self._refresh()
        allowed = [i for i, c in enumerate(self._chunks) if not doc_ids or c["doc_id"] in doc_ids]
        if not allowed:
            return []
        pool = max(cfg.candidates, cfg.top_k)
        bm = self._bm25.scores(query)
        lex = sorted((i for i in allowed if bm[i] > 0), key=lambda i: (-bm[i], i))[:pool]
        lists = [lex]
        if self.semantic_ready and self._matrix is not None:
            sims = self._matrix @ self.embedder.embed_query(query)
            sem = sorted((i for i in allowed if sims[i] >= cfg.min_similarity), key=lambda i: (-float(sims[i]), i))[:pool]
            lists.append(sem)
        fused: dict[int, float] = {}
        for lst in lists:
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:pool]
        method = "hybrid" if len(lists) > 1 else "bm25"
        if cfg.rerank and self.reranker is not None and ranked:
            scores = self.reranker.score(query, [self._chunks[i]["text"] for i, _ in ranked])
            ranked = sorted(((i, s) for (i, _), s in zip(ranked, scores) if s >= cfg.min_rerank_score),
                            key=lambda x: (-x[1], x[0]))
            method += "+rerank"
        return [{**self._chunks[i], "score": round(s, 4), "method": method} for i, s in ranked[:cfg.top_k]]
