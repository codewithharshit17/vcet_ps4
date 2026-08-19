"""Cache for the two expensive local models.

The brief requires cross-encoder scores cached on hash(query + sentence); the
deck lists Redis in the stack. Sentence *embeddings* are cached too, because
profiling showed they cost 2.8s of a 3.5s MMR stage -- and unlike scores, an
embedding does not depend on the query, so it is reusable across every query
that retrieves the same chunk.

This is what makes the dashboard's live sliders usable: changing the budget,
lambda or threshold re-runs selection without re-scoring or re-embedding
anything.

Redis is used when reachable; otherwise an on-disk SQLite cache, so
`clone -> run` works with Docker Desktop stopped.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .config import CACHE_LOCAL_PATH, settings


def score_key(query: str, sentence: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (model, query, sentence):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return "s:" + h.hexdigest()


def embed_key(sentence: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (model, sentence):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return "e:" + h.hexdigest()


def encode_vec(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="float32").tobytes()).decode("ascii")


def decode_vec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="float32")


class _SqliteCache:
    """Dependency-free fallback. Threadsafe via a lock; fine at our scale."""

    backend = "sqlite"

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        out: dict[str, str] = {}
        with self._lock:
            for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
                batch = keys[i : i + 500]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(batch))})"
                out.update(dict(self._conn.execute(q, batch)))
        return out

    def set_many(self, items: dict[str, str]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", list(items.items())
            )
            self._conn.commit()


class _RedisCache:
    backend = "redis"

    def __init__(self, url: str):
        import redis  # lazy: the fallback needs no redis package

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()  # fail fast if unreachable

    def get_many(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        return {k: v for k, v in zip(keys, self._r.mget(keys)) if v is not None}

    def set_many(self, items: dict[str, str]) -> None:
        if items:
            self._r.mset(items)


_cache = None


def get_cache():
    """Redis if configured and reachable, else on-disk SQLite."""
    global _cache
    if _cache is not None:
        return _cache
    if settings.redis_url:
        try:
            _cache = _RedisCache(settings.redis_url)
            return _cache
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] Redis at {settings.redis_url} unreachable ({exc}); using SQLite.")
    _cache = _SqliteCache(CACHE_LOCAL_PATH / "tokendiet.sqlite")
    return _cache


# --- typed helpers -----------------------------------------------------------

def get_scores(keys: list[str]) -> dict[str, float]:
    return {k: json.loads(v) for k, v in get_cache().get_many(keys).items()}


def put_scores(items: dict[str, float]) -> None:
    get_cache().set_many({k: json.dumps(v) for k, v in items.items()})


def get_vectors(keys: list[str]) -> dict[str, np.ndarray]:
    return {k: decode_vec(v) for k, v in get_cache().get_many(keys).items()}


def put_vectors(items: dict[str, np.ndarray]) -> None:
    get_cache().set_many({k: encode_vec(v) for k, v in items.items()})
