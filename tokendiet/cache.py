"""Score cache.

The brief requires cross-encoder scores cached on ``hash(query + sentence)``;
the deck lists Redis in the stack. Redis is used when reachable, otherwise we
fall back to an on-disk cache so ``clone -> run`` works without Docker Desktop.

This is what makes the dashboard's live sliders (budget, lambda, threshold) re-run
in milliseconds: changing a budget does not re-score anything.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .config import CACHE_LOCAL_PATH, settings


def score_key(query: str, sentence: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(query.encode("utf-8"))
    h.update(b"\x00")
    h.update(sentence.encode("utf-8"))
    return h.hexdigest()


class _SqliteCache:
    """Dependency-free fallback. Threadsafe via a lock; fine at our scale."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        if not keys:
            return {}
        out: dict[str, float] = {}
        with self._lock:
            # chunked IN () to stay under SQLite's variable limit
            for i in range(0, len(keys), 500):
                batch = keys[i : i + 500]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(batch))})"
                for k, v in self._conn.execute(q, batch):
                    out[k] = json.loads(v)
        return out

    def set_many(self, items: dict[str, float]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )
            self._conn.commit()

    @property
    def backend(self) -> str:
        return "sqlite"


class _RedisCache:
    def __init__(self, url: str):
        import redis  # imported lazily so the fallback needs no redis package

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()  # fail fast if unreachable

    def get_many(self, keys: list[str]) -> dict[str, float]:
        if not keys:
            return {}
        vals = self._r.mget(keys)
        return {k: json.loads(v) for k, v in zip(keys, vals) if v is not None}

    def set_many(self, items: dict[str, float]) -> None:
        if items:
            self._r.mset({k: json.dumps(v) for k, v in items.items()})

    @property
    def backend(self) -> str:
        return "redis"


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
    _cache = _SqliteCache(CACHE_LOCAL_PATH / "scores.sqlite")
    return _cache
