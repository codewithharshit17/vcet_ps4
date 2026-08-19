"""Qdrant access.

The deck specifies Qdrant. We use it in embedded mode by default (a local
persisted path, no server process) so `clone -> run` works without Docker
Desktop, and switch to the docker-compose service when TOKENDIET_QDRANT_URL is
set. Same client library, same API, one code path.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import QDRANT_LOCAL_PATH, settings

EMBED_DIM = 384  # all-MiniLM-L6-v2


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    char_offset: int
    text: str
    token_count: int

    @property
    def uid(self) -> int:
        """Deterministic point id.

        Must NOT use builtin hash(): Python randomises string hashing per
        process, so ids written at ingest would never match ids recomputed at
        query time, and every dense hit would fail to map back to a chunk --
        silently degrading "hybrid" retrieval to BM25-only.
        """
        digest = hashlib.blake2b(
            f"{self.doc_id}:{self.chunk_id}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") % (2**63)


def get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_LOCAL_PATH))


def backend_name() -> str:
    return f"qdrant-server({settings.qdrant_url})" if settings.qdrant_url else "qdrant-embedded"


def reset_storage() -> None:
    """Hard-reset the index before a rebuild.

    In embedded mode ``delete_collection`` is not reliable: it reports success
    and ``collection_exists`` then returns False *within the same process*, but
    a freshly opened process still sees every original point -- the drop is
    never persisted. Since a stale collection silently turns the next ingest
    into a duplicate-producing append, embedded mode removes the storage
    directory outright. Server mode has correct semantics, so it uses the API.
    """
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url)
        if client.collection_exists(settings.qdrant_collection):
            client.delete_collection(settings.qdrant_collection)
        client.close()
        return
    shutil.rmtree(QDRANT_LOCAL_PATH, ignore_errors=True)


def ensure_collection(client: QdrantClient, *, recreate: bool = False) -> None:
    """Create the collection, optionally dropping it first.

    The postcondition is asserted rather than assumed. A drop that silently
    fails leaves the previous generation of points in place, and because a
    tokenizer/id change alters point ids, the next upsert *adds* rather than
    overwrites -- producing a corpus of duplicates. That inflates the baseline
    token count and would quietly flatter every compression number downstream.
    Fail loudly instead.
    """
    name = settings.qdrant_collection

    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    if recreate:
        remaining = client.count(name, exact=True).count
        if remaining:
            raise RuntimeError(
                f"recreate=True but collection {name!r} still holds {remaining} points. "
                "Refusing to ingest on top of stale data. "
                "Delete the data/qdrant directory (or the server collection) and retry."
            )


def count(client: QdrantClient) -> int:
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    return client.count(settings.qdrant_collection, exact=True).count


def iter_all_chunks(client: QdrantClient) -> list[Chunk]:
    """Read every chunk back. BM25 is rebuilt from this, so Qdrant stays the
    single source of truth rather than us maintaining a parallel index file."""
    out: list[Chunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1024,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            out.append(
                Chunk(
                    doc_id=pl["doc_id"],
                    chunk_id=pl["chunk_id"],
                    char_offset=pl["char_offset"],
                    text=pl["text"],
                    token_count=pl["token_count"],
                )
            )
        if offset is None:
            break
    out.sort(key=lambda c: (c.doc_id, c.chunk_id))
    return out
