import json
import sqlite3
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi
import numpy as np

from ..config import (
    PINECONE_API_KEY,
    PINECONE_BATCH_SIZE,
    PINECONE_INDEX_HOST,
    PINECONE_NAMESPACE,
    VECTOR_DB_PATH,
)


def normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if host.startswith("http://") or host.startswith("https://"):
        return host
    return f"https://{host}"


def load_vectors_from_sqlite(path: Path = VECTOR_DB_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_index.py before uploading to Pinecone."
        )

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                source,
                page,
                chunk_index,
                text,
                embedding_dimensions,
                embedding
            FROM chunks
            ORDER BY source, page, chunk_index
            """
        ).fetchall()
    finally:
        connection.close()

    vectors = []
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        vectors.append(
            {
                "id": row["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "source": row["source"],
                    "page": row["page"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "embedding_dimensions": row["embedding_dimensions"],
                },
            }
        )

    return vectors


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Api-Key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with urllib.request.urlopen(request, context=ssl_context, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Pinecone API error {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Pinecone: {error.reason}") from error


def upsert_vectors_to_pinecone(
    vectors: list[dict[str, Any]],
    api_key: str,
    index_host: str,
    namespace: str = PINECONE_NAMESPACE,
    batch_size: int = PINECONE_BATCH_SIZE,
) -> int:
    if not vectors:
        return 0

    upsert_url = f"{normalize_host(index_host)}/vectors/upsert"
    total_upserted = 0

    for batch in batched(vectors, batch_size):
        response = post_json(
            upsert_url,
            {
                "vectors": batch,
                "namespace": namespace,
            },
            api_key=api_key,
        )
        total_upserted += response.get("upsertedCount", len(batch))

    return total_upserted


def query_pinecone(
    vector: list[float],
    top_k: int,
    api_key: str | None = PINECONE_API_KEY,
    index_host: str | None = PINECONE_INDEX_HOST,
    namespace: str = PINECONE_NAMESPACE,
) -> list[dict[str, Any]]:
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY before querying Pinecone.")
    if not index_host:
        raise RuntimeError("Set PINECONE_INDEX_HOST before querying Pinecone.")

    response = post_json(
        f"{normalize_host(index_host)}/query",
        {
            "vector": vector,
            "topK": top_k,
            "includeMetadata": True,
            "namespace": namespace,
        },
        api_key=api_key,
    )

    results = []
    for match in response.get("matches", []):
        metadata = match.get("metadata") or {}
        results.append(
            {
                "score": round(float(match.get("score", 0.0)), 6),
                "id": match["id"],
                "source": metadata.get("source", ""),
                "page": int(metadata.get("page", 0)),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "text": metadata.get("text", ""),
            }
        )

    return results


def upload_local_vectors_to_pinecone(
    vector_db_path: Path = VECTOR_DB_PATH,
    api_key: str | None = PINECONE_API_KEY,
    index_host: str | None = PINECONE_INDEX_HOST,
    namespace: str = PINECONE_NAMESPACE,
    batch_size: int = PINECONE_BATCH_SIZE,
) -> int:
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY before uploading vectors to Pinecone.")
    if not index_host:
        raise RuntimeError("Set PINECONE_INDEX_HOST before uploading vectors to Pinecone.")

    vectors = load_vectors_from_sqlite(vector_db_path)
    return upsert_vectors_to_pinecone(
        vectors=vectors,
        api_key=api_key,
        index_host=index_host,
        namespace=namespace,
        batch_size=batch_size,
    )
