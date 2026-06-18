import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CHUNKS_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDINGS_PATH,
    VECTOR_DB_PATH,
)


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/build_index.py first. Missing: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error

    return chunks


def init_vector_db(path: Path = VECTOR_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    return connection


def save_vector_db(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    dimensions: int,
    path: Path = VECTOR_DB_PATH,
) -> None:
    connection = init_vector_db(path)
    try:
        with connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM metadata")
            connection.executemany(
                """
                INSERT INTO chunks (
                    id,
                    source,
                    page,
                    chunk_index,
                    text,
                    embedding_model,
                    embedding_dimensions,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record["id"],
                        record["source"],
                        record["page"],
                        record["chunk_index"],
                        record["text"],
                        EMBEDDING_MODEL_NAME,
                        dimensions,
                        np.asarray(embedding, dtype=np.float32).tobytes(),
                    )
                    for record, embedding in zip(records, embeddings)
                ],
            )
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("embedding_model", EMBEDDING_MODEL_NAME),
                    ("embedding_dimensions", str(dimensions)),
                    ("chunk_count", str(len(records))),
                ],
            )
    finally:
        connection.close()


def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install sentence-transformers before generating embeddings: "
            "pip install -r requirements.txt"
        ) from error

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode_texts(texts: list[str]) -> np.ndarray:
    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_embedding_metadata(dimensions: int) -> None:
    EMBEDDING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDING_MODEL_PATH.write_text(
        json.dumps(
            {
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_provider": "sentence-transformers",
                "embedding_dimensions": dimensions,
                "normalized": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def generate_embeddings() -> int:
    chunks = load_chunks()
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")

    texts = [chunk["text"] for chunk in chunks]
    print(f"Generating embeddings with sentence-transformers model: {EMBEDDING_MODEL_NAME}")
    embeddings = encode_texts(texts)
    dimensions = int(embeddings.shape[1])
    save_embedding_metadata(dimensions)

    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as output_file:
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                **chunk,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dimensions": dimensions,
                "embedding": embedding.round(8).tolist(),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_vector_db(chunks, embeddings, dimensions)

    print(f"Created {EMBEDDINGS_PATH} with {len(chunks)} embeddings")
    print(f"Saved embedding metadata to {EMBEDDING_MODEL_PATH}")
    print(f"Saved local vector database to {VECTOR_DB_PATH}")
    return len(chunks)


def main() -> None:
    generate_embeddings()


if __name__ == "__main__":
    main()
