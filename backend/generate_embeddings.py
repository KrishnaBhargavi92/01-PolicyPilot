import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


BASE_DIR = Path(__file__).resolve().parent
EXTRACTED_DIR = BASE_DIR / "extracted_text"
CHUNKS_PATH = EXTRACTED_DIR / "rag_chunks.jsonl"
EMBEDDINGS_PATH = EXTRACTED_DIR / "rag_embeddings.jsonl"
EMBEDDING_MODEL_PATH = EXTRACTED_DIR / "rag_embedding_model.joblib"
VECTOR_DB_PATH = EXTRACTED_DIR / "rag_vector_store.sqlite"

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "local-tfidf-svd",
)
MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "4096"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "128"))


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Run backend/read_text.py first. Missing: {path}")

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


def init_vector_db(path: Path) -> sqlite3.Connection:
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
    path: Path,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    dimensions: int,
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
                        MODEL_NAME,
                        dimensions,
                        np.asarray(embedding, dtype=np.float32).tobytes(),
                    )
                    for record, embedding in zip(records, embeddings)
                ],
            )
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("embedding_model", MODEL_NAME),
                    ("embedding_dimensions", str(dimensions)),
                    ("chunk_count", str(len(records))),
                ],
            )
    finally:
        connection.close()


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
    )
    tfidf = vectorizer.fit_transform(texts)
    dimensions = min(EMBEDDING_DIMENSIONS, tfidf.shape[0] - 1, tfidf.shape[1] - 1)

    if dimensions < 1:
        raise ValueError("Not enough text to create embeddings")

    print(f"Generating {dimensions}-dimensional local embeddings with TF-IDF + SVD")
    model = Pipeline(
        [
            ("tfidf", vectorizer),
            ("svd", TruncatedSVD(n_components=dimensions, random_state=42)),
            ("normalize", Normalizer(copy=False)),
        ]
    )
    embeddings = model.fit_transform(texts)
    joblib.dump(model, EMBEDDING_MODEL_PATH)

    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as output_file:
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                **chunk,
                "embedding_model": MODEL_NAME,
                "embedding_dimensions": dimensions,
                "embedding": embedding.round(8).tolist(),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_vector_db(VECTOR_DB_PATH, chunks, embeddings, dimensions)

    print(f"Created {EMBEDDINGS_PATH} with {len(chunks)} embeddings")
    print(f"Saved fitted embedding model to {EMBEDDING_MODEL_PATH}")
    print(f"Saved local vector database to {VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()
