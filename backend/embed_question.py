import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
EXTRACTED_DIR = BASE_DIR / "extracted_text"
EMBEDDING_MODEL_PATH = EXTRACTED_DIR / "rag_embedding_model.joblib"
VECTOR_DB_PATH = EXTRACTED_DIR / "rag_vector_store.sqlite"


def embed_question(question: str) -> list[float]:
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDING_MODEL_PATH}. Run backend/generate_embeddings.py first."
        )

    model = joblib.load(EMBEDDING_MODEL_PATH)
    embedding = model.transform([question])[0]
    return np.asarray(embedding, dtype=np.float32).round(8).tolist()


def retrieve_chunks(question: str, top_k: int) -> list[dict]:
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing {VECTOR_DB_PATH}. Run backend/generate_embeddings.py first."
        )

    query_embedding = np.asarray(embed_question(question), dtype=np.float32)
    connection = sqlite3.connect(VECTOR_DB_PATH)
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
            """
        ).fetchall()
    finally:
        connection.close()

    results = []
    for row in rows:
        chunk_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        if chunk_embedding.shape != query_embedding.shape:
            raise ValueError(
                f"Embedding dimension mismatch for {row['id']}: "
                f"{chunk_embedding.shape[0]} != {query_embedding.shape[0]}"
            )

        score = float(np.dot(query_embedding, chunk_embedding))
        results.append(
            {
                "score": round(score, 6),
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a user question or retrieve RAG chunks.")
    parser.add_argument("question", help="The user question to embed.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Retrieve the top K most relevant chunks instead of only printing the embedding.",
    )
    args = parser.parse_args()

    if args.top_k > 0:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "top_k": args.top_k,
                    "results": retrieve_chunks(args.question, args.top_k),
                },
                ensure_ascii=False,
            )
        )
        return

    embedding = embed_question(args.question)
    print(
        json.dumps(
            {
                "question": args.question,
                "embedding_dimensions": len(embedding),
                "embedding": embedding,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
