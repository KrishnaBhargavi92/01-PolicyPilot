import argparse
import json
import sqlite3

import joblib
import numpy as np

from .config import DEFAULT_VECTOR_STORE, EMBEDDING_MODEL_PATH, VECTOR_DB_PATH, VECTOR_STORES
from .vectorstores.pinecone import query_pinecone


def embed_question(question: str) -> list[float]:
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDING_MODEL_PATH}. Run scripts/build_index.py first."
        )

    model = joblib.load(EMBEDDING_MODEL_PATH)
    embedding = model.transform([question])[0]
    return np.asarray(embedding, dtype=np.float32).round(8).tolist()


def retrieve_chunks_from_sqlite(question: str, top_k: int) -> list[dict]:
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(f"Missing {VECTOR_DB_PATH}. Run scripts/build_index.py first.")

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


def retrieve_chunks_from_pinecone(question: str, top_k: int) -> list[dict]:
    query_embedding = embed_question(question)
    return query_pinecone(query_embedding, top_k)


def retrieve_chunks(
    question: str,
    top_k: int,
    vector_store: str = DEFAULT_VECTOR_STORE,
) -> list[dict]:
    if vector_store == "pinecone":
        return retrieve_chunks_from_pinecone(question, top_k)
    if vector_store == "sqlite":
        return retrieve_chunks_from_sqlite(question, top_k)

    raise ValueError(f"Vector store must be one of: {', '.join(VECTOR_STORES)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a question or retrieve RAG chunks.")
    parser.add_argument("question", help="Question to embed or retrieve against.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Retrieve top K chunks instead of printing only the embedding.",
    )
    parser.add_argument(
        "--vector-store",
        choices=VECTOR_STORES,
        default=DEFAULT_VECTOR_STORE,
        help="Vector store to use for retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k > 0:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "top_k": args.top_k,
                    "vector_store": args.vector_store,
                    "results": retrieve_chunks(
                        args.question,
                        args.top_k,
                        vector_store=args.vector_store,
                    ),
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
