import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from policylens.config import PINECONE_BATCH_SIZE, PINECONE_NAMESPACE
from policylens.pipeline import build_rag_artifacts
from policylens.vectorstores.pinecone import upload_local_vectors_to_pinecone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local vectors to Pinecone.")
    parser.add_argument(
        "--namespace",
        default=PINECONE_NAMESPACE,
        help="Pinecone namespace to upsert into.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PINECONE_BATCH_SIZE,
        help="Number of vectors per Pinecone upsert request.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not build local RAG artifacts before uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_build:
        build_rag_artifacts()

    upserted_count = upload_local_vectors_to_pinecone(
        namespace=args.namespace,
        batch_size=args.batch_size,
    )
    print(f"Upserted {upserted_count} vectors to Pinecone namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
