from pathlib import Path

from .config import CHUNKS_PATH, EMBEDDING_MODEL_PATH, VECTOR_DB_PATH
from .embeddings import generate_embeddings
from .ingestion import extract_documents


RAG_ARTIFACTS: tuple[Path, ...] = (
    CHUNKS_PATH,
    EMBEDDING_MODEL_PATH,
    VECTOR_DB_PATH,
)


def rag_artifacts_exist() -> bool:
    return all(path.exists() for path in RAG_ARTIFACTS)


def build_rag_artifacts(force: bool = False) -> None:
    if rag_artifacts_exist() and not force:
        return

    extract_documents()
    generate_embeddings()
