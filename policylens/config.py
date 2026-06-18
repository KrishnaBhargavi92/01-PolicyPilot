import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
GENERATED_DIR = DATA_DIR / "generated"

CHUNKS_PATH = GENERATED_DIR / "rag_chunks.jsonl"
EMBEDDINGS_PATH = GENERATED_DIR / "rag_embeddings.jsonl"
EMBEDDING_MODEL_PATH = GENERATED_DIR / "rag_embedding_model.joblib"
VECTOR_DB_PATH = GENERATED_DIR / "rag_vector_store.sqlite"

ENV_PATHS = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "backend" / ".env",
)

PROVIDERS = ("openai", "ollama", "openai-compatible")


def load_env_files() -> None:
    for path in ENV_PATHS:
        if not path.exists():
            continue

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_files()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/responses")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

OPENAI_COMPATIBLE_URL = os.getenv(
    "LLM_API_URL",
    "https://api.openai.com/v1/chat/completions",
)
OPENAI_COMPATIBLE_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_COMPATIBLE_API_KEY = os.getenv("LLM_API_KEY")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "local-tfidf-svd")
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "4096"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "128"))

VECTOR_STORES = ("pinecone", "sqlite")
DEFAULT_VECTOR_STORE = os.getenv("VECTOR_STORE", "pinecone").strip().lower()
if DEFAULT_VECTOR_STORE not in VECTOR_STORES:
    raise ValueError(f"VECTOR_STORE must be one of: {', '.join(VECTOR_STORES)}")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "policy-lens")
PINECONE_BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE", "100"))
