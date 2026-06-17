# RAG App

Local PDF RAG pipeline:

1. Extract PDF text into chunks.
2. Generate local TF-IDF + SVD embeddings.
3. Store vectors in SQLite.
4. Retrieve relevant chunks and answer with OpenAI citations.

The PDFs in `backend/docs/` are anonymized sample policy documents for demo use.
They do not contain real client/company details.

## Setup

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set your own `OPENAI_API_KEY`.

## Build Local RAG Artifacts

To regenerate the anonymized sample PDFs:

```bash
python3 scripts/generate_sample_pdfs.py
```

```bash
venv/bin/python backend/read_text.py
venv/bin/python backend/generate_embeddings.py
```

This creates ignored local files under `backend/extracted_text/`, including:

- `rag_chunks.jsonl`
- `rag_embedding_model.joblib`
- `rag_embeddings.jsonl`
- `rag_vector_store.sqlite`

## Ask A Question

```bash
venv/bin/python backend/ask_llm.py "When must a new employee complete Form I-9?" --top-k 3
```

Debug retrieval and prompt without calling OpenAI:

```bash
venv/bin/python backend/ask_llm.py "When must a new employee complete Form I-9?" --top-k 3 --dry-run
```

## Run Streamlit UI

```bash
venv/bin/streamlit run streamlit_app.py
```

If `backend/extracted_text/` is missing, the UI rebuilds the local RAG index from
the PDFs the first time you ask a question.

For Streamlit Cloud, add these secrets in the app settings:

```toml
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
LLM_PROVIDER = "openai"
RAG_TOP_K = "4"
```

## Git Safety

Do not commit `backend/.env` or generated files in `backend/extracted_text/`.
