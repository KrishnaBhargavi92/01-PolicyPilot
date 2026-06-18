# PolicyLens

Local PDF RAG pipeline:

1. Extract PDF text into chunks.
2. Generate local TF-IDF + SVD embeddings.
3. Store vectors in SQLite.
4. Retrieve relevant chunks and answer with OpenAI citations.

The PDFs in `data/docs/` are anonymized sample policy documents for demo use.
They do not contain real client/company details.

## Setup

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set your own `OPENAI_API_KEY`.

## Project Structure

```text
app/                 Streamlit UI
data/docs/           Sample source PDFs committed to git
data/generated/      Local generated chunks, embeddings, and vector DB
policylens/          Reusable RAG package
policylens/vectorstores/
scripts/             Operational commands
backend/             Backward-compatible script wrappers
```

## Build Local RAG Artifacts

To regenerate the anonymized sample PDFs:

```bash
python3 scripts/generate_sample_pdfs.py
```

```bash
venv/bin/python scripts/build_index.py
```

This creates ignored local files under `data/generated/`, including:

- `rag_chunks.jsonl`
- `rag_embedding_model.joblib`
- `rag_embeddings.jsonl`
- `rag_vector_store.sqlite`

## Store Vectors In Pinecone

Create a Pinecone index with:

- Dimension: the `embedding_dimensions` printed by `scripts/build_index.py`
- Metric: cosine

Then set Pinecone values in `backend/.env`:

```bash
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_HOST="your-index-host"
PINECONE_NAMESPACE="policy-lens"
```

Upload vectors:

```bash
venv/bin/python scripts/upsert_pinecone.py
```

## Ask A Question

```bash
venv/bin/python -m policylens.llm "When must a new employee complete Form I-9?" --top-k 3
```

Debug retrieval and prompt without calling OpenAI:

```bash
venv/bin/python -m policylens.llm "When must a new employee complete Form I-9?" --top-k 3 --dry-run
```

## Run Streamlit UI

```bash
venv/bin/streamlit run app/streamlit_app.py
```

If `data/generated/` is missing, the UI rebuilds the local RAG index from
the PDFs the first time you ask a question.

For Streamlit Cloud, add these secrets in the app settings:

```toml
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
LLM_PROVIDER = "openai"
RAG_TOP_K = "4"
```

## Git Safety

Do not commit `backend/.env` or generated files in `data/generated/`.
