import os
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_streamlit_secrets() -> None:
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}

    for key in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "LLM_PROVIDER",
        "RAG_TOP_K",
        "EMBEDDING_MODEL",
        "EMBEDDING_BATCH_SIZE",
        "VECTOR_STORE",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_HOST",
        "PINECONE_NAMESPACE",
    ):
        if key in secrets and key not in os.environ:
            os.environ[key] = str(secrets[key])


load_streamlit_secrets()

from policylens.llm import answer_question  # noqa: E402
from policylens.pipeline import build_rag_artifacts, rag_artifacts_exist  # noqa: E402
from policylens.config import DEFAULT_VECTOR_STORE, VECTOR_STORES  # noqa: E402


def ensure_rag_index() -> None:
    if rag_artifacts_exist():
        return

    with st.status("Building local RAG index...", expanded=True) as status:
        st.write("Extracting PDF text")
        st.write("Generating embeddings and vector database")
        build_rag_artifacts()
        status.update(label="RAG index ready", state="complete")


st.set_page_config(page_title="PolicyLens", layout="wide")

st.title("PolicyLens")

with st.sidebar:
    st.header("Search")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    include_chunks = st.toggle("Show retrieved chunks", value=False)
    dry_run = st.toggle("Dry run", value=False)

    st.header("Model")
    provider = st.selectbox(
        "Provider",
        options=["openai", "ollama", "openai-compatible"],
        index=0,
    )
    vector_store = st.selectbox(
        "Vector store",
        options=list(VECTOR_STORES),
        index=list(VECTOR_STORES).index(DEFAULT_VECTOR_STORE),
    )

question = st.text_area(
    "Question",
    value="When must a new employee complete Form I-9?",
    height=110,
)

ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            ensure_rag_index()
            result = answer_question(
                question=question.strip(),
                top_k=top_k,
                provider=provider,
                vector_store=vector_store,
                dry_run=dry_run,
                include_chunks=include_chunks,
            )
        except Exception as error:
            st.error(str(error))
        else:
            st.subheader("Answer")
            if result["answer"]:
                st.markdown(result["answer"])
            else:
                st.info("Dry run: no LLM call was made.")

            st.subheader("Citations")
            for citation in result["citations"]:
                st.markdown(f"- {citation['label']}")

            if include_chunks or dry_run:
                st.subheader("Retrieved Chunks")
                for chunk in result.get("chunks", []):
                    with st.expander(
                        f"{chunk['source']} p. {chunk['page']} "
                        f"(score {chunk['score']})"
                    ):
                        st.write(chunk["text"])

            if dry_run and "messages" in result:
                st.subheader("Prompt")
                st.json(result["messages"])
