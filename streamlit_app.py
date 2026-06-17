import os
from pathlib import Path

import streamlit as st


def load_streamlit_secrets() -> None:
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}

    for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "LLM_PROVIDER", "RAG_TOP_K"):
        if key in secrets and key not in os.environ:
            os.environ[key] = str(secrets[key])


load_streamlit_secrets()

from backend.ask_llm import answer_question  # noqa: E402
from backend.generate_embeddings import EMBEDDING_MODEL_PATH, VECTOR_DB_PATH  # noqa: E402
from backend.generate_embeddings import main as generate_embeddings  # noqa: E402
from backend.read_text import CHUNKS_PATH  # noqa: E402
from backend.read_text import main as extract_text  # noqa: E402


def ensure_rag_artifacts() -> None:
    required_paths: tuple[Path, ...] = (
        CHUNKS_PATH,
        EMBEDDING_MODEL_PATH,
        VECTOR_DB_PATH,
    )
    if all(path.exists() for path in required_paths):
        return

    with st.status("Building local RAG index...", expanded=True) as status:
        st.write("Extracting PDF text")
        extract_text()
        st.write("Generating embeddings and vector database")
        generate_embeddings()
        status.update(label="RAG index ready", state="complete")


st.set_page_config(page_title="Policy RAG", layout="wide")

st.title("Policy RAG")

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

question = st.text_area(
    "Question",
    value="When must a new employee complete Form I-9?",
    height=110,
)

ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            ensure_rag_artifacts()
            result = answer_question(
                question=question.strip(),
                top_k=top_k,
                provider=provider,
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
