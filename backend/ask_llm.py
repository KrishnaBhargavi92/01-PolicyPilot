import argparse
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi

try:
    from .embed_question import retrieve_chunks
except ImportError:
    from embed_question import retrieve_chunks


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

PROVIDERS = ("openai", "ollama", "openai-compatible")


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()

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


def provider_model(provider: str) -> str:
    if provider == "openai":
        return OPENAI_MODEL
    if provider == "ollama":
        return OLLAMA_MODEL
    if provider == "openai-compatible":
        return OPENAI_COMPATIBLE_MODEL
    raise ValueError(f"Unsupported provider: {provider}")


def build_instructions() -> str:
    return (
        "You answer questions using only the provided retrieved context. "
        "If the context is insufficient, say what is missing. "
        "Every factual sentence in the answer must include an inline citation "
        "using this exact format: [source.pdf, p. page_number]. "
        "Do not cite sources that are not in the retrieved context."
    )


def build_context(chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] Source: {chunk['source']}, page {chunk['page']}",
                    f"Similarity: {chunk['score']}",
                    chunk["text"],
                ]
            )
        )

    return "\n\n---\n\n".join(context_blocks)


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    return f"Question: {question}\n\nRetrieved context:\n{build_context(chunks)}"


def build_chat_messages(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_instructions()},
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]


def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: dict[tuple[str, int], dict[str, Any]] = {}

    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        citation = citations.setdefault(
            key,
            {
                "label": f"[{chunk['source']}, p. {chunk['page']}]",
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_ids": [],
            },
        )
        citation["chunk_ids"].append(chunk["id"])

    return list(citations.values())


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
            context=ssl_context,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach LLM endpoint: {error.reason}") from error


def extract_openai_text(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return response["output_text"]

    text_parts = [
        content.get("text", "")
        for item in response.get("output", [])
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    ]
    if text_parts:
        return "\n".join(text_parts)

    raise RuntimeError(f"Could not find text in OpenAI response: {response}")


def call_openai(question: str, chunks: list[dict[str, Any]]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY before using OpenAI")

    response = post_json(
        OPENAI_URL,
        payload={
            "model": OPENAI_MODEL,
            "instructions": build_instructions(),
            "input": build_user_prompt(question, chunks),
        },
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    )
    return extract_openai_text(response)


def call_ollama(messages: list[dict[str, str]]) -> str:
    response = post_json(
        OLLAMA_URL,
        payload={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        },
    )
    return response["message"]["content"]


def call_openai_compatible(messages: list[dict[str, str]]) -> str:
    if not OPENAI_COMPATIBLE_API_KEY:
        raise RuntimeError("Set LLM_API_KEY before using --provider openai-compatible")

    response = post_json(
        OPENAI_COMPATIBLE_URL,
        payload={
            "model": OPENAI_COMPATIBLE_MODEL,
            "messages": messages,
            "temperature": 0.2,
        },
        headers={"Authorization": f"Bearer {OPENAI_COMPATIBLE_API_KEY}"},
    )
    return response["choices"][0]["message"]["content"]


def call_llm(provider: str, question: str, chunks: list[dict[str, Any]]) -> str:
    messages = build_chat_messages(question, chunks)

    if provider == "openai":
        return call_openai(question, chunks)
    if provider == "ollama":
        return call_ollama(messages)
    if provider == "openai-compatible":
        return call_openai_compatible(messages)

    raise ValueError(f"Provider must be one of: {', '.join(PROVIDERS)}")


def build_response(
    question: str,
    chunks: list[dict[str, Any]],
    provider: str,
    top_k: int,
    answer: str | None = None,
    include_chunks: bool = False,
    include_prompt: bool = False,
) -> dict[str, Any]:
    response = {
        "question": question,
        "provider": provider,
        "model": provider_model(provider),
        "top_k": top_k,
        "answer": answer,
        "citations": build_citations(chunks),
    }

    if include_chunks:
        response["chunks"] = chunks
    if include_prompt:
        response["messages"] = build_chat_messages(question, chunks)

    return response


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    provider: str = DEFAULT_PROVIDER,
    dry_run: bool = False,
    include_chunks: bool = False,
) -> dict[str, Any]:
    chunks = retrieve_chunks(question, top_k)

    if dry_run:
        return build_response(
            question=question,
            chunks=chunks,
            provider=provider,
            top_k=top_k,
            include_chunks=True,
            include_prompt=True,
        )

    answer = call_llm(provider, question, chunks)
    return build_response(
        question=question,
        chunks=chunks,
        provider=provider,
        top_k=top_k,
        answer=answer,
        include_chunks=include_chunks,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a question with RAG citations.")
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show retrieved chunks and prompt without calling the LLM.",
    )
    parser.add_argument(
        "--include-chunks",
        action="store_true",
        help="Include raw retrieved chunks in the final JSON response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_question(
        question=args.question,
        top_k=args.top_k,
        provider=args.provider,
        dry_run=args.dry_run,
        include_chunks=args.include_chunks,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
