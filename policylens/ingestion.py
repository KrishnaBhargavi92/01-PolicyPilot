import json
import re
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNKS_PATH, DOCS_DIR, GENERATED_DIR


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("; "))
            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    with pdf_path.open("rb") as file:
        reader = PdfReader(file)
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"page": page_number, "text": text})

        return pages


def extract_documents(
    docs_dir: Path = DOCS_DIR,
    output_dir: Path = GENERATED_DIR,
    chunks_path: Path = CHUNKS_PATH,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(docs_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    chunk_count = 0

    with chunks_path.open("w", encoding="utf-8") as chunks_file:
        for pdf_path in pdf_paths:
            pages = extract_pdf_pages(pdf_path)
            document_text = "\n\n".join(page["text"] for page in pages)
            text_path = output_dir / f"{pdf_path.stem}.txt"
            text_path.write_text(document_text, encoding="utf-8")

            for page in pages:
                for chunk_index, chunk in enumerate(chunk_text(page["text"]), start=1):
                    chunk_count += 1
                    record = {
                        "id": f"{pdf_path.stem}:page-{page['page']}:chunk-{chunk_index}",
                        "source": pdf_path.name,
                        "page": page["page"],
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                    chunks_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Extracted {pdf_path.name} -> {text_path.name}")

    print(f"Created {chunks_path} with {chunk_count} chunks")
    return chunk_count


def main() -> None:
    extract_documents()


if __name__ == "__main__":
    main()
