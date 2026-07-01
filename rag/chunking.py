import re

from rag.models import Chunk, Document


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_chunk_start(text: str, start: int, end: int) -> int:
    """Pomjera pocetak chunk-a da prikaz ne pocinje usred recenice."""

    if start == 0:
        return start

    search_limit = min(start + 160, end)
    sentence_starts = [
        text.find(". ", start, search_limit),
        text.find("! ", start, search_limit),
        text.find("? ", start, search_limit),
        text.find("\n", start, search_limit),
    ]
    sentence_starts = [position for position in sentence_starts if position != -1]

    if sentence_starts:
        clean_start = min(sentence_starts) + 1
        while clean_start < end and text[clean_start].isspace():
            clean_start += 1
        return clean_start

    next_space = text.find(" ", start, min(start + 40, end))
    if next_space != -1:
        return next_space + 1

    return start


def chunk_document(
    document: Document,
    chunk_size: int = 900,
    overlap: int = 180,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size mora biti veci od nule")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap mora biti >= 0 i manji od chunk_size")

    text = normalize_text(document.text)
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            sentence_end = max(
                text.rfind(".", start, end),
                text.rfind("!", start, end),
                text.rfind("?", start, end),
                text.rfind("\n", start, end),
            )
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1

        display_start = clean_chunk_start(text, start, end)
        chunk_text = text[display_start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    id=f"{document.path}::chunk-{index}",
                    document_path=document.path,
                    text=chunk_text,
                    start=display_start,
                    end=end,
                )
            )
            index += 1

        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 900,
    overlap: int = 180,
) -> list[Chunk]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap))
    return chunks


