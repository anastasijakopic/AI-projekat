from pathlib import Path

from rag.models import Document


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Za PDF dokumente instalirajte pypdf: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = load_txt(path)
    elif suffix == ".pdf":
        text = load_pdf(path)
    else:
        raise ValueError(f"Nepodrzan format dokumenta: {path.suffix}")

    return Document(path=str(path), text=text.strip())


def load_documents(folder: str | Path) -> list[Document]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder ne postoji: {folder_path}")

    documents = []
    for path in sorted(folder_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            document = load_document(path)
            if document.text:
                documents.append(document)

    if not documents:
        raise RuntimeError(
            f"Nema TXT/PDF dokumenata za indeksiranje u folderu: {folder_path}"
        )
    return documents
