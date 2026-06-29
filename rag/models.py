from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Originalni dokument ucitan sa diska."""

    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """Manji dio dokumenta koji se indeksira u vector store."""

    id: str
    document_path: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SearchResult:
    """Rezultat semanticke pretrage."""

    chunk: Chunk
    score: float
    metric: str
