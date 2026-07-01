from __future__ import annotations

from pathlib import Path

from rag.chunking import chunk_documents
from rag.document_loader import load_documents
from rag.embeddings import EmbeddingModel
from rag.models import SearchResult
from rag.response import build_answer
from rag.vector_store import create_vector_store


class RagPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_backend: str = "numpy",
        metric: str = "cosine",
        chunk_size: int = 900,
        overlap: int = 180,
    ):
        self.embedding_model = embedding_model
        self.vector_backend = vector_backend
        self.metric = metric
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.store = None

    def index_folder(self, folder: str | Path) -> None:
        # Prva faza RAG-a: ucitavanje dokumenata iz lokalnog foldera.
        documents = load_documents(folder)

        # Dokumenti se dijele na manje dijelove da bi pretraga bila preciznija.
        self.chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        # Svaki chunk se pretvara u embedding vektor i indeksira u vector store.
        texts = [chunk.text for chunk in self.chunks]
        if hasattr(self.embedding_model, "fit"):
            self.embedding_model.fit(texts)
        embeddings = self.embedding_model.encode(texts)
        self.store = create_vector_store(
            self.chunks,
            embeddings,
            backend=self.vector_backend,
            metric=self.metric,
        )

    def ask(
        self,
        question: str,
        top_k: int = 3,
        metric: str | None = None,
    ) -> tuple[str, list[SearchResult]]:
        if self.store is None:
            raise RuntimeError("Prvo pozovite index_folder().")

        # Druga faza RAG-a: pitanje se pretvara u embedding i poredi sa chunkovima.
        used_metric = metric or self.metric
        query_embedding = self.embedding_model.encode([question])[0]
        results = self.store.search(query_embedding, top_k=top_k, metric=used_metric)
        return build_answer(question, results), results

    def compare_metrics(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, list[SearchResult]]:
        # Ista pitanja se pretrazuju kroz dvije metrike radi poredjenja rezultata.
        return {
            "cosine": self.ask(question, top_k=top_k, metric="cosine")[1],
            "euclidean": self.ask(question, top_k=top_k, metric="euclidean")[1],
        }


