from __future__ import annotations

import numpy as np

from rag.embeddings import l2_normalize
from rag.models import Chunk, SearchResult


class NumpyVectorStore:
    """Jednostavan vector store za cosine i Euclidean pretragu."""

    backend_name = "numpy"

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Broj chunkova mora odgovarati broju embeddinga")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.normalized_embeddings = l2_normalize(self.embeddings.copy())

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        metric: str = "cosine",
    ) -> list[SearchResult]:
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if metric == "cosine":
            normalized_query = l2_normalize(query.copy())[0]
            scores = self.normalized_embeddings @ normalized_query
            order = np.argsort(-scores)[:top_k]
            return [
                SearchResult(self.chunks[index], float(scores[index]), metric)
                for index in order
            ]

        if metric == "euclidean":
            distances = np.linalg.norm(self.embeddings - query[0], axis=1)
            order = np.argsort(distances)[:top_k]
            return [
                SearchResult(self.chunks[index], float(distances[index]), metric)
                for index in order
            ]

        raise ValueError(f"Nepodrzana metrika: {metric}")


class FaissVectorStore:
    """FAISS vector store, ako je biblioteka instalirana."""

    backend_name = "faiss"

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, metric: str):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS nije instaliran: pip install faiss-cpu") from exc

        self.chunks = chunks
        self.metric = metric
        self.embeddings = embeddings.astype(np.float32)

        if metric == "cosine":
            vectors = l2_normalize(self.embeddings.copy())
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        elif metric == "euclidean":
            vectors = self.embeddings
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        else:
            raise ValueError(f"Nepodrzana metrika za FAISS: {metric}")

        self.index.add(vectors)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        metric: str | None = None,
    ) -> list[SearchResult]:
        used_metric = metric or self.metric
        query = query_embedding.reshape(1, -1).astype(np.float32)
        if used_metric == "cosine":
            query = l2_normalize(query.copy())

        scores, indices = self.index.search(query, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(SearchResult(self.chunks[int(index)], float(score), used_metric))
        return results


def create_vector_store(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    backend: str = "numpy",
    metric: str = "cosine",
):
    if backend == "numpy":
        return NumpyVectorStore(chunks, embeddings)
    if backend == "faiss":
        return FaissVectorStore(chunks, embeddings, metric)
    raise ValueError(f"Nepoznat vector store backend: {backend}")
