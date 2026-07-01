from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np


TOKEN_PATTERN = re.compile(r"[\wčćžšđČĆŽŠĐ]+", re.UNICODE)


class EmbeddingModel(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Vraca matricu oblika (broj_tekstova, dimenzija_embeddinga)."""


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HashingEmbeddingModel:
    """Lokalni embedding model koji radi bez interneta.

    Model koristi hashing trik, unigram/bigram tokene, IDF tezine i L2
    normalizaciju. IDF smanjuje uticaj cestih izraza i pojacava rijetke,
    specificne termine kao sto su nazivi algoritama ili biblioteka.
    """

    name = "local-hashing-idf-embedding"

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.idf: dict[str, float] = {}
        self.default_idf = 1.0
        self.is_fitted = False

    def fit(self, texts: list[str]) -> None:
        document_frequency: Counter[str] = Counter()

        for text in texts:
            features = set(self._extract_features(text))
            document_frequency.update(features)

        document_count = max(len(texts), 1)
        self.idf = {
            feature: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for feature, frequency in document_frequency.items()
        }
        self.default_idf = math.log(1 + document_count) + 1.0
        self.is_fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            self.fit(texts)

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            counts = Counter(self._extract_features(text))

            for token, count in counts.items():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                tf = 1.0 + math.log(count)
                vectors[row, index] += sign * tf * self.idf.get(token, self.default_idf)

        return l2_normalize(vectors)

    def _extract_features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams


class SentenceTransformerEmbeddingModel:
    """Wrapper oko biblioteke sentence-transformers."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "SentenceTransformer model se ne moze ucitati. "
                "Najcesci uzrok na Windowsu je neispravna PyTorch instalacija ili DLL greska. "
                "Provjerite komandom: python -c \"import torch\". "
                "Dok se PyTorch ne popravi, koristite: "
                "python main.py --embedding local --vector-store faiss --demo"
            ) from exc

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def create_embedding_model(kind: str, model_name: str | None = None) -> EmbeddingModel:
    if kind == "local":
        return HashingEmbeddingModel()
    if kind == "sentence-transformer":
        try:
            return SentenceTransformerEmbeddingModel(
                model_name or "paraphrase-multilingual-MiniLM-L12-v2"
            )
        except RuntimeError as exc:
            print(f"Upozorenje: {exc}")
            print("Koristi se lokalni embedding model kao fallback.")
            return HashingEmbeddingModel()
    raise ValueError(f"Nepoznat embedding model: {kind}")
