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
    # Izdvajamo rijeci iz teksta i pretvaramo ih u mala slova
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HashingEmbeddingModel:
    
    name = "local-hashing-embedding"

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        # Svaki tekst dobija svoj vektor fiksne duzine
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [
                f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
            ]
            counts = Counter(features)

            for token, count in counts.items():
                # Hash odredjuje poziciju u vektoru na koju se dodaje vrijednost
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign * (1.0 + math.log(count))

        return l2_normalize(vectors)


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
    # Normalizacija pomaze da se vektori porede preko slicnosti
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def create_embedding_model(kind: str, model_name: str | None = None) -> EmbeddingModel:
    # Bira koji embedding model ce se koristiti
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


