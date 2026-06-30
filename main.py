import argparse
import sys
from pathlib import Path

from rag.embeddings import create_embedding_model
from rag.rag_pipeline import RagPipeline
from rag.response import format_results

if hasattr(sys.stdout, "reconfigure"):
    # Omogucava ispravan ispis slova u konzoli
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_DOCS = Path("data/documents")
DEMO_QUESTIONS = [
    "Sta je RAG sistem?",
    "Kako se koristi cosine similarity?",
    "Koja je razlika izmedju fine-tuninga i RAG-a?",
]


def build_parser() -> argparse.ArgumentParser:
    # Definise argumente koji se mogu proslijediti iz komandne linije
    parser = argparse.ArgumentParser(
        description="RAG sistem sa semantickom pretragom nad lokalnim dokumentima."
    )
    parser.add_argument("--docs", default=str(DEFAULT_DOCS), help="Folder sa TXT/PDF dokumentima.")
    parser.add_argument("--question", help="Pitanje koje se postavlja sistemu.")
    parser.add_argument("--demo", action="store_true", help="Pokreni demo pitanja.")
    parser.add_argument("--compare", action="store_true", help="Uporedi cosine i Euclidean metriku.")
    parser.add_argument("--top-k", type=int, default=3, help="Broj rezultata pretrage.")
    parser.add_argument(
        "--embedding",
        choices=["local", "sentence-transformer"],
        default="local",
        help="Embedding model.",
    )
    parser.add_argument("--model", default=None, help="Naziv SentenceTransformer modela.")
    parser.add_argument(
        "--vector-store",
        choices=["numpy", "faiss"],
        default="numpy",
        help="Backend za vector store.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Metrika za pretragu.",
    )
    return parser


