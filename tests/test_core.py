from rag.chunking import chunk_document
from rag.embeddings import HashingEmbeddingModel
from rag.models import Document
from rag.vector_store import NumpyVectorStore


def test_chunk_document_creates_chunks():
    document = Document(path="test.txt", text="Ovo je prva recenica. " * 80)
    chunks = chunk_document(document, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)


def test_hashing_embedding_shape():
    model = HashingEmbeddingModel(dimension=64)
    embeddings = model.encode(["RAG sistem", "semanticka pretraga"])
    assert embeddings.shape == (2, 64)


def test_numpy_vector_store_cosine_search():
    model = HashingEmbeddingModel(dimension=128)
    chunks = chunk_document(
        Document(
            path="rag.txt",
            text="RAG koristi dokumente za odgovaranje. A star je algoritam pretrage puta.",
        ),
        chunk_size=45,
        overlap=0,
    )
    embeddings = model.encode([chunk.text for chunk in chunks])
    store = NumpyVectorStore(chunks, embeddings)
    query = model.encode(["dokumenti i odgovaranje"])[0]
    results = store.search(query, top_k=1, metric="cosine")
    assert len(results) == 1
    assert "RAG" in results[0].chunk.text or "dokumente" in results[0].chunk.text
