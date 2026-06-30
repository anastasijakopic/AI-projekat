import re

from rag.embeddings import tokenize
from rag.models import SearchResult


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def extract_relevant_context(question: str, text: str, max_chars: int = 900) -> str:
    context = text.strip()
    if len(context) <= max_chars:
        sentences = split_sentences(context)
        focused = focus_sentences(question, sentences)
        return focused if focused else context

    sentences = split_sentences(context)
    focused = focus_sentences(question, sentences)
    if focused:
        return focused[:max_chars].rsplit(" ", 1)[0] + ("..." if len(focused) > max_chars else "")

    return context[:max_chars].rsplit(" ", 1)[0] + "..."


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def focus_sentences(question: str, sentences: list[str]) -> str:
    if not sentences:
        return ""

    query_tokens = {token for token in tokenize(question) if len(token) > 3}
    if not query_tokens:
        return " ".join(sentences[:3])

    best_index = 0
    best_score = -1
    for index, sentence in enumerate(sentences):
        sentence_tokens = set(tokenize(sentence))
        score = len(query_tokens & sentence_tokens)
        if score > best_score:
            best_score = score
            best_index = index

    start = max(0, best_index - 1)
    end = min(len(sentences), best_index + 2)
    return " ".join(sentences[start:end])


def build_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "Nisam pronasao relevantan kontekst u dokumentima."

    best = results[0]
    context = extract_relevant_context(question, best.chunk.text)

    return (
        f"Pitanje: {question}\n\n"
        "Odgovor na osnovu pronadjenog konteksta:\n"
        f"{context}\n\n"
        f"Izvor: {best.chunk.document_path}\n"
        f"Metrika: {best.metric}, skor: {best.score:.4f}"
    )

