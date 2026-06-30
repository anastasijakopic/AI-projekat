import re

from rag.embeddings import tokenize
from rag.models import SearchResult


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def extract_relevant_context(question: str, text: str, max_chars: int = 900) -> str:
    # Izdvaja dio teksta koji je najkorisniji za odgovor
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
    # Dijeli tekst na recenice i uklanja visak razmaka
    text = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def focus_sentences(question: str, sentences: list[str]) -> str:
    # Pronalazi recenice koje imaju najvise zajednickih rijeci sa pitanjem
    if not sentences:
        return ""
    # Kratke rijeci preskacemo jer uglavnom nisu informativne
    query_tokens = {token for token in tokenize(question) if len(token) > 3}
    if not query_tokens:
        return " ".join(sentences[:3])

    # Trazimo recenicu koja se najvise poklapa sa pitanjem
    best_index = 0
    best_score = -1
    for index, sentence in enumerate(sentences):
        sentence_tokens = set(tokenize(sentence))
        score = len(query_tokens & sentence_tokens)
        if score > best_score:
            best_score = score
            best_index = index

    # Uzimamo najbolju recenicu
    start = max(0, best_index - 1)
    end = min(len(sentences), best_index + 2)
    return " ".join(sentences[start:end])


def build_answer(question: str, results: list[SearchResult]) -> str:
    # Pravi konacan odgovor na osnovu najboljeg pronadjenog rezultata
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

def format_results(results: list[SearchResult]) -> str:
    # Formatira rezultate pretrage za ispis u konzoli
    lines = []
    for index, result in enumerate(results, start=1):
        text = result.chunk.text.replace("\n", " ")
        if len(text) > 220:
            text = text[:220].rsplit(" ", 1)[0] + "..."
        lines.append(
            f"{index}. score={result.score:.4f} | file={result.chunk.document_path}\n"
            f"   {text}"
        )
    return "\n".join(lines)

