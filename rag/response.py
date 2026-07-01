import re

from rag.embeddings import tokenize
from rag.models import SearchResult


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
COSINE_CONFIDENCE_THRESHOLD = 0.18
MIN_KEYWORD_OVERLAP = 0.20

BASIC_STOPWORDS = {
    "sta", "sto", "koja", "koji", "koje", "kako", "zasto", "kada", "gdje",
    "cemu", "nije", "jeste", "ima", "nema", "odnosu", "pogledu",
    "znaci", "cijena",
}

SPECIFIC_STOPWORDS = BASIC_STOPWORDS


def fold_accents(text: str) -> str:
    replacements = str.maketrans({
        "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
        "Č": "c", "Ć": "c", "Š": "s", "Ž": "z", "Đ": "dj",
    })
    return text.translate(replacements)


def normalize_keyword(token: str) -> str:
    token = fold_accents(token.lower())
    for suffix in ("skog", "skom", "osti", "anje", "enje", "ovima", "ama", "ima", "om", "em", "og", "oj", "u", "a", "e", "i"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def make_keyword_set(text: str, stopwords: set[str]) -> set[str]:
    keywords = set()
    for token in tokenize(fold_accents(text)):
        normalized = normalize_keyword(token)
        if len(normalized) >= 3 and normalized not in stopwords:
            keywords.add(normalized)
    return keywords


def keyword_set(text: str) -> set[str]:
    return make_keyword_set(text, SPECIFIC_STOPWORDS)


def focus_keyword_set(text: str) -> set[str]:
    return make_keyword_set(text, BASIC_STOPWORDS)


def token_matches(query_token: str, text_tokens: set[str]) -> bool:
    for text_token in text_tokens:
        if query_token == text_token:
            return True
        if len(query_token) >= 5 and len(text_token) >= 5:
            if query_token[:5] == text_token[:5]:
                return True
    return False


def overlap_ratio(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    matches = sum(1 for token in query_tokens if token_matches(token, text_tokens))
    return matches / len(query_tokens)


def keyword_overlap(question: str, text: str) -> float:
    return overlap_ratio(keyword_set(question), keyword_set(text))


def focus_overlap(question: str, text: str) -> float:
    return overlap_ratio(focus_keyword_set(question), focus_keyword_set(text))


def best_sentence_window(question: str, sentences: list[str], window_size: int = 3) -> tuple[float, int, int]:
    if not sentences:
        return 0.0, 0, 0

    query_focus = focus_keyword_set(question)
    query_specific = keyword_set(question)
    best_score = -1.0
    best_start = 0
    best_end = 1

    windows = []
    for start in range(len(sentences)):
        for end in range(start + 1, min(len(sentences), start + window_size) + 1):
            windows.append((start, end))
    for index in range(1, len(sentences)):
        windows.append((index - 1, index + 1))

    for start, end in windows:
        window_text = " ".join(sentences[start:end])
        score = focus_overlap(question, window_text)
        score += 0.60 * overlap_ratio(query_specific, keyword_set(window_text))

        if score > best_score:
            best_score = score
            best_start = start
            best_end = end

    return best_score, best_start, best_end



def starts_mid_sentence(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].islower()

def combined_relevance(question: str, result: SearchResult) -> float:
    sentences = split_sentences(result.chunk.text)
    window_score, _, _ = best_sentence_window(question, sentences)
    specific_overlap = keyword_overlap(question, result.chunk.text)

    fragment_penalty = 0.25 if starts_mid_sentence(result.chunk.text) else 0.0

    if result.metric == "cosine":
        return result.score + 0.25 * specific_overlap + 0.35 * window_score - fragment_penalty
    if result.metric == "euclidean":
        return -result.score + 0.25 * specific_overlap + 0.35 * window_score - fragment_penalty
    return specific_overlap + window_score - fragment_penalty


def select_best_result(question: str, results: list[SearchResult]) -> SearchResult:
    return max(results, key=lambda result: combined_relevance(question, result))


def is_confident(question: str, result: SearchResult) -> bool:
    specific_overlap = keyword_overlap(question, result.chunk.text)
    window_score, _, _ = best_sentence_window(question, split_sentences(result.chunk.text))

    if result.metric == "cosine":
        return (
            result.score >= COSINE_CONFIDENCE_THRESHOLD
            or specific_overlap >= MIN_KEYWORD_OVERLAP
            or window_score >= MIN_KEYWORD_OVERLAP
        )
    return specific_overlap >= MIN_KEYWORD_OVERLAP or window_score >= MIN_KEYWORD_OVERLAP


def extract_relevant_context(question: str, text: str, max_chars: int = 900) -> str:
    context = text.strip()
    sentences = split_sentences(context)
    focused = focus_sentences(question, sentences)

    if focused:
        if len(focused) > max_chars:
            return focused[:max_chars].rsplit(" ", 1)[0] + "..."
        return focused

    if len(context) > max_chars:
        return context[:max_chars].rsplit(" ", 1)[0] + "..."
    return context


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def focus_sentences(question: str, sentences: list[str]) -> str:
    if not sentences:
        return ""

    query_focus = focus_keyword_set(question)
    if not query_focus:
        return " ".join(sentences[:3])

    score, start, end = best_sentence_window(question, sentences)
    if score == 0:
        return " ".join(sentences[:2])

    while end - start > 1 and focus_overlap(question, sentences[start]) == 0:
        start += 1

    return " ".join(sentences[start:end])


def build_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "Nisam pronasao relevantan kontekst u dokumentima."

    best = select_best_result(question, results[:3])
    if not is_confident(question, best):
        return (
            f"Pitanje: {question}\n\n"
            "Sistem nije pronasao dovoljno pouzdan kontekst u dokumentima. "
            "Pokusajte preformulisati pitanje ili dodati konkretnije kljucne rijeci.\n\n"
            f"Najbolji pronadjeni skor: {best.score:.4f} ({best.metric})"
        )

    context = extract_relevant_context(question, best.chunk.text)

    return (
        f"Pitanje: {question}\n\n"
        "Odgovor na osnovu pronadjenog konteksta:\n"
        f"{context}\n\n"
        f"Izvor: {best.chunk.document_path}\n"
        f"Metrika: {best.metric}, skor: {best.score:.4f}"
    )


def format_results(results: list[SearchResult]) -> str:
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