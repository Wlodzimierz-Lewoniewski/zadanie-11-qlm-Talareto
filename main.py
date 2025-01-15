import math
from collections import Counter

def tokenize(text):
    """Konwertuje tekst na małe litery i dzieli na słowa, usuwając znaki interpunkcyjne."""
    return text.lower().replace('.', '').split()

def compute_document_frequencies(documents):
    """Oblicza częstotliwości terminów dla każdego dokumentu."""
    return [Counter(tokenize(doc)) for doc in documents]

def compute_corpus_frequencies(documents):
    """Oblicza częstotliwości terminów w całym korpusie."""
    corpus_counter = Counter()
    for doc in documents:
        corpus_counter.update(tokenize(doc))
    return corpus_counter

def calculate_query_probability(doc_freqs, corpus_freqs, query_tokens, doc_size, corpus_size, smoothing=0.5):
    """Oblicza logarytmiczne prawdopodobieństwo zapytania dla danego dokumentu."""
    log_prob = 0
    for token in query_tokens:
        prob_in_doc = doc_freqs[token] / doc_size if doc_size > 0 else 0
        prob_in_corpus = corpus_freqs[token] / corpus_size if corpus_size > 0 else 0
        smoothed_prob = smoothing * prob_in_doc + (1 - smoothing) * prob_in_corpus
        if smoothed_prob > 0:
            log_prob += math.log(smoothed_prob)
    return log_prob

def rank_documents(documents, query, smoothing=0.5):
    """Sortuje dokumenty według prawdopodobieństwa zapytania."""
    query_tokens = tokenize(query)
    doc_freqs_list = compute_document_frequencies(documents)
    corpus_freqs = compute_corpus_frequencies(documents)
    corpus_size = sum(corpus_freqs.values())

    scores = []
    for idx, doc_freqs in enumerate(doc_freqs_list):
        doc_size = sum(doc_freqs.values())
        score = calculate_query_probability(doc_freqs, corpus_freqs, query_tokens, doc_size, corpus_size, smoothing)
        scores.append((idx, score))

    # Sortowanie malejąco po wyniku, a następnie rosnąco po indeksie
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [index for index, _ in scores]

if __name__ == "__main__":
    num_docs = int(input("Podaj liczbę dokumentów: "))
    documents = [input(f"Dokument {i + 1}: ").strip() for i in range(num_docs)]
    query = input("Podaj zapytanie: ").strip()

    ranked_indices = rank_documents(documents, query)
    print("Posortowane indeksy dokumentów:", ranked_indices)
