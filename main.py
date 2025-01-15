import math
from collections import Counter

def tokenize(document):
    """Tokenizuje dokument na pojedyncze słowa."""
    return document.lower().split()

def compute_term_frequencies(documents):
    """Oblicza częstotliwości terminów w dokumentach."""
    return [Counter(tokenize(doc)) for doc in documents]

def compute_collection_frequencies(documents):
    """Oblicza częstotliwość każdego terminu w całej kolekcji dokumentów."""
    collection_counter = Counter()
    for doc in documents:
        collection_counter.update(tokenize(doc))
    return collection_counter

def query_likelihood(doc_freqs, coll_freqs, query, total_terms, lam=0.5):
    """Oblicza prawdopodobieństwo generowania zapytania przez dokument."""
    query_tokens = tokenize(query)
    score = 0
    for token in query_tokens:
        doc_prob = doc_freqs.get(token, 0) / total_terms
        coll_prob = coll_freqs.get(token, 0) / sum(coll_freqs.values())
        smoothed_prob = lam * doc_prob + (1 - lam) * coll_prob
        if smoothed_prob > 0:
            score += math.log(smoothed_prob)
    return score

def rank_documents(documents, query, lam=0.5):
    """Sortuje dokumenty według prawdopodobieństwa generowania zapytania."""
    doc_freqs_list = compute_term_frequencies(documents)
    coll_freqs = compute_collection_frequencies(documents)
    total_terms_list = [sum(freq.values()) for freq in doc_freqs_list]

    scores = []
    for i, doc_freqs in enumerate(doc_freqs_list):
        score = query_likelihood(doc_freqs, coll_freqs, query, total_terms_list[i], lam)
        scores.append((score, i))

    # Sortowanie malejąco według wyniku, zachowując kolejność przy równych wynikach
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [index for _, index in scores]

if __name__ == "__main__":
    # Dane wejściowe
    n = int(input().strip())
    documents = [input().strip() for _ in range(n)]
    query = input().strip()

    # Ranking dokumentów
    ranked_indices = rank_documents(documents, query)
    print(ranked_indices)
