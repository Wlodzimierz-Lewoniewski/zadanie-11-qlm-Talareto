import math
from collections import Counter

def tokenize(text):
    """Tokenizacja tekstu: konwersja do małych liter i podział na słowa."""
    return text.lower().split()

def calculate_document_scores(documents, query, smoothing=0.5):
    """Obliczanie prawdopodobieństw zapytania dla dokumentów."""
    query_tokens = tokenize(query)
    tokenized_documents = [tokenize(doc) for doc in documents]

    # Statystyki korpusu
    corpus_tokens = [token for doc in tokenized_documents for token in doc]
    corpus_frequency = Counter(corpus_tokens)
    total_corpus_tokens = len(corpus_tokens)

    document_scores = []

    for idx, doc_tokens in enumerate(tokenized_documents):
        doc_frequency = Counter(doc_tokens)
        total_doc_tokens = len(doc_tokens)

        log_prob = 0

        for token in query_tokens:
            # Prawdopodobieństwo w dokumencie
            prob_in_doc = doc_frequency[token] / total_doc_tokens if total_doc_tokens > 0 else 0
            # Prawdopodobieństwo w korpusie
            prob_in_corpus = corpus_frequency[token] / total_corpus_tokens if total_corpus_tokens > 0 else 0
            # Wygładzanie
            smoothed_prob = smoothing * prob_in_doc + (1 - smoothing) * prob_in_corpus

            if smoothed_prob > 0:
                log_prob += math.log(smoothed_prob)

        document_scores.append((idx, log_prob))

    # Sortowanie według prawdopodobieństwa malejąco, a następnie po indeksie rosnąco
    document_scores.sort(key=lambda x: (-x[1], x[0]))

    return [idx for idx, _ in document_scores]

if __name__ == "__main__":
    num_documents = int(input())
    docs = [input().strip() for _ in range(num_documents)]
    query = input().strip()

    ranked_indices = calculate_document_scores(docs, query)
    print(ranked_indices)
