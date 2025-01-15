import math
from collections import Counter

def tokenize(text):
    """Converts text to lowercase and splits into words without punctuation."""
    return text.lower().replace('.', '').split()

def compute_document_scores(documents, query, smoothing=0.5):
    query_tokens = tokenize(query)
    tokenized_docs = [tokenize(doc) for doc in documents]

    # Flatten the corpus and calculate word frequencies
    corpus_tokens = [word for doc in tokenized_docs for word in doc]
    corpus_freq = Counter(corpus_tokens)
    corpus_size = len(corpus_tokens)

    scores = []

    for doc_index, doc_tokens in enumerate(tokenized_docs):
        doc_freq = Counter(doc_tokens)
        doc_size = len(doc_tokens)

        log_probability = 0

        for word in query_tokens:
            prob_word_doc = doc_freq[word] / doc_size if doc_size > 0 else 0
            prob_word_corpus = corpus_freq[word] / corpus_size if corpus_size > 0 else 0
            smoothed_prob = smoothing * prob_word_doc + (1 - smoothing) * prob_word_corpus

            if smoothed_prob > 0:
                log_probability += math.log(smoothed_prob)

        scores.append((log_probability, doc_index))

    # Sort by log probability descending, then by index ascending
    scores.sort(key=lambda item: (-item[0], item[1]))

    return [index for _, index in scores]

if __name__ == "__main__":
    num_docs = int(input())
    docs = [input().strip() for _ in range(num_docs)]
    search_query = input().strip()

    ranked_indices = compute_document_scores(docs, search_query)
    print(ranked_indices)
