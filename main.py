import math
from collections import defaultdict

def preprocess_text(text):
    return text.lower().split()

def calculate_tfidf(corpus):
    term_frequencies = []
    document_frequencies = defaultdict(int)

    for doc in corpus:
        tf = defaultdict(int)
        tokens = preprocess_text(doc)
        for token in tokens:
            tf[token] += 1
        term_frequencies.append(tf)

        for token in set(tokens):
            document_frequencies[token] += 1

    num_docs = len(corpus)
    tfidf_vectors = []

    for tf in term_frequencies:
        tfidf = {}
        for term, freq in tf.items():
            idf = math.log((num_docs + 1) / (document_frequencies[term] + 1)) + 1
            tfidf[term] = freq * idf
        tfidf_vectors.append(tfidf)

    return tfidf_vectors

def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def classify_knn(training_docs, labels, test_doc, k):
    tfidf_training = calculate_tfidf(training_docs)
    tfidf_test = calculate_tfidf([test_doc])[0]

    similarities = []
    for i, train_vector in enumerate(tfidf_training):
        similarity = cosine_similarity(train_vector, tfidf_test)
        similarities.append((similarity, labels[i]))

    similarities.sort(key=lambda x: (-x[0], x[1]))

    top_k = similarities[:k]

    label_count = defaultdict(int)
    for _, label in top_k:
        label_count[label] += 1

    return max(label_count.items(), key=lambda x: (x[1], x[0]))[0]

if __name__ == "__main__":
    num_training_docs = int(input())
    training_documents = [input().strip() for _ in range(num_training_docs)]
    training_labels = list(map(int, input().strip().split()))
    test_document = input().strip()
    k_neighbors = int(input())

    result = classify_knn(training_documents, training_labels, test_document, k_neighbors)
    print([result])
