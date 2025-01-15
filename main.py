import math
from collections import Counter

def tokenize_text(input_text):
    return input_text.lower().split()

def compute_tf_idf(doc_list):
    tf_idf_matrix = []
    term_doc_freq = Counter()
    doc_count = len(doc_list)

    for doc in doc_list:
        term_freq = Counter(tokenize_text(doc))
        tf_idf_matrix.append(term_freq)
        for term in term_freq:
            term_doc_freq[term] += 1

    for doc_index, term_freq in enumerate(tf_idf_matrix):
        tf_idf_matrix[doc_index] = {term: (freq * math.log((doc_count + 1) / (term_doc_freq[term] + 1)) + 1)
                                    for term, freq in term_freq.items()}

    return tf_idf_matrix

def compute_similarity(vec_a, vec_b):
    intersection = set(vec_a.keys()).union(vec_b.keys())
    dot_product = sum(vec_a.get(term, 0) * vec_b.get(term, 0) for term in intersection)
    magnitude_a = math.sqrt(sum(value ** 2 for value in vec_a.values()))
    magnitude_b = math.sqrt(sum(value ** 2 for value in vec_b.values()))
    if not magnitude_a or not magnitude_b:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def knn_classifier(train_set, train_labels, target_doc, k):
    tf_idf_train = compute_tf_idf(train_set)
    target_vector = compute_tf_idf([target_doc])[0]

    distances = [(compute_similarity(train_vec, target_vector), train_labels[i])
                 for i, train_vec in enumerate(tf_idf_train)]
    distances.sort(key=lambda x: (-x[0], x[1]))

    top_k_labels = [label for _, label in distances[:k]]
    return 1 if top_k_labels.count(1) >= top_k_labels.count(0) else 0

if __name__ == "__main__":
    try:
        num_training_docs = int(input("Enter number of training documents: "))
        training_documents = [input(f"Document {i + 1}: ").strip() for i in range(num_training_docs)]
        
        training_labels = input("Enter space-separated binary labels (e.g., '1 0 1'): ").strip().split()
        if not all(label in {"0", "1"} for label in training_labels):
            raise ValueError("Labels must be binary (0 or 1).")
        training_labels = list(map(int, training_labels))
        
        test_document = input("Enter test document: ").strip()
        k_neighbors = int(input("Enter number of neighbors (k): "))

        result = classify_knn(training_documents, training_labels, test_document, k_neighbors)
        print(result)

    except ValueError as e:
        print(f"Input error: {e}")

