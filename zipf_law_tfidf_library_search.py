from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "machine learning neural networks",
    "neural learning deep learning",
    "deep neural network optimization"
]

# Закон Ципфа
word_counts = Counter(" ".join(texts).split())
sorted_counts = sorted(word_counts.values(), reverse=True)
print("🔹 Закон Ципфа (частоти):", sorted_counts)

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(texts)
weights = dict(zip(vectorizer.get_feature_names_out(), tfidf.toarray().sum(axis=0)))
print("🔹 Ваги слів (TF-IDF):", weights)