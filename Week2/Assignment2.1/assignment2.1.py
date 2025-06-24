import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample corpus
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

# Manual TF-IDF implementation 
def compute_tf_idf(corpus):
    N = len(corpus)
    vocab = set()
    docs = []

    # Tokenize and count word frequencies
    for doc in corpus:
        words = doc.split()
        vocab.update(words)
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        docs.append(freq)

    vocab = sorted(vocab)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in docs if word in doc)
        idf[word] = math.log(N / (1 + df))  # Smoothing with +1 to avoid division by 0

    tfidf_matrix = []
    for freq in docs:
        tfidf = []
        for word in vocab:
            tf = freq.get(word, 0)
            tfidf.append(tf * idf[word])
        tfidf_matrix.append(tfidf)

    return vocab, tfidf_matrix

# Compute manual TF-IDF
vocab, tfidf_matrix = compute_tf_idf(corpus)

print("Manual TF-IDF Matrix:")
print("Vocabulary:", vocab)
for row in tfidf_matrix:
    print(row)

#  Using scikit-learn CountVectorizer 
print("\nCountVectorizer Matrix:")
cv = CountVectorizer()
cv_matrix = cv.fit_transform(corpus).toarray()
print("Vocabulary:", cv.get_feature_names_out())
print(cv_matrix)

#  Using scikit-learn TfidfVectorizer 
print("\nTfidfVectorizer Matrix:")
tv = TfidfVectorizer()
tv_matrix = tv.fit_transform(corpus).toarray()
print("Vocabulary:", tv.get_feature_names_out())
print(tv_matrix)
