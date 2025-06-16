from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
texts = ["I love this product." for _ in range(50)] + \
        ["This is the worst purchase." for _ in range(50)]
labels = ["good"] * 50 + ["bad"] * 50

# TF-IDF
vectorizer = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

# Text preprocessing function
def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

# Example
example_text = ["Terrible item."]
print("Vectorized sample:", text_preprocess_vectorize(example_text, vectorizer).toarray())
