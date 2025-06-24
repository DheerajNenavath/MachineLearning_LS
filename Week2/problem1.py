import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gensim.downloader as api

# Load the pre-trained Google News Word2Vec model (this downloads and caches it)
model = api.load("word2vec-google-news-300")

nltk.download('stopwords')

# Load stop words
stop_words = set(stopwords.words('english'))

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Confirm 'spam' and 'ham' are present
print("Label counts:\n", df['Label'].value_counts())

# Preprocess messages
def preprocess_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['Message'].apply(preprocess_text)

# Convert each message to a vector
def vectorize_text(tokens, model, dim=300):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# TEMPORARY: Use dummy random vectors (replace with real model for actual use)
df['vector'] = df['tokens'].apply(lambda x: np.random.rand(300))

# Prepare features and labels
X = np.stack(df['vector'].values)
y = df['Label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 (ham) and 1 (spam)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_message_class(model, w2v_model, message):
    tokens = preprocess_text(message)
    vec = vectorize_text(tokens, w2v_model)
    pred = model.predict([vec])[0]
    return 'spam' if pred == 1 else 'ham'

# Example call (replace model with real Word2Vec if available)
print(predict_message_class(clf, model, "Free entry in 2 a wkly comp to win tickets!"))
