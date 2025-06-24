import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gensim.downloader as api

# Load the pre-trained Google News Word2Vec model (this downloads and caches it)
model = api.load("word2vec-google-news-300")

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Preprocess each tweet ---
def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^a-z\s]", "", text)             # Remove punctuation and digits
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

df = pd.read_csv("Tweets.csv")[['airline_sentiment', 'text']]
df.dropna(inplace=True)
df['tokens'] = df['text'].apply(preprocess_tweet)


# Convert tweet to average Word2Vec vector
def vectorize_tweet(tokens, model, dim=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# TEMPORARY: Dummy vectors for demonstration 
df['vector'] = df['tokens'].apply(lambda x: np.random.rand(300))  # Replace with actual model

# Prepare data 
X = np.stack(df['vector'].values)
y = df['airline_sentiment']  # Classes: 'positive', 'neutral', 'negative'

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier 
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function 
def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess_tweet(tweet)
    vec = vectorize_tweet(tokens, w2v_model)
    return model.predict([vec])[0]

# Example call (replace model with real Word2Vec if available) 
print(predict_tweet_sentiment(clf, model, "Had a great flight with United!"))
