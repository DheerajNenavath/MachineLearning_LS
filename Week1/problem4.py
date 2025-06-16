import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create dataset
positive_reviews = [
    "This movie was great!",
    "I really enjoyed the film.",
    "Amazing performance and direction.",
    "One of the best movies I’ve seen!",
    "Absolutely loved it!",
    "Fantastic storytelling.",
    "Great acting by the cast.",
    "A masterpiece!",
    "Brilliant cinematography.",
    "Highly recommended film."
] * 5  

negative_reviews = [
    "I hated this movie.",
    "The plot was boring.",
    "Terrible acting.",
    "Not worth watching.",
    "Very disappointing.",
    "Worst movie ever.",
    "I fell asleep.",
    "Complete waste of time.",
    "Unwatchable and dull.",
    "Bad script and direction."
] * 5  


reviews = positive_reviews + negative_reviews
sentiments = ["positive"] * 50 + ["negative"] * 50
df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})


# Tokenize
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict single review
def predict_review_sentiment(model, vectorizer, review):
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    print(f"Probabilities: {dict(zip(model.classes_, prob))}")
    return pred


print("Example prediction:", predict_review_sentiment(model, vectorizer, "Worst movie that I saw in recent time"))
