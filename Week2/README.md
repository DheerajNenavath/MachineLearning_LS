# Assignment 2: SMS Spam Detection and Twitter Sentiment Classification

## Problem 1: SMS Spam Detection using Word2Vec + Logistic Regression

### Objective
Classify SMS messages as "spam" or "ham" using text preprocessing and a pre-trained Word2Vec embedding model.

### Dataset
- **Source:** [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Columns Used:**
  - `Label`: "spam" or "ham"
  - `Message`: SMS text content

### Approach
1. **Text Preprocessing**
   - Lowercasing and tokenizing
   - Removing stop words
2. **Word Embedding**
   - Load pre-trained **Google News Word2Vec** model using `gensim`
   - Each message is converted to a vector by averaging the Word2Vec vectors of its words
3. **Model Training**
   - Train a **Logistic Regression** model on the embedded vectors
4. **Evaluation**
   - Use 80/20 train-test split
   - Measure accuracy on the test set
5. **Prediction Function**
   - Takes a single message and predicts whether it is "spam" or "ham"

### Outcome
- The model performs well with properly preprocessed and vectorized messages.
- Messages with unseen words or noisy formatting may reduce accuracy if not properly cleaned.

---

## Problem 2: Twitter Sentiment Analysis using Word2Vec + Logistic Regression

### Objective
Classify airline-related tweets into sentiments: "positive", "neutral", or "negative".

### Dataset
- **Source:** [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Columns Used:**
  - `airline_sentiment`: target sentiment
  - `text`: the tweet content

### Approach
1. **Text Preprocessing**
   - Convert to lowercase
   - Remove URLs, mentions, hashtags, punctuation
   - Expand contractions (e.g., "don't" → "do not")
   - Lemmatization using `WordNetLemmatizer`
2. **Word Embedding**
   - Use **Google News Word2Vec** to convert tokens to vectors
   - Average vectors to form a fixed-length representation
3. **Model Training**
   - Train a **multiclass Logistic Regression** model on tweet vectors
4. **Evaluation**
   - 80/20 split for training/testing
   - Report overall accuracy
5. **Prediction Function**
   - Accepts a tweet and returns predicted sentiment

### Outcome
- Preprocessing is critical due to noisy nature of tweets.
- Lemmatization and cleaning improves vector quality.
- Logistic Regression works well when embedded vectors capture enough semantic meaning.

---

## Notes
- In both problems, Word2Vec vectors were averaged to form a document-level representation.
- In practice, better results may be achieved using fine-tuned models or deep learning approaches (e.g., BERT).
- Dummy vectors were used during testing; real evaluation requires the actual Word2Vec model.

---

These assignments demonstrate how classic machine learning models can be combined with modern embeddings for effective NLP classification tasks.
