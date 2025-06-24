# Assignment 2.1: Manual TF-IDF vs Scikit-learn Vectorizers

## Objective
This task demonstrates a manual implementation of the TF-IDF algorithm and compares its results to scikit-learn's `CountVectorizer` and `TfidfVectorizer`.

---

## Corpus Used
```python
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]
```

---

## Implementation Summary

### Manual TF-IDF
- Tokenizes each document
- Computes term frequency (TF) for each word in each document
- Computes inverse document frequency (IDF) using the formula:
  ```
  idf(word) = log(N / (1 + df(word)))
  ```
- Computes TF-IDF as: `tf * idf`
- Uses simple dictionaries and lists (no external libraries)

### Scikit-learn Methods
- `CountVectorizer` gives raw word counts
- `TfidfVectorizer` computes weighted TF-IDF using normalization and smoothing

---

## Comparison
| Feature            | Manual TF-IDF             | CountVectorizer         | TfidfVectorizer           |
|--------------------|---------------------------|--------------------------|----------------------------|
| Stop words removed | No                        | No                       | No                         |
| Weighting          | TF * IDF (log)            | Raw count               | TF-IDF (log + normalization) |
| Normalization      | No                        | No                       | Yes                        |

---

## Why do common words like "the" have low TF-IDF scores?
- Because they appear in every document, their IDF is low, reducing their importance.
- `TfidfVectorizer` handles this well by assigning low weight to such frequent terms.

---

## Conclusion
- Manual TF-IDF helps understand the math behind the concept.
- Scikit-learn's implementation is more refined (e.g., with normalization and smoothing).
- Words unique to a document have higher TF-IDF values, while common words (like "the") have low values.

---

Feel free to experiment with your own corpus and compare results!
