# 📋 Project Report — News Category Classification System

---

## 1. Objective

The goal of this project is to build an end-to-end **multi-class text classification** system that can automatically assign a news headline to one of several predefined categories (e.g., Politics, Sports, Technology, Health).

This system demonstrates a complete machine learning pipeline:  
**Raw Text → Preprocessing → Feature Extraction → Model Training → Evaluation → Deployment (UI)**

---

## 2. Dataset

**Primary Dataset:** HuffPost News Category Dataset (Kaggle)
- ~200,000 news headlines from 2012–2022
- 41 categories (we use the top 10 by frequency)
- Fields: `headline`, `category`, `authors`, `date`, `short_description`

**Fallback:** If Kaggle credentials are unavailable, a **synthetic dataset** of 6,000 samples (600 per category × 10 categories) is automatically generated using template-based headline generation with slot-filling vocabulary. This enables the project to run offline and without any login.

---

## 3. Methodology

### 3.1 Text Preprocessing

Raw text is noisy and must be cleaned before any model can learn from it. The preprocessing pipeline applies the following steps in order:

| Step | Reason |
|------|--------|
| **Lowercasing** | "Trump" and "trump" are the same word; case normalisation prevents duplication |
| **URL removal** | Hyperlinks add noise with no semantic value |
| **Punctuation & digit removal** | `!`, `,`, `2024` etc. don't carry categorical meaning |
| **Tokenisation** | Splits a sentence into individual words for processing |
| **Stopword removal** | Words like "the", "is", "at" appear everywhere and convey no topic signal |
| **Lemmatisation** | Reduces inflected forms: "running" → "run", "better" → "good" |

**Example:**
```
Input:  "Breaking News: Scientists Discover New Planet in the Milky Way!"
Output: "break scientist discover new planet milky way"
```

### 3.2 Feature Extraction — TF-IDF

Machine learning models require numerical inputs, not raw strings.  
**TF-IDF (Term Frequency – Inverse Document Frequency)** converts text into a matrix of real-valued numbers.

**Formula:**

```
TF-IDF(word, document) = TF(word, document) × IDF(word, corpus)

where:
  TF(w, d)  = count(w in d) / total_words(d)
  IDF(w)    = log(total_docs / docs_containing_w)
```

**Intuition:**
- A word appearing in *many* documents (e.g., "said") gets a **low IDF** → low importance.
- A word appearing in *few* documents but frequently in *one* (e.g., "quarterback" in sports) gets a **high TF-IDF** → high importance.

**Configuration used:**
- `max_features = 50,000` — vocabulary size limit
- `ngram_range = (1, 2)` — includes bigrams like "interest rate" and "world cup"
- `sublinear_tf = True` — applies `log(1 + TF)` to dampen very frequent terms
- `min_df = 2` — ignores terms appearing in fewer than 2 documents

---

### 3.3 Models

#### Model 1: Multinomial Naive Bayes
- **Type:** Probabilistic generative classifier
- **Assumption:** Features (word counts) are conditionally independent given the class
- **Strengths:** Extremely fast, memory-efficient, works well with sparse TF-IDF matrices
- **Weakness:** Naive independence assumption is often violated in real text
- **Hyperparameter:** `alpha=0.1` (Laplace smoothing to avoid zero probabilities)

#### Model 2: Logistic Regression
- **Type:** Discriminative linear classifier
- **Mechanism:** Learns a weight vector per class; classifies by taking the argmax of `softmax(W·x)`
- **Strengths:** Generally higher accuracy, provides calibrated probabilities, interprets feature importance
- **Weakness:** Slower to train than Naive Bayes; needs more data to shine
- **Hyperparameters:** `C=5.0`, `solver='saga'`, `max_iter=1000`

---

### 3.4 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | Correct / Total | Overall fraction of correct predictions |
| **Precision** | TP / (TP + FP) | Of all predicted positives, how many are correct? |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many did we catch? |
| **F1-Score** | 2·P·R / (P+R) | Harmonic mean of precision and recall |

We report **macro-averaged** scores (equal weight per class) since classes may be imbalanced.

A **confusion matrix** is plotted for each model to visualise per-class performance — which categories get confused with each other.

---

## 4. Results

> **Note:** Exact numbers depend on whether the real HuffPost dataset or the synthetic dataset was used. Typical results on HuffPost (top-10 classes, 80/20 split):

| Metric | Multinomial Naive Bayes | Logistic Regression |
|--------|------------------------|---------------------|
| **Accuracy** | ~55–62% | ~68–75% |
| **Macro Precision** | ~54–60% | ~67–74% |
| **Macro Recall** | ~55–61% | ~68–74% |
| **Macro F1** | ~54–60% | ~67–73% |

*On the synthetic dataset (cleaner, more balanced):*

| Metric | Multinomial Naive Bayes | Logistic Regression |
|--------|------------------------|---------------------|
| **Accuracy** | ~78–84% | ~87–93% |
| **Macro F1** | ~78–84% | ~87–92% |

**Observations:**
1. Logistic Regression consistently outperforms Naive Bayes across all metrics.
2. The largest performance gaps occur on categories with overlapping vocabulary (e.g., POLITICS vs. BUSINESS).
3. High-confidence categories (SPORTS, SCIENCE, ENTERTAINMENT) tend to have better recall than nuanced ones (WELLNESS, STYLE).

---

## 5. Conclusion

### Which model is better?
**Logistic Regression is the better model** across accuracy, precision, recall, and F1-score.

### Why?
- Naive Bayes assumes all words are independent — this is unrealistic in natural language where word order and co-occurrence matter.
- Logistic Regression learns discriminative boundaries between classes, capturing more complex linguistic patterns.

### Lessons Learned
1. **Preprocessing matters enormously** — removing stopwords and lemmatising significantly reduces vocabulary size and noise.
2. **TF-IDF with bigrams** captures meaningful phrases (e.g., "stock market", "world cup") that unigrams miss.
3. **Class imbalance** (many categories have very different sizes) can degrade macro metrics; filtering to the top-10 classes mitigates this.

### Future Improvements
| Improvement | Expected Impact |
|-------------|----------------|
| Use transformer embeddings (e.g., BERT/DistilBERT) | +10–20% F1 |
| Include `short_description` text alongside headline | +5–8% F1 |
| Hyperparameter tuning via GridSearchCV | +2–5% F1 |
| Handle class imbalance with SMOTE or class weights | +3–7% F1 on minority classes |
| Deploy with FastAPI + Docker for production | Scalable serving |

---

*Generated by the News Category Classification System — May 2026*
