# 📰 News Category Classification System

A complete Machine Learning project that classifies news headlines into categories using NLP and classical ML algorithms.

---

## 📁 Project Structure

```
news/
├── main.py              # ← Run this first! ML pipeline entry point
├── preprocessing.py     # Text cleaning & preprocessing functions
├── model.py             # TF-IDF, model training, evaluation, saving
├── app.py               # Streamlit web UI
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── REPORT.md            # Full project report
│
├── data/                # (auto-created) Dataset storage
│   └── news_data.json
│
├── models/              # (auto-created) Saved model artefacts
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   └── label_classes.pkl
│
└── outputs/             # (auto-created) Charts and visualisations
    ├── cm_naive_bayes.png
    ├── cm_logistic_regression.png
    └── model_comparison.png
```

---

## ⚙️ Installation

### Step 1 — Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3 — (Optional) Configure Kaggle API for the real HuffPost dataset

1. Sign up at [kaggle.com](https://www.kaggle.com/) and go to **Account → API → Create New Token**.
2. Place the downloaded `kaggle.json` in `C:\Users\<YourName>\.kaggle\`.
3. Install the Kaggle library: `pip install kaggle`

> **Without Kaggle credentials:** The project automatically generates a high-quality synthetic dataset covering 10 news categories — no action needed.

---

## 🚀 How to Run

### Step 1 — Train the models

```powershell
python main.py
```

This will:
- Load (or generate) the dataset
- Preprocess all headlines
- Train Multinomial Naive Bayes + Logistic Regression
- Print evaluation metrics and comparison table
- Save confusion matrix plots to `outputs/`
- Save trained models to `models/`

### Step 2 — Launch the Streamlit UI

```powershell
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 What Each File Does

| File | Purpose |
|------|---------|
| `preprocessing.py` | Lowercasing, punctuation removal, stopword removal, lemmatization |
| `model.py` | TF-IDF vectorisation, model training, evaluation, plotting, save/load |
| `main.py` | Orchestrates the full ML pipeline from data → trained models |
| `app.py` | Streamlit web app for interactive headline classification |

---

## 🧠 Machine Learning Concepts Used

| Concept | Description |
|---------|-------------|
| **TF-IDF** | Converts text to weighted numerical vectors |
| **Multinomial Naive Bayes** | Probabilistic classifier, fast and effective for text |
| **Logistic Regression** | Linear classifier with learned feature weights |
| **Train/Test Split** | 80/20 stratified split for unbiased evaluation |
| **Macro F1-Score** | Balanced metric for multi-class classification |
| **Confusion Matrix** | Visual breakdown of per-class accuracy |

---

## 🔍 Quick Test (no UI)

```python
from preprocessing import clean_text
from model import load_models, predict_category

vectorizer, nb_model, lr_model, label_classes = load_models()
headline = "Scientists discover a new planet that could support life"
category, probs = predict_category(headline, vectorizer, lr_model, clean_text)
print(f"Category: {category}")
```
