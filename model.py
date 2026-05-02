"""
model.py
========
This module is responsible for:
    - Loading and preparing training/test data
    - TF-IDF vectorisation (with a clear explanation)
    - Training Multinomial Naive Bayes and Logistic Regression models
    - Evaluating and comparing both models
    - Saving trained models to disk for reuse

What is TF-IDF?
---------------
TF-IDF stands for Term Frequency – Inverse Document Frequency.

  • Term Frequency (TF):
        How often a word appears in a single document.
        TF(word, doc) = count(word in doc) / total words in doc

  • Inverse Document Frequency (IDF):
        How rare a word is across ALL documents.
        IDF(word) = log( total docs / docs containing word )

  • TF-IDF score = TF × IDF

  Why does this matter?
    A word like "the" appears in every article (high TF) but carries
    no useful category signal (low IDF → low TF-IDF score).
    A word like "quarterback" appears rarely across all articles (high IDF)
    but frequently in sports articles (high TF for those docs) → high score.

  Result: TF-IDF gives high scores to words that are important for a
  specific document but rare elsewhere — exactly what we need for classification.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
NB_MODEL_PATH   = os.path.join(MODEL_DIR, 'naive_bayes.pkl')
LR_MODEL_PATH   = os.path.join(MODEL_DIR, 'logistic_regression.pkl')
LABELS_PATH     = os.path.join(MODEL_DIR, 'label_classes.pkl')


# ─── Vectorisation ────────────────────────────────────────────────────────────
def build_tfidf_vectorizer(max_features: int = 50_000,
                            ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Create a TF-IDF vectoriser.

    Parameters
    ----------
    max_features : int   – Keep only the top-N most frequent n-grams.
                           Larger = more vocabulary, slower training.
    ngram_range  : tuple – (1,1) = unigrams only;
                           (1,2) = unigrams + bigrams ('sports', 'world cup').

    Returns
    -------
    sklearn TfidfVectorizer (unfitted)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,      # apply log(1+TF) to dampen very high frequencies
        min_df=2,               # ignore terms appearing in fewer than 2 documents
    )


# ─── Data Split ───────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame,
               text_col: str = 'clean_text',
               label_col: str = 'category',
               test_size: float = 0.20,
               random_state: int = 42):
    """
    Stratified train/test split to keep class proportions balanced.

    Returns
    -------
    X_train, X_test, y_train, y_test (all as numpy arrays / pandas Series)
    """
    X = df[text_col].astype(str).to_numpy()
    y = df[label_col].astype(str).to_numpy()

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,             # ensures class balance in both splits
    )


# ─── Model Training ───────────────────────────────────────────────────────────
def train_models(X_train_raw, y_train):
    """
    Fit TF-IDF vectoriser and both classifiers on training data.

    Returns
    -------
    vectorizer, nb_model, lr_model
    """
    print("\n[Model] Fitting TF-IDF vectoriser …")
    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)   # learn vocabulary + transform

    # ── Multinomial Naive Bayes ────────────────────────────────────────────────
    # Why Naive Bayes?
    #   • Very fast to train on large text datasets
    #   • Works well with TF-IDF counts/frequencies
    #   • Strong baseline for multi-class text classification
    print("[Model] Training Multinomial Naive Bayes …")
    nb_model = MultinomialNB(alpha=0.1)   # alpha = Laplace smoothing parameter
    nb_model.fit(X_train, y_train)

    # ── Logistic Regression ───────────────────────────────────────────────────
    # Why Logistic Regression?
    #   • Learns feature weights (positive/negative) for each class
    #   • Generally more accurate than Naive Bayes on large datasets
    #   • Provides well-calibrated probability estimates
    print("[Model] Training Logistic Regression …")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=5.0,                  # inverse regularisation strength
        solver='saga',          # efficient solver for multi-class + large data
        n_jobs=-1,              # use all CPU cores
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    return vectorizer, nb_model, lr_model


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, vectorizer, X_test_raw, y_test, model_name: str) -> dict:
    """
    Evaluate a trained classifier and print a detailed report.

    Returns
    -------
    dict with accuracy, macro precision/recall/F1
    """
    X_test = vectorizer.transform(X_test_raw)
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {report['macro avg']['precision']:.4f}")
    print(f"  Recall    : {report['macro avg']['recall']:.4f}")
    print(f"  F1-Score  : {report['macro avg']['f1-score']:.4f}")
    print(f"\n  Full Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'model_name': model_name,
        'accuracy':   acc,
        'precision':  report['macro avg']['precision'],
        'recall':     report['macro avg']['recall'],
        'f1':         report['macro avg']['f1-score'],
        'y_pred':     y_pred,
    }


# ─── Confusion Matrix Plot ────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, class_names,
                           model_name: str, save_path: str = None):
    """
    Plot a colour-coded confusion matrix using seaborn.

    Parameters
    ----------
    y_test      : true labels
    y_pred      : predicted labels
    class_names : list of category strings
    model_name  : string used in the plot title
    save_path   : if provided, save the figure to this path
    """
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Normalise rows to show recall rates (0–1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f'Confusion Matrix — {model_name}\n(row-normalised recall)', fontsize=14)
    ax.set_xlabel('Predicted Category', fontsize=11)
    ax.set_ylabel('True Category',      fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Confusion matrix saved -> {save_path}")
    plt.close()


# ─── Model Comparison Bar Chart ───────────────────────────────────────────────
def plot_comparison(results: list, save_path: str = None):
    """
    Create a grouped bar chart comparing NB vs LR across metrics.

    Parameters
    ----------
    results   : list of dicts returned by evaluate_model()
    save_path : optional file path to save the figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels  = [r['model_name'] for r in results]
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#4C72B0', '#DD8452']

    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=res['model_name'], color=colors[i], alpha=0.85)
        # Add value labels on top of each bar
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Naive Bayes vs Logistic Regression')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Comparison chart saved -> {save_path}")
    plt.close()


# ─── Persist / Load Models ────────────────────────────────────────────────────
def save_models(vectorizer, nb_model, lr_model, label_classes):
    """Pickle all artefacts for reuse in the Streamlit app."""
    with open(VECTORIZER_PATH, 'wb') as f: pickle.dump(vectorizer,    f)
    with open(NB_MODEL_PATH,   'wb') as f: pickle.dump(nb_model,      f)
    with open(LR_MODEL_PATH,   'wb') as f: pickle.dump(lr_model,      f)
    with open(LABELS_PATH,     'wb') as f: pickle.dump(label_classes,  f)
    print(f"\n[Model] All artefacts saved to: {MODEL_DIR}/")


def load_models():
    """
    Load persisted models from disk.

    Returns
    -------
    vectorizer, nb_model, lr_model, label_classes
    Raises FileNotFoundError if models have not been trained yet.
    """
    for path in [VECTORIZER_PATH, NB_MODEL_PATH, LR_MODEL_PATH, LABELS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Please run `python main.py` first to train the models."
            )
    with open(VECTORIZER_PATH, 'rb') as f: vectorizer    = pickle.load(f)
    with open(NB_MODEL_PATH,   'rb') as f: nb_model      = pickle.load(f)
    with open(LR_MODEL_PATH,   'rb') as f: lr_model      = pickle.load(f)
    with open(LABELS_PATH,     'rb') as f: label_classes = pickle.load(f)
    return vectorizer, nb_model, lr_model, label_classes


# ─── Prediction Helper ────────────────────────────────────────────────────────
def predict_category(headline: str,
                      vectorizer,
                      model,
                      clean_fn) -> tuple[str, dict]:
    """
    Predict the category of a single news headline.

    Parameters
    ----------
    headline    : str – raw input from user
    vectorizer  : fitted TfidfVectorizer
    model       : trained classifier (NB or LR)
    clean_fn    : preprocessing function (clean_text from preprocessing.py)

    Returns
    -------
    (predicted_category, probability_dict)
    """
    cleaned  = clean_fn(headline)
    vec      = vectorizer.transform([cleaned])
    category = model.predict(vec)[0]

    # Probability scores per class (if the model supports it)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vec)[0]
        prob_dict = dict(zip(model.classes_, probs))
    else:
        prob_dict = {category: 1.0}

    return category, prob_dict
