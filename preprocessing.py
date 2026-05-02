"""
preprocessing.py
================
This module handles all text preprocessing steps for the News Category Classification System.

Steps:
    1. Lowercasing
    2. Removing punctuation and special characters
    3. Removing stopwords (common words like 'the', 'is', etc.)
    4. Lemmatization (converting words to their base form: "running" -> "run")

Why preprocess?
    Raw text contains noise (punctuation, uppercase, filler words) that doesn't
    help the model learn meaningful patterns. Cleaning the text improves accuracy.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─── Download Required NLTK Resources ────────────────────────────────────────
def download_nltk_resources():
    """Download all required NLTK data packages if not already present."""
    import os
    
    # On Vercel (Serverless), we must use /tmp for downloads
    if os.environ.get('VERCEL'):
        nltk_path = '/tmp/nltk_data'
        if nltk_path not in nltk.data.path:
            nltk.data.path.append(nltk_path)
    else:
        nltk_path = None

    resources = [
        ('tokenizers/punkt',        'punkt'),
        ('tokenizers/punkt_tab',    'punkt_tab'),
        ('corpora/stopwords',       'stopwords'),
        ('corpora/wordnet',         'wordnet'),
        ('corpora/omw-1.4',         'omw-1.4'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True, download_dir=nltk_path)

download_nltk_resources()


# ─── Initialise Tools ─────────────────────────────────────────────────────────
STOP_WORDS   = set(stopwords.words('english'))
LEMMATIZER   = WordNetLemmatizer()


# ─── Core Cleaning Function ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Apply a full preprocessing pipeline to a single text string.

    Pipeline:
        1. Lowercase  – normalises case so 'News' == 'news'
        2. Remove URLs – strips http/www links
        3. Remove punctuation & digits – keeps only alphabetic characters
        4. Tokenise – splits string into individual words
        5. Remove stopwords – drops common English filler words
        6. Lemmatize – reduces words to dictionary base forms

    Parameters
    ----------
    text : str
        Raw news headline or body text.

    Returns
    -------
    str
        Cleaned, space-joined token string ready for vectorisation.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Step 1 – Lowercase
    text = text.lower()

    # Step 2 – Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3 – Remove punctuation, digits, and extra whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep only a-z and spaces
    text = re.sub(r'\s+', ' ', text).strip() # collapse multiple spaces

    # Step 4 – Tokenise
    tokens = word_tokenize(text)

    # Step 5 – Remove stopwords and very short tokens (length < 2)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # Step 6 – Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# ─── DataFrame-Level Preprocessing ───────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame,
                          text_col: str = 'headline',
                          label_col: str = 'category') -> pd.DataFrame:
    """
    Preprocess an entire DataFrame for model training.

    Parameters
    ----------
    df       : pd.DataFrame  – raw dataset
    text_col : str           – column containing news headlines/text
    label_col: str           – column containing category labels

    Returns
    -------
    pd.DataFrame with columns ['clean_text', 'category']
    """
    print("[Preprocessing] Cleaning text …")

    # Drop rows with missing values in key columns
    df = df[[text_col, label_col]].dropna().copy()

    # Apply text cleaning
    df['clean_text'] = df[text_col].apply(clean_text)

    # Drop rows where cleaning produced an empty string
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

    print(f"[Preprocessing] Done. {len(df):,} samples ready.")
    return df[['clean_text', 'category']]


# ─── Quick Demo ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    samples = [
        "Breaking News: Scientists Discover New Planet in the Milky Way!",
        "Top 10 Tips for Healthy Living in 2024 -- A Complete Guide",
        "U.S. Stock Markets Surge After Federal Reserve Policy Update",
    ]
    print("=== Preprocessing Demo ===\n")
    for s in samples:
        print(f"  Original : {s}")
        print(f"  Cleaned  : {clean_text(s)}\n")
