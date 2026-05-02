"""
main.py
=======
Entry point for the News Category Classification System.

What this script does (in order):
    1. Downloads the HuffPost News dataset from Kaggle / a fallback mirror
    2. Preprocesses the raw text
    3. Trains Multinomial Naive Bayes + Logistic Regression on TF-IDF features
    4. Evaluates both models and prints a comparison
    5. Saves confusion matrix and comparison charts to /outputs/
    6. Saves trained models to /models/ (used later by app.py)

Run:
    python main.py
"""

import os
import sys
import json
import requests
import zipfile
import io
import numpy as np
import pandas as pd

# ─── Local modules ────────────────────────────────────────────────────────────
from preprocessing import preprocess_dataframe, clean_text
from model import (
    split_data,
    train_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_comparison,
    save_models,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, 'news_data.json')

# ─── Dataset Sources ─────────────────────────────────────────────────────────
# Primary  : HuffPost News dataset on Kaggle
# Fallback : GitHub-hosted subset (no login required) for CI / offline usage
KAGGLE_DATASET = 'rmisra/news-category-dataset'
FALLBACK_URL   = (
    'https://raw.githubusercontent.com/explosion/projects/master/'
    'tutorials/textcat_goemotions/data/train.jsonl'   # placeholder — replaced below
)

# ─── Step 1: Data Acquisition ─────────────────────────────────────────────────
def download_kaggle_dataset() -> bool:
    """
    Try to download the dataset using the Kaggle CLI.
    Requires ~/.kaggle/kaggle.json with your API credentials.

    Returns True on success, False if Kaggle is unavailable.
    """
    try:
        import kaggle  # noqa: F401 – just check importability
        print("[Data] Downloading HuffPost dataset from Kaggle …")
        import subprocess
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET,
             '-p', DATA_DIR, '--unzip'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # Rename the downloaded JSON to our standard name
            for fname in os.listdir(DATA_DIR):
                if fname.endswith('.json') and 'news' in fname.lower():
                    os.rename(
                        os.path.join(DATA_DIR, fname),
                        RAW_DATA_PATH
                    )
                    break
            if os.path.exists(RAW_DATA_PATH):
                print("[Data] Kaggle download successful.")
                return True
        print(f"[Data] Kaggle CLI error: {result.stderr.strip()}")
    except Exception as e:
        print(f"[Data] Kaggle unavailable ({e}).")
    return False


def generate_synthetic_dataset(n_per_class: int = 600) -> pd.DataFrame:
    """
    Generate a realistic synthetic news dataset when the real dataset
    is not available (e.g. no Kaggle credentials).

    The dataset covers 10 common news categories with domain-specific
    headlines generated from templates + varied vocabulary.

    Parameters
    ----------
    n_per_class : int – number of samples per category (default 600)

    Returns
    -------
    pd.DataFrame with columns ['headline', 'category']
    """
    import random
    random.seed(42)
    np.random.seed(42)

    categories = {
        'POLITICS': [
            "{party} wins {region} election by wide margin",
            "President signs new {topic} bill into law",
            "Senate debates controversial {policy} reform",
            "Governor announces major infrastructure investment",
            "{candidate} leads polls ahead of midterm elections",
            "Congress passes bipartisan {topic} legislation",
            "White House unveils new foreign policy strategy",
            "Supreme Court rules on landmark {topic} case",
            "Mayor proposes record city budget amid housing crisis",
            "Parliamentary session opens with heated {topic} debate",
        ],
        'SPORTS': [
            "{team} defeats {rival} in thrilling championship final",
            "Star {sport} player signs record-breaking contract",
            "{athlete} breaks world record at international games",
            "Coach fired after team's worst {sport} season in decades",
            "{team} advances to playoffs with stunning comeback victory",
            "Olympics committee announces new host city for {year} games",
            "Injury scare for {athlete} ahead of major tournament",
            "{sport} league introduces new rules to speed up play",
            "Transfer window: {team} signs top international striker",
            "{team} clinches title on final day of the season",
        ],
        'TECHNOLOGY': [
            "{company} launches revolutionary new AI {product}",
            "Tech giant accused of {issue} by regulators",
            "New {device} promises to change the way we work",
            "Cybersecurity breach exposes millions of {company} users",
            "Silicon Valley startup raises $1 billion in Series C funding",
            "{company} unveils next-generation quantum computing chip",
            "Scientists develop {material} that could replace silicon",
            "Social media platform faces backlash over {issue} policy",
            "Electric vehicle maker announces breakthrough battery technology",
            "Cloud computing spending hits new record as businesses digitalise",
        ],
        'ENTERTAINMENT': [
            "{celebrity} wins Oscar for best {role} in {film}",
            "Box office smash: {film} earns $1 billion in opening weekend",
            "{show} renewed for fourth season after record viewership",
            "{singer} announces world tour with 50 stadium shows",
            "Hollywood strike: writers demand fairer streaming deal",
            "{celebrity} and {partner} announce surprise engagement",
            "Streaming wars: {platform} surpasses 200 million subscribers",
            "Director's cut of classic {film} released after 20 years",
            "{band} reunites for one-night-only anniversary concert",
            "Award show controversy: snubbed artists speak out",
        ],
        'HEALTH': [
            "New study links {food} to reduced risk of {disease}",
            "FDA approves breakthrough drug for treating {condition}",
            "Doctors warn of rising {illness} cases this winter",
            "Mental health crisis: one in five adults reports {symptom}",
            "Exercise found to significantly lower risk of {disease}",
            "Vaccine for {disease} shows 95% efficacy in Phase 3 trial",
            "Sugar consumption linked to increased {condition} risk",
            "Sleep deprivation affects {percentage}% of working adults",
            "New cancer screening method improves early detection rates",
            "WHO declares {disease} a global health emergency",
        ],
        'BUSINESS': [
            "{company} stock surges {pct}% after earnings beat expectations",
            "Merger talks between {company} and {rival} confirmed",
            "Central bank raises interest rates to curb inflation",
            "Retail giant announces closure of {n} stores nationwide",
            "Startup disrupts {industry} with AI-powered platform",
            "{commodity} prices hit record high amid supply shortage",
            "Trade deal between {country1} and {country2} signed",
            "Unemployment falls to lowest level in {n} years",
            "CEO of {company} steps down amid accounting scandal",
            "IPO frenzy: {company} valued at $50 billion on first day",
        ],
        'SCIENCE': [
            "Astronomers discover Earth-like planet in nearby star system",
            "New research reveals secrets of the {phenomenon}",
            "Scientists clone endangered {animal} for the first time",
            "Breakthrough in {field} could transform medicine",
            "Space agency confirms {mission} will launch in {year}",
            "Deep-sea expedition uncovers unknown species of {creature}",
            "Climate study finds Arctic ice melting faster than predicted",
            "Physicists detect gravitational waves from {event}",
            "Gene editing technique corrects hereditary {condition}",
            "Ancient fossil rewrites history of early {species}",
        ],
        'TRAVEL': [
            "Best budget destinations for {year}: top 10 picks",
            "{country} opens borders to tourists after two-year closure",
            "Airline announces new direct route from {city1} to {city2}",
            "Travel hack: how to find cheap flights every time",
            "{destination} named world's top travel destination of the year",
            "Overtourism forces {landmark} to limit daily visitors",
            "Luxury hotel opens on remote {destination} island",
            "Road trip guide: exploring {region} by car",
            "Solo travel: tips for women travelling {destination} safely",
            "Adventure tourism booms as travellers seek extreme experiences",
        ],
        'FOOD': [
            "Chef reveals secret recipe behind award-winning {dish}",
            "The best {cuisine} restaurants in {city} right now",
            "Food trend: why everyone is eating {ingredient} in {year}",
            "Street food festival attracts thousands to {city} this weekend",
            "How to make the perfect {dish} at home in 30 minutes",
            "Michelin-starred restaurant opens second location in {city}",
            "Plant-based {dish} gains mainstream appeal among meat lovers",
            "Food scientists develop lab-grown {protein} with real texture",
            "Why {ingredient} is the new superfood you need to try",
            "Local bakery's {item} goes viral with 10 million social views",
        ],
        'ENVIRONMENT': [
            "Carbon emissions reach record high despite climate pledges",
            "{country} commits to net-zero by {year} in landmark deal",
            "Wildfires devastate {region}, destroying thousands of acres",
            "Ocean plastic pollution threatens {species} with extinction",
            "Renewable energy surpasses coal for first time in history",
            "Flooding forces evacuation of {n} thousand residents in {region}",
            "Government bans single-use plastics starting next year",
            "Deforestation in {region} accelerates to alarming new rate",
            "Solar panel efficiency breaks new record in lab test",
            "Marine biologists warn of coral reef collapse within {n} years",
        ],
    }

    # Slot-filling vocabulary
    slots = {
        '{party}':       ['Democrats', 'Republicans', 'Labour', 'Conservatives', 'Green Party'],
        '{region}':      ['California', 'Texas', 'New York', 'London', 'Paris', 'Berlin'],
        '{topic}':       ['healthcare', 'education', 'immigration', 'tax', 'climate'],
        '{policy}':      ['gun control', 'immigration', 'healthcare', 'tax', 'education'],
        '{candidate}':   ['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'],
        '{team}':        ['Lakers', 'Yankees', 'Chelsea', 'Real Madrid', 'Patriots'],
        '{rival}':       ['Celtics', 'Red Sox', 'Arsenal', 'Barcelona', 'Chiefs'],
        '{sport}':       ['basketball', 'football', 'tennis', 'cricket', 'soccer'],
        '{athlete}':     ['Williams', 'Hamilton', 'Federer', 'Mbappe', 'Bolt'],
        '{year}':        ['2025', '2026', '2027', '2028', '2030'],
        '{company}':     ['Apple', 'Google', 'Microsoft', 'Meta', 'Amazon', 'Tesla'],
        '{product}':     ['assistant', 'smartphone', 'laptop', 'chip', 'platform'],
        '{issue}':       ['data privacy', 'antitrust', 'misinformation', 'tax avoidance'],
        '{device}':      ['smartwatch', 'AR headset', 'foldable phone', 'AI tablet'],
        '{material}':    ['graphene', 'perovskite', 'carbon nanotube', 'aerogel'],
        '{celebrity}':   ['Pitt', 'Jolie', 'DiCaprio', 'Streep', 'Hanks'],
        '{role}':        ['actor', 'actress', 'director', 'producer'],
        '{film}':        ['Horizon', 'Eclipse', 'The Last Stand', 'Infinity', 'Genesis'],
        '{show}':        ['The Crown', 'Stranger Things', 'Succession', 'Euphoria'],
        '{singer}':      ['Taylor Swift', 'Beyoncé', 'Ed Sheeran', 'Adele', 'Drake'],
        '{partner}':     ['long-term partner', 'co-star', 'childhood sweetheart'],
        '{platform}':    ['Netflix', 'Disney+', 'HBO Max', 'Apple TV+', 'Prime Video'],
        '{band}':        ['The Beatles', 'Rolling Stones', 'ABBA', 'Fleetwood Mac'],
        '{food}':        ['blueberries', 'olive oil', 'green tea', 'turmeric', 'walnuts'],
        '{disease}':     ['cancer', 'diabetes', 'Alzheimer\'s', 'heart disease', 'flu'],
        '{condition}':   ['depression', 'obesity', 'anxiety', 'hypertension', 'arthritis'],
        '{illness}':     ['flu', 'RSV', 'norovirus', 'measles', 'whooping cough'],
        '{symptom}':     ['burnout', 'chronic stress', 'insomnia', 'anxiety'],
        '{percentage}':  ['47', '55', '62', '38', '71'],
        '{pct}':         ['12', '18', '25', '8', '35'],
        '{n}':           ['200', '500', '50', '100', '1,000'],
        '{industry}':    ['finance', 'logistics', 'retail', 'healthcare', 'education'],
        '{commodity}':   ['oil', 'wheat', 'copper', 'lithium', 'coffee'],
        '{country1}':    ['USA', 'UK', 'Germany', 'Japan', 'India'],
        '{country2}':    ['China', 'France', 'Canada', 'Brazil', 'Australia'],
        '{country}':     ['Japan', 'Thailand', 'Italy', 'Greece', 'New Zealand'],
        '{phenomenon}':  ['dark matter', 'black holes', 'quantum entanglement', 'dark energy'],
        '{animal}':      ['Siberian tiger', 'white rhino', 'giant panda', 'snow leopard'],
        '{field}':       ['genomics', 'neuroscience', 'materials science', 'immunology'],
        '{mission}':     ['Mars mission', 'lunar gateway', 'asteroid mission', 'space telescope'],
        '{creature}':    ['fish', 'octopus', 'shrimp', 'jellyfish', 'crab'],
        '{event}':       ['neutron star merger', 'black hole collision', 'supernova'],
        '{species}':     ['human', 'dinosaur', 'mammal', 'primate', 'bird'],
        '{city1}':       ['New York', 'London', 'Dubai', 'Singapore', 'Sydney'],
        '{city2}':       ['Tokyo', 'Paris', 'Mumbai', 'Toronto', 'Cape Town'],
        '{destination}': ['Maldives', 'Iceland', 'Patagonia', 'Bhutan', 'Faroe Islands'],
        '{landmark}':    ['Machu Picchu', 'Santorini', 'Venice', 'Angkor Wat', 'Colosseum'],
        '{city}':        ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
        '{dish}':        ['pasta', 'ramen', 'biryani', 'tacos', 'croissant', 'sushi'],
        '{cuisine}':     ['Italian', 'Japanese', 'Mexican', 'Indian', 'French'],
        '{ingredient}':  ['moringa', 'ashwagandha', 'miso', 'sumac', 'tahini'],
        '{protein}':     ['chicken', 'beef', 'salmon', 'tuna', 'shrimp'],
        '{item}':        ['sourdough', 'croissant', 'cinnamon roll', 'pretzel', 'brioche'],
    }

    def fill_template(template: str) -> str:
        """Replace all {slot} placeholders with random values."""
        for slot, options in slots.items():
            while slot in template:
                template = template.replace(slot, random.choice(options), 1)
        return template

    records = []
    for category, templates in categories.items():
        for _ in range(n_per_class):
            template = random.choice(templates)
            headline = fill_template(template)
            records.append({'headline': headline, 'category': category})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[Data] Generated synthetic dataset: {len(df):,} samples across {len(categories)} categories.")
    return df


def load_dataset() -> pd.DataFrame:
    """
    Load the news dataset.

    Priority:
        1. Already-downloaded local JSON (skip re-download)
        2. Kaggle CLI download
        3. Synthetic dataset (fallback, no credentials needed)
    """
    # ── Already cached? ──────────────────────────────────────────────────────
    if os.path.exists(RAW_DATA_PATH):
        print(f"[Data] Loading cached dataset from {RAW_DATA_PATH} …")
        try:
            # HuffPost dataset is stored as newline-delimited JSON
            df = pd.read_json(RAW_DATA_PATH, lines=True)
            if 'headline' in df.columns and 'category' in df.columns:
                print(f"[Data] Loaded {len(df):,} records.")
                return df
        except Exception:
            pass

    # ── Try Kaggle ─────────────────────────────────────────────────────────
    if download_kaggle_dataset():
        df = pd.read_json(RAW_DATA_PATH, lines=True)
        print(f"[Data] Loaded {len(df):,} records.")
        return df

    # ── Fallback: synthetic ───────────────────────────────────────────────
    print("[Data] Using synthetic dataset (Kaggle credentials not found).")
    df = generate_synthetic_dataset(n_per_class=600)

    # Persist to disk for caching
    df.to_json(RAW_DATA_PATH, orient='records', lines=True)
    return df


# ─── Step 2-5: Main Orchestration ─────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("   NEWS CATEGORY CLASSIFICATION SYSTEM")
    print("   Machine Learning Pipeline")
    print("="*65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    raw_df = load_dataset()

    # Show class distribution
    print("\n[Data] Category distribution (top 10):")
    print(raw_df['category'].value_counts().head(10).to_string())

    # Optional: keep only top-N most frequent categories (helps with rare classes)
    TOP_N = 10
    top_cats = raw_df['category'].value_counts().head(TOP_N).index.tolist()
    df_filtered = raw_df[raw_df['category'].isin(top_cats)].copy()
    print(f"\n[Data] Keeping top {TOP_N} categories -> {len(df_filtered):,} samples.")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    clean_df = preprocess_dataframe(df_filtered, text_col='headline', label_col='category')

    # ── 3. Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(clean_df)
    print(f"\n[Split] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 4. Train models ───────────────────────────────────────────────────────
    vectorizer, nb_model, lr_model = train_models(X_train, y_train)

    # ── 5. Evaluate models ────────────────────────────────────────────────────
    class_names = sorted(nb_model.classes_)

    nb_results = evaluate_model(nb_model, vectorizer, X_test, y_test,
                                 'Multinomial Naive Bayes')
    lr_results = evaluate_model(lr_model, vectorizer, X_test, y_test,
                                 'Logistic Regression')

    # ── 6. Confusion matrix plots ─────────────────────────────────────────────
    plot_confusion_matrix(
        y_test, nb_results['y_pred'], class_names,
        model_name='Multinomial Naive Bayes',
        save_path=os.path.join(OUTPUT_DIR, 'cm_naive_bayes.png')
    )
    plot_confusion_matrix(
        y_test, lr_results['y_pred'], class_names,
        model_name='Logistic Regression',
        save_path=os.path.join(OUTPUT_DIR, 'cm_logistic_regression.png')
    )

    # ── 7. Comparison chart ───────────────────────────────────────────────────
    plot_comparison(
        [nb_results, lr_results],
        save_path=os.path.join(OUTPUT_DIR, 'model_comparison.png')
    )

    # ── 8. Winner announcement ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  MODEL COMPARISON SUMMARY")
    print("="*65)
    print(f"  {'Metric':<20} {'Naive Bayes':>15} {'Logistic Reg.':>15}")
    print(f"  {'-'*50}")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        nb_val = nb_results[metric]
        lr_val = lr_results[metric]
        winner = '<-- WIN' if lr_val >= nb_val else ''
        print(f"  {metric.capitalize():<20} {nb_val:>15.4f} {lr_val:>15.4f}  {winner}")

    best = 'Logistic Regression' if lr_results['f1'] >= nb_results['f1'] else 'Multinomial Naive Bayes'
    print(f"\n  [WINNER]  Best Model: {best}  (by macro F1-Score)")
    print("="*65)

    # ── 9. Save artefacts ─────────────────────────────────────────────────────
    save_models(vectorizer, nb_model, lr_model, class_names)

    # ── 10. Quick demo prediction ─────────────────────────────────────────────
    from model import predict_category

    demo_headlines = [
        "Scientists discover new exoplanet that could support life",
        "Champions League final: Real Madrid defeat Manchester City",
        "Fed raises interest rates to fight inflation",
        "New study links coffee consumption to lower risk of diabetes",
        "Apple unveils the most powerful chip ever built",
    ]
    print("\n[Demo] Sample Predictions:")
    print(f"  {'Headline':<55} {'NB':>15} {'LR':>15}")
    print(f"  {'-'*85}")
    for hl in demo_headlines:
        nb_cat, _ = predict_category(hl, vectorizer, nb_model, clean_text)
        lr_cat, _ = predict_category(hl, vectorizer, lr_model, clean_text)
        print(f"  {hl[:53]:<55} {nb_cat:>15} {lr_cat:>15}")

    print("\n[DONE] Training complete! Run `streamlit run app.py` to launch the UI.\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
