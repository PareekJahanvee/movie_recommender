import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
df = None
feature_matrices = {
    "all": {"tfidf_matrix": None, "cosine_sim": None},
    "genre": {"tfidf_matrix": None, "cosine_sim": None},
    "director": {"tfidf_matrix": None, "cosine_sim": None},
    "star": {"tfidf_matrix": None, "cosine_sim": None}
}
title_to_idx = None

def clean_text(text):
    """Clean text by removing special characters and converting to lowercase."""
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower())

def load_data():
    """Load and preprocess movie data."""
    global df, feature_matrices, title_to_idx
    start_time = time.time()
    logger.info("⏳ Loading and vectorizing movie data...")
    
    # Load dataset
    df = pd.read_csv("detail_movies.csv").fillna("")
    
    # Clean text columns
    for col in ['genre', 'director', 'star', 'title']:
        df[col] = df[col].apply(clean_text)
    
    # Combine features for TF-IDF
    df['combined'] = df['genre'] + ' ' + df['director'] + ' ' + df['star']
    
    # Compute TF-IDF and cosine similarity for each filter
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    for feature in feature_matrices:
        if feature == "all":
            feature_matrices[feature]["tfidf_matrix"] = vectorizer.fit_transform(df['combined'])
        else:
            feature_matrices[feature]["tfidf_matrix"] = vectorizer.fit_transform(df[feature])
        feature_matrices[feature]["cosine_sim"] = cosine_similarity(feature_matrices[feature]["tfidf_matrix"])
    
    # Map titles to indices
    title_to_idx = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    logger.info(f"✅ Vectorization complete in {time.time() - start_time:.2f} seconds.")
    return df

def get_recommendations(title, filter="all", threshold=0.3):
    """Get movie recommendations based on cosine similarity."""
    start_time = time.time()
    
    # Find closest matching title
    best_match = max(
        [(t, fuzz.partial_ratio(title, t)) for t in title_to_idx.index],
        key=lambda x: x[1]
    )
    if best_match[1] < 50:
        return title, []
    title = best_match[0]
    
    if title not in title_to_idx:
        return title, []
    
    idx = title_to_idx[title]
    cosine_sim = feature_matrices[filter]["cosine_sim"]
    sim_scores = [(i, s) for i, s in enumerate(cosine_sim[idx]) if i != idx and s > threshold]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    recommendations = [
        {
            'title': df['title'].iloc[i].title(),
            'similarity': round(s * 100, 2),
            'year': int(df['release_year'].iloc[i]),
            'rating': float(df['imdb_rating'].iloc[i]),
            'genre': df['genre'].iloc[i].title()
        }
        for i, s in sim_scores
    ]
    
    logger.info(f"✅ Recommendations for '{title}' ({filter}) generated in {time.time() - start_time:.2f} seconds.")
    return title, recommendations