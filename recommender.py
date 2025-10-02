# recommender.py
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------------
# Data load + preprocessing
# -------------------------
def load_data(movies_csv='data/movies.csv', ratings_csv='data/ratings.csv'):
    movies = pd.read_csv(movies_csv)
    ratings = pd.read_csv(ratings_csv)
    agg = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    agg.columns = ['movieId','mean_rating','rating_count']
    movies = movies.merge(agg, on='movieId', how='left')
    movies['mean_rating'] = movies['mean_rating'].fillna(0)
    movies['rating_count'] = movies['rating_count'].fillna(0).astype(int)
    return movies

def clean_text(s):
    if pd.isna(s):
        return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocess(movies):
    movies = movies.copy()
    movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')
    movies['title'] = movies['title'].fillna('')
    movies['metadata'] = (movies['title'] + ' ' + movies['genres']).apply(clean_text)
    return movies

# -------------------------
# Model building
# -------------------------
def build_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['metadata'])
    return tfidf, tfidf_matrix

def build_similarity_matrix(tfidf_matrix):
    sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return sim

def create_title_index(movies):
    return pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()

# -------------------------
# Recommendation function
# -------------------------
def recommend(title, movies, similarity_matrix, title_index, top_n=10):
    title_key = title.lower()
    if title_key not in title_index.index:
        raise ValueError(f"Title '{title}' not found. Try another title or slightly different spelling.")
    idx = int(title_index[title_key])
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # exclude the item itself
    indices = [i[0] for i in sim_scores]
    results = movies.iloc[indices][['movieId','title','mean_rating','rating_count']].copy()
    results['score'] = [i[1] for i in sim_scores]
    return results.reset_index(drop=True)

# -------------------------
# Save/load helpers
# -------------------------
def save_artifacts(tfidf, similarity_matrix, title_index, movies,
                   tfidf_path='models/tfidf.pkl',
                   sim_path='models/similarity.npy',
                   idx_path='models/title_index.pkl',
                   movies_path='models/movies.pkl'):
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf, tfidf_path)
    np.save(sim_path, similarity_matrix)
    joblib.dump(title_index, idx_path)
    joblib.dump(movies, movies_path)

def load_artifacts(tfidf_path='models/tfidf.pkl',
                   sim_path='models/similarity.npy',
                   idx_path='models/title_index.pkl',
                   movies_path='models/movies.pkl'):
    tfidf = joblib.load(tfidf_path)
    sim = np.load(sim_path, allow_pickle=True)
    title_index = joblib.load(idx_path)
    movies = joblib.load(movies_path)
    return tfidf, sim, title_index, movies

# -------------------------
# CLI builder script
# -------------------------
if __name__ == '__main__':
    print("Loading data...")
    movies = load_data()
    print("Preprocessing...")
    movies = preprocess(movies)
    print("Building TF-IDF matrix...")
    tfidf, tfidf_matrix = build_tfidf_matrix(movies)
    print("Building similarity matrix (this may take a moment)...")
    similarity = build_similarity_matrix(tfidf_matrix)
    print("Creating title index...")
    title_index = create_title_index(movies)
    print("Saving artifacts to models/ ...")
    save_artifacts(tfidf, similarity, title_index, movies)
    print("Done. Files saved in ./models/")
