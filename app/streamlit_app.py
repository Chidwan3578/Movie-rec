import streamlit as st
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recommender import load_artifacts, recommend, load_data, preprocess, create_title_index
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender (Content-based)")

tmdb_key = st.text_input("Optional: TMDB API Key (for posters)", type="password")
show_posters = st.checkbox("Show posters (requires TMDB key)", value=False)

@st.cache_data(show_spinner=False)
def load_models_cached():
    try:
        tfidf, sim, title_index, movies = load_artifacts()
        return tfidf, sim, title_index, movies
    except Exception:
        movies = load_data()
        movies = preprocess(movies)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['metadata'])
        sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        title_index = create_title_index(movies)
        return tfidf, sim, title_index, movies

tfidf, sim_matrix, title_index, movies = load_models_cached()

movie_choice = st.selectbox("Pick a movie", sorted(movies['title'].unique()))
top_n = st.slider("Number of recommendations", 3, 20, 6)

if st.button("Get Recommendations"):
    try:
        recs = recommend(movie_choice, movies, sim_matrix, title_index, top_n=top_n)
        for idx, row in recs.iterrows():
            st.markdown(f"**{row['title']}** — score: `{row['score']:.3f}`  — avg rating: `{row['mean_rating']:.2f}` ({int(row['rating_count'])} ratings)")
            if show_posters and tmdb_key:
                try:
                    params = {"api_key": tmdb_key, "query": row['title']}
                    r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
                    data = r.json()
                    if data.get('results'):
                        poster_path = data['results'][0].get('poster_path')
                        if poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w300{poster_path}"
                            st.image(poster_url, width=150)
                except Exception:
                    pass
    except Exception as e:
        st.error(str(e))
