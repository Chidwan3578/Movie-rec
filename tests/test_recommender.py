# tests/test_recommender.py
import unittest
from recommender import load_data, preprocess, build_tfidf_matrix, build_similarity_matrix, create_title_index, recommend

class TestRec(unittest.TestCase):
    def test_recommend_basic(self):
        movies = load_data()
        movies = preprocess(movies)
        tfidf, tfidf_matrix = build_tfidf_matrix(movies)
        sim = build_similarity_matrix(tfidf_matrix)
        idx = create_title_index(movies)
        # use first movie title as input
        first_title = movies['title'].iloc[0]
        res = recommend(first_title, movies, sim, idx, top_n=5)
        self.assertTrue(len(res) > 0)

if __name__ == '__main__':
    unittest.main()
