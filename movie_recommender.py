"""
Movie Recommendation System
Content-Based Filtering using Bag of Words and Cosine Similarity
Author: Yogesh Latchireddi
"""

# Required: Download NLTK resources once before running
# import nltk
# nltk.download('punkt')

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_data():
    try:
        movies = pd.read_csv("tmdb_5000_movies.csv")
        credits = pd.read_csv("tmdb_5000_credits.csv")
    except FileNotFoundError:
        print("Dataset files not found. Please place TMDB CSV files in the project directory.")
        exit()

    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
    movies.dropna(inplace=True)
    return movies


# -----------------------------
# 2. Helper Functions
# -----------------------------
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]


def convert_cast(text):
    L = []
    for i in ast.literal_eval(text)[:3]:
        L.append(i['name'])
    return L


def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []


# -----------------------------
# 3. Feature Engineering
# -----------------------------
def preprocess(movies):
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Remove spaces from names
    for column in ['genres', 'keywords', 'cast', 'crew']:
        movies[column] = movies[column].apply(
            lambda x: [i.replace(" ", "") for i in x]
        )

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    return new_df


# -----------------------------
# 4. Stemming
# -----------------------------
def apply_stemming(new_df):
    ps = PorterStemmer()

    def stem(text):
        return " ".join(ps.stem(word) for word in text.split())

    new_df['tags'] = new_df['tags'].apply(stem)
    return new_df


# -----------------------------
# 5. Vectorization
# -----------------------------
def vectorize(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    return vectors


# -----------------------------
# 6. Similarity Matrix
# -----------------------------
def compute_similarity(vectors):
    return cosine_similarity(vectors)


# -----------------------------
# 7. Recommendation Function
# -----------------------------
def recommend(movie, new_df, similarity):
    if movie not in new_df['title'].values:
        print("Movie not found in dataset.")
        return

    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nTop 5 recommendations for '{movie}':\n")
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# -----------------------------
# 8. Main Execution
# -----------------------------
if __name__ == "__main__":
    movies = load_data()
    new_df = preprocess(movies)
    new_df = apply_stemming(new_df)
    vectors = vectorize(new_df)
    similarity = compute_similarity(vectors)

    # Example usage
    recommend("Avatar", new_df, similarity)
