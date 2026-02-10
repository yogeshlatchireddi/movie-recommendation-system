# ==========================================
# Movie Recommendation System
# Content-Based Filtering using TMDB Dataset
# ==========================================

import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------
# Load Dataset
# ------------------------------------------

try:
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
except FileNotFoundError:
    print("Dataset files not found. Please download and place them in the project folder.")
    exit()

# ------------------------------------------
# Merge Datasets
# ------------------------------------------

movies = movies.merge(credits, on="title")

# ------------------------------------------
# Select Important Columns
# ------------------------------------------

movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)

# ------------------------------------------
# Helper Functions
# ------------------------------------------

def convert(text):
    """Extract names from list of dictionaries"""
    L = []
    for item in ast.literal_eval(text):
        L.append(item['name'])
    return L


def fetch_director(text):
    """Extract director name from crew"""
    L = []
    for item in ast.literal_eval(text):
        if item['job'] == 'Director':
            L.append(item['name'])
    return L


# Apply extraction
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(fetch_director)

# ------------------------------------------
# Text Preprocessing
# ------------------------------------------

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces inside words
for col in ['genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

# Combine features
movies['tags'] = movies['overview'] + movies['genres'] + \
                 movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']].copy()

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# ------------------------------------------
# Stemming
# ------------------------------------------

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

# ------------------------------------------
# Vectorization
# ------------------------------------------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# ------------------------------------------
# Similarity Matrix
# ------------------------------------------

similarity = cosine_similarity(vectors)

# ------------------------------------------
# Recommendation Function
# ------------------------------------------

def recommend(movie_name):
    movie_name = movie_name.lower()

    # Case-insensitive match
    matches = new_df[new_df['title'].str.lower() == movie_name]

    if matches.empty:
        print("Movie not found in database.")
        return

    movie_index = matches.index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nTop 5 recommendations for '{new_df.iloc[movie_index].title}':\n")

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# ------------------------------------------
# Main Execution
# ------------------------------------------

if __name__ == "__main__":
    movie = input("Enter a movie name: ")
    recommend(movie)

