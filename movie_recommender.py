# ==========================================
# Movie Recommendation System
# Content-Based Filtering using TMDB Dataset
# ==========================================

import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------
# Download NLTK resources (run once)
# ------------------------------------------
nltk.download('punkt')

# ------------------------------------------
# Load Dataset
# ------------------------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# ------------------------------------------
# Merge Datasets
# ------------------------------------------
movies = movies.merge(credits, on='title')

# ------------------------------------------
# Select Important Columns
# ------------------------------------------
movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]

# ------------------------------------------
# Handle Missing Values
# ------------------------------------------
movies.dropna(inplace=True)


# ------------------------------------------
# Helper Functions to Extract Data
# ------------------------------------------

def convert(text):
    """Extract names from list of dictionaries"""
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


def fetch_director(text):
    """Extract director name from crew"""
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# Apply extraction functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(fetch_director)

# ------------------------------------------
# Text Preprocessing
# ------------------------------------------

# Remove spaces between words (e.g., "Sam Worthington" -> "SamWorthington")
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create Tags Column (combine all features)
movies['tags'] = movies['overview'] + movies['genres'] + \
                 movies['keywords'] + movies['cast'] + movies['crew']

# Keep only required columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# ------------------------------------------
# Stemming
# ------------------------------------------

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


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

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    print(f"\nTop 5 recommendations for '{movie}':\n")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# ------------------------------------------
# Main Execution
# ------------------------------------------

if __name__ == "__main__":
    movie_name = input("Enter a movie name: ")
    recommend(movie_name)

