# 🎬 Movie Recommendation System

## 📌 Overview
This project implements a **content-based movie recommendation system** using the TMDB 5000 Movies dataset.  
The system recommends similar movies based on metadata such as genres, keywords, cast, director, and overview using **Bag-of-Words vectorization** and **Cosine Similarity**.

The objective is to demonstrate text feature engineering, similarity-based modeling, and recommendation logic implementation.

---

## 📂 Dataset

Dataset: **TMDB 5000 Movie Metadata**

Download from Kaggle:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Place the following files inside the project directory before running:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Note: The dataset is not included in this repository.

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- CountVectorizer
- Cosine Similarity

---

## 🔍 Methodology

1. Data cleaning and merging movie and credits datasets  
2. Feature extraction from:
   - Genres
   - Keywords
   - Cast (Top 3 actors)
   - Director
   - Overview text  
3. Text preprocessing and stemming  
4. Bag-of-Words vectorization (max 5000 features)  
5. Cosine similarity matrix computation  
6. Top-5 recommendation generation based on similarity scores  

---

## 🚀 Example Output

```text
Top 5 recommendations for 'Avatar':

Guardians of the Galaxy
Aliens
Star Trek
The Avengers
John Carter
```

---

## ▶️ How to Run

1. Download the dataset (see Dataset section above)
2. Place the CSV files in the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the script:

```bash
python movie_recommender.py
```

---

## 📁 Project Structure

```text
movie-recommendation-system/
│
├── movie_recommender.py
├── requirements.txt
└── README.md
```

---

## 🎯 Key Concepts Demonstrated

- Content-Based Filtering
- Natural Language Processing (Basic)
- Feature Engineering
- Text Vectorization
- Similarity-Based Recommendation
- Data Preprocessing Pipeline

---

## 📌 Future Improvements

- Deploy as a web application
- Use TF-IDF instead of Bag-of-Words
- Add collaborative filtering
- Optimize performance for large-scale data
