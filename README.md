# 🎬 Movie Recommendation System

## 📌 Overview
This project implements a **content-based movie recommendation system** using the TMDB 5000 dataset.  
It recommends similar movies based on metadata such as genres, keywords, cast, and director using similarity-based modeling.

---

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  

---

## 🔍 Methodology
1. Data cleaning and merging movie and credits datasets  
2. Feature engineering from genres, keywords, cast, and director  
3. Text preprocessing and stemming  
4. Bag-of-Words vectorization (5000 features)  
5. Cosine similarity matrix computation  
6. Top-5 recommendation generation  

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

1. Download the TMDB 5000 dataset  
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project directory  

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

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

