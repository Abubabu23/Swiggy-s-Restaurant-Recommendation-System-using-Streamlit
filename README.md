# Swiggy Restaurant Recommendation System using Cosine Similarity

A machine-learning powered restaurant recommendation system built using Python, Streamlit, One-Hot Encoding, and Cosine Similarity.

## 🚀 Project Overview
This project recommends restaurants based on:
- City
- Cuisine
- Cost
- Rating

## 📂 Dataset Description
Columns: id, name, city, rating, cost, cuisine, link, address, menu

## 🧹 Data Cleaning
- Removed duplicates
- Handled missing values
- Standardized text
- Saved cleaned_data.csv

## 🛠️ Preprocessing
- One-Hot Encoding for city & cuisine
- Saved encoder.pkl and encoded_data.csv

## 🤖 Recommendation Engine – Cosine Similarity
- Encodes user input
- Computes similarity against restaurant vectors
- Returns top-N recommended restaurants

## 🎨 Streamlit App
Interactive UI for:
- Selecting city, cuisine
- Filtering by cost and rating
- Displaying recommendations

## ▶️ Run the App
```
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## 📊 Results
- Accurate similarity-based recommendations
- Clean UI

## 📌 Future Work
- Add TF-IDF for menus
- Add embeddings
- Geo-distance filtering
