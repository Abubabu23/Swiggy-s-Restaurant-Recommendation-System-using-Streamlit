import pandas as pd
import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


##Load Data
cleanned_data = pd.read_csv(r"C:\Users\Abuthahir\project\Swiggy Food Recommentation System\data\cleaned_data.csv")
encoder_data = pd.read_csv(r"C:\Users\Abuthahir\project\Swiggy Food Recommentation System\data\encoded_data.csv")
with open("data/encoder.pkl","rb")as f:
    encoder = pickle.load(f)

##Cleaning Encoded Data

encoder_data = encoder_data.replace('--', np.nan)
encoder_data = encoder_data.apply(pd.to_numeric, errors='coerce')
encoder_data = encoder_data.fillna(0)

##Streamlit UI

st.title ("Swiggy Restaurant Recommendations")
st.write("Select Your Preferences:")

##User input

city = st.selectbox('City',cleanned_data['city'].unique())
cuisine = st.selectbox('Cuisine',cleanned_data['cuisine'].unique())
min_rating = st.slider('Minimum Rating',0.0,5.0,3.5)
max_cost = st.number_input("Maximum cost",min_value=0,value=500)

if st.button("Show Recommendation"):
    user_input = pd.DataFrame({
        'city': [city],
        'cuisine': [cuisine],
        'rating': [min_rating],
        'cost': [max_cost]
    })

    # FIX 1: remove .toarray()
    user_encoded = encoder.transform(user_input[['city', 'cuisine']])

    user_vector = np.hstack((
        user_input[['rating', 'cost']].values,
        user_encoded
    ))

    similarities = cosine_similarity(encoder_data.values, user_vector)
    top_indices = similarities.flatten().argsort()[-10:][::-1]

    recommendations = cleanned_data.iloc[top_indices]

    st.subheader("Recommended Restaurants")
    for _, row in recommendations.iterrows():
        # FIX 2: quote issue
        st.write(f"**{row['name']}** | {row['cuisine']} | {row['city']}")
        st.write(f"Rating: {row['rating']} | Cost: {row['cost']}")
        st.write(f"Address: {row['address']}")
        st.write("---")
