import streamlit as sl
import pandas as pd
import joblib as jb
from sklearn.metrics.pairwise import sigmoid_kernel

# Load precomputed TF-IDF vectorizer and matrix
tfidfvec = jb.load("tfidfvec.pkl")
tfidf_matrix = jb.load("tfidf_matrix.pkl")

# Load the restaurant dataset (make sure it's available in the same directory or adjust the path)
df = pd.read_csv('Dataset1.csv')

# Function to calculate similarity using the sigmoid kernel
def recommend_by_cuisine(preferred_cuisines, top_n=10):
    # Convert the user's preferred cuisines to a single string and then transform it using the TF-IDF vectorizer
    user_pref_str = " ".join(preferred_cuisines)
    user_pref_vector = tfidfvec.transform([user_pref_str])

    # Calculate the similarity scores using the sigmoid kernel
    sig = sigmoid_kernel(user_pref_vector, tfidf_matrix).flatten()

    # Sort the similarity scores in descending order and get the top N restaurants
    top_indices = sig.argsort()[::-1][:top_n]

    # Return top restaurants' details
    return df[['Restaurant Name', 'Cuisines', 'Aggregate rating']].iloc[top_indices].sort_values("Aggregate rating",ascending=False) 

# Set up Streamlit interface
sl.set_page_config(layout="centered")
sl.title("Restaurant Recommendation App")
sl.caption("This app helps you find restaurants based on your cuisine preferences!")
sl.divider()

# Input for preferred cuisine
preferred_cuisines_input = sl.text_input("Please Enter your preferred cuisine(s): ", "")

# Button to trigger recommendations
if sl.button("Get Recommendations") and preferred_cuisines_input:
    # Split the input string into a list of preferred cuisines
    preferred_cuisines = [cuisine.strip().lower() for cuisine in preferred_cuisines_input.split(",")]

    # Display the user's preferences
    sl.write(f"Your preferred cuisines: {', '.join(preferred_cuisines)}")

    # Get recommendations based on the input cuisines
    recommendations = recommend_by_cuisine(preferred_cuisines)

    if isinstance(recommendations, str):
        sl.write(recommendations)
    else:
        sl.write("Top 10 Restaurant Recommendations:")
        sl.dataframe(recommendations)
else:
    sl.write("Enter your preferred cuisines and press the button to get recommendations.")

sl.divider()

