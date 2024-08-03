import streamlit as st
import pandas as pd
import numpy as np
import openai
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the embeddings
embeddings = pd.read_pickle("embeddings.pkl")

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Function to get the most relevant articles
def get_relevant_articles(query, model="text-embedding-ada-002"):
    query_embedding = get_embedding(query, model=model)
    if query_embedding is None:
        return pd.DataFrame()  # Return empty DataFrame if embedding fails
    embeddings["similarity"] = embeddings["embedding"].apply(lambda x: np.dot(np.array(x), np.array(query_embedding)))
    relevant_articles = embeddings.sort_values(by="similarity", ascending=False).head(5)
    return relevant_articles[["title", "description", "date"]]  # Changed 'summary' to 'description'

# Streamlit app interface
st.title("Aluminum Industry News Chatbot")

query = st.text_input("Enter your query:")
if query:
    results = get_relevant_articles(query)
    if not results.empty:
        for index, row in results.iterrows():
            st.write(f"### {row['title']}")
            st.write(f"**Date:** {row['date']}")
            st.write(f"**Description:** {row['description']}")  # Changed 'summary' to 'description'
    else:
        st.write("No relevant articles found.")
