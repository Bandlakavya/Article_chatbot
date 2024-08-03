import openai
import pandas as pd
import numpy as np
import time
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to fetch API key
def fetch_api_key(email):
    url = "http://52.66.239.27:8504/get_keys"
    headers = {'Content-Type': 'application/json'}
    payload = {"email": email}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            api_key = response.json().get("key")  # Correct key name
            if api_key:
                return api_key
            else:
                print("API Key not found in the response.")
                return None
        else:
            print("Failed to fetch API key. Status code:", response.status_code)
            return None
    except Exception as e:
        print(f"Error during API key fetching: {e}")
        return None

# Load the API key
email = os.getenv("USER_EMAIL")  # Ensure USER_EMAIL is set in your .env file
api_key = fetch_api_key(email)
if api_key:
    openai.api_key = api_key
else:
    raise ValueError("API key could not be retrieved.")

# Load the articles
df = pd.read_csv("articles.csv")

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Convert articles to embeddings
def get_embeddings(df, column_name="description"):
    embeddings = []
    for text in df[column_name]:
        embedding = get_embedding(text)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append([None]*len(df.columns))  # Handle errors gracefully
        time.sleep(1)  # Adding a delay to avoid rate limits
    return embeddings

df["embedding"] = get_embeddings(df)

# Save the embeddings
embeddings = df[["title", "description", "date", "embedding"]]
embeddings.to_pickle("embeddings.pkl")

print("Embeddings have been saved to embeddings.pkl")