import openai
import pandas as pd
import numpy as np
import time
import requests
import json

# Function to fetch API key
def fetch_api_key(email):
    url = "http://52.66.239.27:8504/get_keys"
    headers = {'Content-Type': 'application/json'}
    payload = {"email": email}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Print the status code and response for debugging
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        
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

# Function to get embeddings
def get_embedding(text, email, model="text-embedding-ada-002"):
    api_key = fetch_api_key(email)
    if not api_key:
        raise ValueError("API key could not be retrieved.")
    
    openai.api_key = api_key
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response["data"][0]["embedding"]
    except openai.error.AuthenticationError as e:
        print(f"Error getting embedding: {e}")
        print("It seems the API key is invalid. Please check the key and try again.")
        return None
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Convert articles to embeddings
def get_embeddings(df, email, column_name="description"):
    embeddings = []
    for text in df[column_name]:
        embedding = get_embedding(text, email)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append([None]*len(df.columns))  # Handle errors gracefully
        time.sleep(1)  # Adding a delay to avoid rate limits
    return embeddings

# Load the articles
df = pd.read_csv("articles.csv")

# User email
email = "kbandla248@gmail.com"  # Replace with your email address

df["embedding"] = get_embeddings(df, email)

# Save the embeddings
embeddings = df[["title", "description", "date", "embedding"]]
embeddings.to_pickle("embeddings.pkl")

print("Embeddings have been saved to embeddings.pkl")
