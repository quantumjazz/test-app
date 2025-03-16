import os
import csv
import pickle
import sys
import time
import numpy as np
from typing import List, Dict, Any

# For OpenAI embeddings
from openai import OpenAI

# For SentenceTransformer embeddings
from sentence_transformers import SentenceTransformer

def read_settings(settings_path: str) -> dict:
    """
    Reads simple key-value pairs from a settings.txt file.
    Expected format (one key=value per line).
    """
    settings = {}
    with open(settings_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            settings[key.strip()] = value.strip()
    return settings

def read_chopped_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Reads chunked data from a CSV file.
    Returns a list of dicts like:
      [
        {'filename': 'doc.pdf', 'chunk_index': 0, 'chunk_text': '...'},
        ...
      ]
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'filename': row['filename'],
                'chunk_index': int(row['chunk_index']),
                'chunk_text': row['chunk_text']
            })
    return data

def embed_with_openai(texts: List[str], model: str, max_tokens_per_batch: int, client: OpenAI) -> List[Dict[str, Any]]:
    """
    Batches texts based on a maximum token count and sends them to the OpenAI API.
    Uses a simple token count estimation (splitting on whitespace).
    """
    def count_tokens(text: str) -> int:
        return len(text.split())

    embeddings = []
    batch = []
    current_tokens = 0
    for text in texts:
        tokens = count_tokens(text)
        if current_tokens + tokens > max_tokens_per_batch:
            # Process the current batch
            response = client.embeddings.create(model=model, input=batch)
            embeddings.extend(response.data)
            batch = []
            current_tokens = 0
            print("Waiting 60 seconds before sending next batch...")
            time.sleep(60)
        batch.append(text)
        current_tokens += tokens
    if batch:
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(response.data)
    return embeddings

def generate_embeddings_openai(data: List[Dict[str, Any]], model_name: str, max_tokens_per_batch: int, client: OpenAI) -> List[Dict[str, Any]]:
    """
    Generates embeddings using OpenAI's API.
    """
    texts = [d['chunk_text'] for d in data]
    print(f"Generating embeddings using OpenAI model: {model_name} ...")
    embeddings_response = embed_with_openai(texts, model=model_name, max_tokens_per_batch=max_tokens_per_batch, client=client)
    for i, emb in enumerate(embeddings_response):
        data[i]['embedding'] = emb['embedding']
    return data

def generate_embeddings_sentence_transformer(data: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    """
    Generates embeddings using a SentenceTransformer model.
    """
    model = SentenceTransformer(model_name)
    texts = [d['chunk_text'] for d in data]
    print(f"Generating embeddings using SentenceTransformer model: {model_name} ...")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
    for i, emb in enumerate(embeddings):
        data[i]['embedding'] = emb.tolist()  # convert numpy array to list for serialization
    return data

def main():
    """
    Main routine:
      1. Reads chunked text from 'data/chopped_text.csv'.
      2. Generates embeddings using either OpenAI or SentenceTransformer (based on settings).
      3. Saves the resulting data (with text and embeddings) to 'data/embedded_data.pkl'.
    """
    # Determine the project base directory (parent of the 'scripts' folder)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_path = os.path.join(base_dir, 'settings.txt')
    data_dir = os.path.join(base_dir, 'data')
    chopped_csv_path = os.path.join(data_dir, 'chopped_text.csv')
    output_pickle_path = os.path.join(data_dir, 'embedded_data.pkl')

    # Read settings
    settings = read_settings(settings_path)

    # Determine which embedding method to use: "openai" or "sentence-transformers"
    embedding_method = settings.get("embedding_method", "sentence-transformers").lower()

    # Read chunked text data
    if not os.path.exists(chopped_csv_path):
        print(f"Chopped CSV file not found: {chopped_csv_path}. Please run your document preparation script first.")
        sys.exit(0)
    print(f"Reading chopped data from: {chopped_csv_path}")
    chopped_data = read_chopped_csv(chopped_csv_path)
    if not chopped_data:
        print("No data found in CSV. Exiting.")
        sys.exit(0)

    if embedding_method == "openai":
        # Load OpenAI API key from APIkey.txt
        api_key_path = os.path.join(base_dir, "APIkey.txt")
        if not os.path.exists(api_key_path):
            print("APIkey.txt not found. Exiting.")
            sys.exit(0)
        with open(api_key_path, 'r') as key_file:
            api_key = key_file.read().strip()
        client = OpenAI(api_key=api_key)  # Initialize the OpenAI client here.
        model_name = settings.get("openai_embedding_model", "text-embedding-ada-002")
        max_tokens_per_batch = int(settings.get("max_tokens_per_batch", 250000))
        embedded_data = generate_embeddings_openai(chopped_data, model_name, max_tokens_per_batch, client)
    else:
        # Default to SentenceTransformer embeddings
        model_name = settings.get("sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2")
        embedded_data = generate_embeddings_sentence_transformer(chopped_data, model_name)

    # Save the embedded data to a pickle file
    os.makedirs(data_dir, exist_ok=True)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(embedded_data, f)

    print(f"Successfully wrote embeddings to {output_pickle_path}")
    print("Sample record:", embedded_data[0])
    print("Done!")

if __name__ == "__main__":
    main()

