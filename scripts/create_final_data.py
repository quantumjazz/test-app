import os
import pickle
import faiss
import numpy as np
import json
import sys
from typing import List, Dict, Any
from typing import Tuple


def build_faiss_index(embedded_data: List[Dict[str, Any]], embedding_dim: int) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """
    Builds a FAISS index from the embedded data.
    
    Returns:
      - The FAISS index built with L2 distance.
      - A metadata list where each entry includes:
          'filename', 'chunk_index', and 'chunk_text'
    """
    vectors = []
    metadata = []

    for record in embedded_data:
        # Ensure the embedding is a float32 numpy array (FAISS requires float32)
        embedding = np.array(record['embedding'], dtype=np.float32)
        vectors.append(embedding)
        metadata.append({
            'filename': record['filename'],
            'chunk_index': record['chunk_index'],
            'chunk_text': record['chunk_text']
        })

    vectors_np = np.vstack(vectors)
    index = faiss.IndexFlatL2(embedding_dim)  # Simple L2 distance index; for large datasets, consider IVF or HNSW
    index.add(vectors_np)

    return index, metadata

def main():
    """
    Main routine:
      1. Loads the embedded data from 'data/embedded_data.pkl'
      2. Builds a FAISS index and corresponding metadata structure.
      3. Saves the FAISS index as 'faiss_index.bin' and metadata as 'faiss_metadata.json' in the 'data/' folder.
    """
    # Set base directory (parent of scripts/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    embedded_data_path = os.path.join(data_dir, 'embedded_data.pkl')

    if not os.path.exists(embedded_data_path):
        print(f"Could not find {embedded_data_path}. Please run your embedding script first.")
        sys.exit(0)

    # 1. Load the embedded data
    print(f"Loading embedded data from {embedded_data_path}...")
    with open(embedded_data_path, 'rb') as f:
        embedded_data = pickle.load(f)
    if not embedded_data:
        print("No embedded data found. Exiting.")
        sys.exit(0)

    # Determine embedding dimension from the first record
    first_embedding = embedded_data[0]['embedding']
    embedding_dim = len(first_embedding)
    print(f"Detected embedding dimension: {embedding_dim}")

    # 2. Build the FAISS index
    print("Building FAISS index...")
    faiss_index, metadata_list = build_faiss_index(embedded_data, embedding_dim)
    print(f"Index built and populated with {len(metadata_list)} vectors.")

    # 3. Save the FAISS index and metadata
    faiss_index_path = os.path.join(data_dir, 'faiss_index.bin')
    metadata_path = os.path.join(data_dir, 'faiss_metadata.json')

    print(f"Saving FAISS index to {faiss_index_path}...")
    faiss.write_index(faiss_index, faiss_index_path)

    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print("Done! FAISS index and metadata are ready for retrieval.")

if __name__ == "__main__":
    main()
