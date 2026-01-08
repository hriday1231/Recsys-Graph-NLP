import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

# Paths
ARTIFACTS_DIR = Path("artifacts")
METADATA_PATH = ARTIFACTS_DIR / "metadata" / "item_metadata.parquet"
OUT_PATH = ARTIFACTS_DIR / "embeddings" / "item_text_embeddings.npy"

def generate_embeddings():
    print("[INFO] Loading metadata...")
    df = pd.read_parquet(METADATA_PATH)
    
    # Create rich text representation
    # "Toy Story (1995)" + "Animation|Children's" -> "Toy Story (1995). Genre: Animation, Children's"
    df['text'] = df['title'] + ". Genre: " + df['genres'].str.replace('|', ', ')
    
    print(f"[INFO] Encoding {len(df)} items with SBERT (all-MiniLM-L6-v2)...")
    
    # Load SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    # Encode (this handles batching automatically)
    embeddings = model.encode(
        df['text'].tolist(), 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True # Crucial for cosine similarity
    )
    
    # We need to ensure the embeddings are saved in the order of item_idx (0..M-1)
    # The dataframe might be unordered, but our preprocessing ensured item_idx is a column.
    # Let's sort by item_idx to be safe.
    df['embedding'] = list(embeddings)
    df = df.sort_values('item_idx')
    
    # Extract sorted stack
    final_embeddings = np.stack(df['embedding'].values)
    
    print(f"[INFO] Saving text embeddings shape: {final_embeddings.shape}")
    np.save(OUT_PATH, final_embeddings)
    print("[SUCCESS] NLP Embeddings generated.")

if __name__ == "__main__":
    generate_embeddings()