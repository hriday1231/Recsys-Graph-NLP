import os
import pandas as pd
import numpy as np
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm

# Paths
RAW_DIR = Path("training/data/raw/ml-1m")
PROCESSED_DIR = Path("training/data/processed")
ARTIFACTS_DIR = Path("artifacts")

def load_data():
    """Loads ratings and movies from .dat files (MovieLens specific encoding)."""
    # Ratings: UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(
        RAW_DIR / "ratings.dat", 
        sep="::", 
        engine="python", 
        names=["uid", "mid", "rating", "timestamp"],
        encoding="ISO-8859-1"
    )
    
    # Movies: MovieID::Title::Genres
    movies = pd.read_csv(
        RAW_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["mid", "title", "genres"],
        encoding="ISO-8859-1"
    )
    return ratings, movies

def preprocess(config_path):
    print(f"[INFO] Preprocessing started...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metadata").mkdir(parents=True, exist_ok=True)

    ratings, movies = load_data()
    print(f"[INFO] Raw ratings: {len(ratings)}")

    # 1. Convert to Implicit Feedback (Rating >= 4 is a positive interaction)
    # We only keep positive interactions for the graph.
    ratings = ratings[ratings['rating'] >= 4].copy()
    print(f"[INFO] Positive interactions (>=4): {len(ratings)}")

    # 2. Filter users with too few interactions (Iterative to ensure graph consistency)
    min_interactions = 10
    print(f"[INFO] Filtering users with < {min_interactions} interactions...")
    
    user_counts = ratings['uid'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    ratings = ratings[ratings['uid'].isin(valid_users)].copy()
    print(f"[INFO] Ratings after filtering: {len(ratings)}")

    # 3. Create Contiguous IDs (0 to N-1)
    # We need mappings for the UI later.
    unique_users = ratings['uid'].unique()
    unique_items = ratings['mid'].unique()

    user_id_map = {id: i for i, id in enumerate(unique_users)} # raw -> idx
    item_id_map = {id: i for i, id in enumerate(unique_items)} # raw -> idx
    
    # Reverse maps for inference
    item_id_map_inv = {i: id for id, i in item_id_map.items()}

    ratings['user_idx'] = ratings['uid'].map(user_id_map)
    ratings['item_idx'] = ratings['mid'].map(item_id_map)

    # 4. Save Metadata (for UI/Inference)
    # Filter movies to only those that exist in our filtered interactions
    movies = movies[movies['mid'].isin(unique_items)].copy()
    movies['item_idx'] = movies['mid'].map(item_id_map)
    
    # Save mappings
    with open(ARTIFACTS_DIR / "metadata" / "id_maps.json", "w") as f:
        json.dump({
            "user_to_idx": {str(k): int(v) for k,v in user_id_map.items()},
            "item_to_idx": {str(k): int(v) for k,v in item_id_map.items()},
            "idx_to_item": {str(k): int(v) for k,v in item_id_map_inv.items()}
        }, f)
    
    # Save rich metadata as parquet
    movies.to_parquet(ARTIFACTS_DIR / "metadata" / "item_metadata.parquet", index=False)

    # 5. Perform Stratified Split (80/10/10)
    print("[INFO] Splitting data (80/10/10 per user)...")
    
    train_list, val_list, test_list = [], [], []

    # Group by user and split indices
    # Using numpy per group is much faster than pandas apply for 6k users
    grouped = ratings.groupby('user_idx')
    
    for user_idx, group in tqdm(grouped, desc="Splitting users"):
        # Shuffle interactions for this user
        interactions = group['item_idx'].values
        np.random.shuffle(interactions)
        
        n = len(interactions)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        # n_test is remainder
        
        train_items = interactions[:n_train]
        val_items = interactions[n_train:n_train+n_val]
        test_items = interactions[n_train+n_val:]
        
        # Add to lists (User, Item)
        train_list.append(np.stack([np.full(len(train_items), user_idx), train_items], axis=1))
        val_list.append(np.stack([np.full(len(val_items), user_idx), val_items], axis=1))
        test_list.append(np.stack([np.full(len(test_items), user_idx), test_items], axis=1))

    # Concatenate and save
    train_edge_index = np.vstack(train_list).T # Shape: (2, Num_Edges)
    val_edge_index = np.vstack(val_list).T
    test_edge_index = np.vstack(test_list).T
    
    print(f"[STATS] Train Edges: {train_edge_index.shape[1]}")
    print(f"[STATS] Val Edges:   {val_edge_index.shape[1]}")
    print(f"[STATS] Test Edges:  {test_edge_index.shape[1]}")

    np.save(PROCESSED_DIR / "train_edge_index.npy", train_edge_index)
    np.save(PROCESSED_DIR / "val_edge_index.npy", val_edge_index)
    np.save(PROCESSED_DIR / "test_edge_index.npy", test_edge_index)
    
    # Save global stats
    stats = {
        "num_users": len(unique_users),
        "num_items": len(unique_items)
    }
    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f)
        
    print("[SUCCESS] Data processing complete.")

if __name__ == "__main__":
    preprocess(None)