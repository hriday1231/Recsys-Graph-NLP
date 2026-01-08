import numpy as np
import torch
import json
from pathlib import Path
from eval import get_metrics

# Paths
ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DIR = Path("training/data/processed")

def run_ablation():
    print("[INFO] Loading Artifacts...")
    user_emb = np.load(ARTIFACTS_DIR / "embeddings" / "user_embeddings.npy")
    item_emb_graph = np.load(ARTIFACTS_DIR / "embeddings" / "item_embeddings.npy")
    
    # Load splits
    train_edge_index = np.load(PROCESSED_DIR / "train_edge_index.npy")
    val_edge_index = np.load(PROCESSED_DIR / "val_edge_index.npy")

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIX: Use the correct variable name (item_emb_graph) on the right side
    user_emb_tensor = torch.tensor(user_emb, device=device)
    item_emb_tensor = torch.tensor(item_emb_graph, device=device)
    train_edges = torch.tensor(train_edge_index, device=device)
    val_edges = torch.tensor(val_edge_index, device=device)

    print("\n[EXPERIMENT] Pure LightGCN (Graph Only)...")
    metrics = get_metrics(user_emb_tensor, item_emb_tensor, train_edges, val_edges, k_list=[10, 20])
    
    print("-" * 40)
    print("METRIC        | SCORE")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:<13} | {v:.4f}")
    print("-" * 40)
    
    # Save results
    results = {"lightgcn_only": metrics}
    (Path("results")).mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[NOTE] Offline ablation complete.")

if __name__ == "__main__":
    run_ablation()