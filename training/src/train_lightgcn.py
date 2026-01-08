import torch
import torch.optim as optim
import numpy as np
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from model import LightGCN
from eval import get_metrics

# Paths
PROCESSED_DIR = Path("training/data/processed")
ARTIFACTS_DIR = Path("artifacts")

def train(config_path=None):
    # Hyperparams
    EMBEDDING_DIM = 64
    LR = 0.001
    EPOCHS = 200   # Enough for LightGCN convergence
    BATCH_SIZE = 4096 * 4 # Large batch for your 5070 Ti
    K_NEG = 1  # 1 negative per positive

    # Load Data
    print("[INFO] Loading data...")
    train_edge_index = np.load(PROCESSED_DIR / "train_edge_index.npy")
    val_edge_index = np.load(PROCESSED_DIR / "val_edge_index.npy")
    
    with open(PROCESSED_DIR / "stats.json") as f:
        stats = json.load(f)
    num_users = stats["num_users"]
    num_items = stats["num_items"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    # Convert to Tensor & Move to GPU
    # Graph needs to be bidirectional for LightGCN message passing (User <-> Item)
    # But for BPR Loss, we only iterate over (User, Pos_Item) edges.
    
    # 1. Create full graph for Message Passing
    train_edge_index_tensor = torch.LongTensor(train_edge_index)
    # Add reverse edges (Item -> User) to make it undirected
    # Row 0: Src, Row 1: Dst. We flip and concat.
    # Note: Users are 0..N-1, Items are 0..M-1. We need to shift item indices for PyG?
    # NO. PyG LightGCN usually expects a bipartite graph where nodes are 0...(N+M).
    # We must offset item indices by num_users.
    
    print("[INFO] Adjusting graph for PyG (offsetting item IDs)...")
    train_edges_u = train_edge_index_tensor[0]
    train_edges_i = train_edge_index_tensor[1] + num_users # Shift items
    
    edge_index = torch.stack([
        torch.cat([train_edges_u, train_edges_i]),
        torch.cat([train_edges_i, train_edges_u])
    ], dim=0).to(device)

    # Model
    model = LightGCN(num_users, num_items, embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    print("[INFO] Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Shuffle training edges
        indices = np.random.permutation(train_edge_index.shape[1])
        
        # Batch Loop
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            
            # Get Users and Pos Items
            users = torch.LongTensor(train_edge_index[0, batch_idx]).to(device)
            pos_items = torch.LongTensor(train_edge_index[1, batch_idx]).to(device)
            
            # Negative Sampling (Random items)
            # We blindly sample; if we hit a true positive, noise is minimal at this scale.
            neg_items = torch.randint(0, num_items, (len(users),), device=device)

            # Forward Pass (Get final embeddings)
            # Note: Model internally handles the shift if we pass the correct edge_index
            user_emb_final, item_emb_final = model(edge_index)

            # Look up specific embeddings
            u_emb = user_emb_final[users]
            pos_emb = item_emb_final[pos_items]
            neg_emb = item_emb_final[neg_items]

            # BPR Loss
            # loss = -ln(sigmoid(pos_score - neg_score)) + reg
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            
            # L2 Regularization (optional but recommended)
            reg_loss = (1/2) * (u_emb.norm(2).pow(2) + 
                                pos_emb.norm(2).pow(2) + 
                                neg_emb.norm(2).pow(2)) / float(len(users))
            
            final_loss = loss + 1e-4 * reg_loss

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            total_loss += final_loss.item()

        # Evaluation (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model(edge_index)
                metrics = get_metrics(
                    user_emb, item_emb, 
                    torch.LongTensor(train_edge_index).to(device), 
                    torch.LongTensor(val_edge_index).to(device)
                )
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Recall@20: {metrics['Recall@20']:.4f} | Val NDCG@20: {metrics['NDCG@20']:.4f}")

    # Save Artifacts
    print("[INFO] Saving artifacts...")
    model.eval()
    with torch.no_grad():
        final_user_emb, final_item_emb = model(edge_index)
        
    (ARTIFACTS_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "model").mkdir(parents=True, exist_ok=True)

    np.save(ARTIFACTS_DIR / "embeddings" / "user_embeddings.npy", final_user_emb.cpu().numpy())
    np.save(ARTIFACTS_DIR / "embeddings" / "item_embeddings.npy", final_item_emb.cpu().numpy())
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model" / "lightgcn.pt")
    print("[SUCCESS] Training complete.")

if __name__ == "__main__":
    train() 