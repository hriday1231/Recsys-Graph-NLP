import torch
import numpy as np

def get_metrics(user_emb, item_emb, train_edge_index, eval_edge_index, k_list=[10, 20]):
    """
    Computes Recall@K and NDCG@K.
    """
    user_emb = user_emb.detach()
    item_emb = item_emb.detach()
    
    device = user_emb.device
    num_users = user_emb.shape[0]
    num_items = item_emb.shape[0]

    # Group ground truth by user
    # eval_edge_index is [2, Num_Edges], row 0 is users, row 1 is items
    # We create a list where eval_dict[u] = {set of true items}
    import collections
    eval_dict = collections.defaultdict(set)
    for u, i in eval_edge_index.T.cpu().numpy():
        eval_dict[u].add(i)

    # Also track training items to mask them out (don't recommend what they already saw)
    train_dict = collections.defaultdict(set)
    for u, i in train_edge_index.T.cpu().numpy():
        train_dict[u].add(i)

    # Metrics accumulators
    recall = {k: [] for k in k_list}
    ndcg = {k: [] for k in k_list}

    # Evaluate in batches to save VRAM
    BATCH_SIZE = 1024
    users_to_eval = list(eval_dict.keys())
    
    for i in range(0, len(users_to_eval), BATCH_SIZE):
        batch_users = users_to_eval[i : i + BATCH_SIZE]
        
        # 1. Compute scores for this batch: (Batch, Emb) @ (Item, Emb)^T -> (Batch, Num_Items)
        batch_user_emb = user_emb[batch_users]
        scores = torch.matmul(batch_user_emb, item_emb.t())

        # 2. Mask training items (set score to -infinity)
        for idx, u in enumerate(batch_users):
            if u in train_dict:
                mask_items = list(train_dict[u])
                scores[idx, mask_items] = -float('inf')

        # 3. Get Top-K largest scores (we take max K)
        max_k = max(k_list)
        _, topk_indices = torch.topk(scores, k=max_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        # 4. Compute metrics
        for idx, u in enumerate(batch_users):
            ground_truth = eval_dict[u]
            if len(ground_truth) == 0:
                continue

            hits = [1 if item in ground_truth else 0 for item in topk_indices[idx]]
            
            for k in k_list:
                # Recall
                k_hits = hits[:k]
                recall[k].append(sum(k_hits) / min(k, len(ground_truth)))
                
                # NDCG
                dcg = np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(k_hits)])
                idcg = np.sum([1 / np.log2(idx + 2) for idx in range(min(k, len(ground_truth)))])
                ndcg[k].append(dcg / idcg if idcg > 0 else 0)

    # Average over users
    return {
        f"Recall@{k}": np.mean(recall[k]) for k in k_list
    } | {
        f"NDCG@{k}": np.mean(ndcg[k]) for k in k_list
    }