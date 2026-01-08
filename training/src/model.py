import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 0-th layer embeddings (the learnable part)
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        
        # Initialize weights (Normal distribution is standard for LightGCN)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # LightGCN convolution layers (no weights, just aggregation)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index):
        # 1. Start with initial embeddings (E0)
        emb = self.embedding.weight
        embs = [emb]

        # 2. Propagate messages (E1, E2, ... Ek)
        for conv in self.convs:
            emb = conv(emb, edge_index)
            embs.append(emb)

        # 3. Average all layers (Standard LightGCN combination)
        # Final = (1/(K+1)) * sum(E0...Ek)
        out = torch.stack(embs, dim=1)
        out = torch.mean(out, dim=1)

        # Split back into users and items
        user_emb, item_emb = torch.split(out, [self.num_users, self.num_items])
        return user_emb, item_emb