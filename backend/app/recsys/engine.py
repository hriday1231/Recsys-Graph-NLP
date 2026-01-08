import numpy as np
import pandas as pd
import json
import torch
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import sys

class RecommendationEngine:
    def __init__(self):
        self.user_emb = None
        self.item_emb = None
        self.item_text_emb = None
        self.metadata = None
        self.id_map = None
        self.sbert = None
        self.device = "cpu"

    def load_artifacts(self):
        print("[INFO] Loading artifacts into memory...", flush=True)
        sys.stdout.flush()
        
        try:
            print(f"[INFO] Loading user embeddings from: {settings.EMBEDDING_PATH_USER}", flush=True)
            self.user_emb = np.load(settings.EMBEDDING_PATH_USER)
            print(f"[SUCCESS] Loaded user embeddings: {self.user_emb.shape}", flush=True)
            
            print(f"[INFO] Loading item embeddings from: {settings.EMBEDDING_PATH_ITEM}", flush=True)
            self.item_emb = np.load(settings.EMBEDDING_PATH_ITEM)
            print(f"[SUCCESS] Loaded item embeddings: {self.item_emb.shape}", flush=True)
            
            print(f"[INFO] Loading text embeddings from: {settings.EMBEDDING_PATH_TEXT}", flush=True)
            self.item_text_emb = np.load(settings.EMBEDDING_PATH_TEXT)
            print(f"[SUCCESS] Loaded text embeddings: {self.item_text_emb.shape}", flush=True)
            
            print(f"[INFO] Loading metadata from: {settings.METADATA_PATH}", flush=True)
            self.metadata = pd.read_parquet(settings.METADATA_PATH)
            self.meta_dict = self.metadata.set_index('item_idx').to_dict('index')
            print(f"[SUCCESS] Loaded metadata: {len(self.metadata)} items", flush=True)

            print(f"[INFO] Loading ID maps from: {settings.ID_MAP_PATH}", flush=True)
            with open(settings.ID_MAP_PATH) as f:
                self.id_map = json.load(f)
            print(f"[SUCCESS] Loaded ID maps", flush=True)
                
            print("[INFO] Loading SBERT for query encoding (from cache)...", flush=True)
            self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print("[SUCCESS] Engine ready.", flush=True)
            
        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise

    def recommend(self, user_idx: int, k: int = 10):
        u_vec = self.user_emb[user_idx]
        scores = np.dot(self.item_emb, u_vec)
        
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            meta = self.meta_dict.get(idx, {"title": "Unknown", "genres": "Unknown"})
            results.append({
                "id": int(idx),
                "title": meta['title'],
                "genres": meta['genres'],
                "score": float(scores[idx]),
                "explanation": "Based on your viewing history"
            })
        return results

    def recommend_hybrid(self, user_idx: int, query: str, alpha: float = 0.5, k: int = 10):
        N_CANDIDATES = 200
        u_vec = self.user_emb[user_idx]
        graph_scores = np.dot(self.item_emb, u_vec)
        
        candidate_indices = np.argpartition(graph_scores, -N_CANDIDATES)[-N_CANDIDATES:]
        
        query_vec = self.sbert.encode(query, convert_to_numpy=True)
        
        cand_text_vecs = self.item_text_emb[candidate_indices]
        
        text_scores = np.dot(cand_text_vecs, query_vec)
        
        g_scores_cand = graph_scores[candidate_indices]
        
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9)

        g_norm = normalize(g_scores_cand)
        t_norm = normalize(text_scores)
        
        final_scores = alpha * g_norm + (1 - alpha) * t_norm
        
        top_k_local = np.argsort(final_scores)[::-1][:k]
        top_indices = candidate_indices[top_k_local]
        
        results = []
        for i, idx in enumerate(top_indices):
            meta = self.meta_dict.get(idx, {"title": "Unknown", "genres": "Unknown"})
            local_idx = top_k_local[i]
            
            reason = "Hybrid Match"
            if t_norm[local_idx] > 0.8:
                reason = f"Strong match for '{query}'"
            elif g_norm[local_idx] > 0.8:
                reason = "Strongly aligns with your history"
                
            results.append({
                "id": int(idx),
                "title": meta['title'],
                "genres": meta['genres'],
                "score": float(final_scores[local_idx]),
                "explanation": reason
            })
            
        return results

engine = RecommendationEngine()