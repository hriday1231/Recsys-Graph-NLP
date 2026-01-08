# Graph + NLP Hybrid Recommender System

**Live Demo:** [https://recsys-frontend-793739516196.us-central1.run.app](https://recsys-frontend-793739516196.us-central1.run.app)

## 📌 Overview
This project implements a **Hybrid Recommender System** that combines collaborative filtering signals (User-Item Interaction Graph) with semantic understanding (Natural Language Queries). It bridges the gap between classic matrix factorization and modern NLP-driven search.

## 🏗 Architecture
* **Graph Model:** LightGCN (PyTorch Geometric) trained on MovieLens 1M.
* **NLP Layer:** Sentence-BERT (`all-MiniLM-L6-v2`) for semantic reranking.
* **Inference:** FastAPI + Uvicorn (Lazy-loading artifacts for optimizing Cloud Run cold-starts).
* **Frontend:** Next.js 16 + TypeScript + Tailwind CSS.
* **Infrastructure:** Dockerized containers deployed on Google Cloud Run (Serverless).

## 🔬 Methodology
1.  **Collaborative Signal:** We construct a bipartite graph of users and movies. LightGCN propagates embeddings to capture high-order connectivity (e.g., "users like you also liked...").
2.  **Semantic Reranking:**
    * **Baseline:** Top-200 items retrieved via Graph Dot Product.
    * **Rerank:** The user's query (e.g., *"cartoon robots"*) is encoded into vector space.
    * **Hybrid Score:** $S_{final} = \alpha \cdot S_{graph} + (1-\alpha) \cdot S_{semantic}$

## 📊 Evaluation Results (Offline)
Trained on MovieLens 1M (80/10/10 Stratified Split).

| Model Variant | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| **LightGCN (Final)** | **0.2398** | **0.1762** |
| Matrix Factorization | 0.1924 | 0.1450 |

*Analysis: LightGCN outperforms standard MF by ~24% on Recall, proving the value of graph convolution for capturing sparse signals.*

## 🚀 How to Run Locally

### 1. Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start Inference Server
python -m uvicorn backend.app.main:app --reload