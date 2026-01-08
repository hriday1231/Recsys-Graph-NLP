# Graph + NLP Hybrid Recommender System

**Live Demo:** [https://recsys-frontend-793739516196.us-central1.run.app](https://recsys-frontend-793739516196.us-central1.run.app)

## 📌 Overview
This project implements a **Hybrid Recommender System** that bridges the gap between collaborative filtering signals (User-Item Interaction Graph) and semantic understanding (Natural Language Queries). It allows users to receive personalized recommendations based on their history, then "steer" those recommendations using natural language (e.g., *"cartoon robots"*) without losing the personalization signal.

## 🏗 Architecture
The system is built as a microservices architecture deployed on **Google Cloud Run**.

* **Graph Model:** LightGCN (PyTorch Geometric) trained on MovieLens 1M to capture high-order collaborative signals.
* **NLP Layer:** Sentence-BERT (`all-MiniLM-L6-v2`) for real-time semantic embedding and reranking.
* **Inference Engine:** FastAPI + Uvicorn with lazy-loading artifacts to optimize serverless cold-starts.
* **Frontend:** Next.js 16 (App Router) + TypeScript + Tailwind CSS.
* **Infrastructure:** Dockerized containers on GCP Cloud Run with automated builds via Cloud Build.

## 🔬 Methodology
### 1. Collaborative Signal (Graph)
We model user-item interactions as a bipartite graph. **LightGCN** (He et al., 2020) propagates user and item embeddings through the graph structure to capture "collaborative" similarity (e.g., *"Users who liked X also liked Y"*). We removed non-linearities to focus purely on graph convolution.

### 2. Semantic Reranking (Hybrid)
Pure graph models fail when users have explicit, transient intent (e.g., wanting a specific *vibe* right now).
* **Candidate Generation:** Retrieve top $N=200$ items via Graph Dot Product.
* **Query Encoding:** Encode user query $Q$ into vector space using SBERT.
* **Hybrid Scoring:**
    $$S_{final} = \alpha \cdot \text{norm}(S_{graph}) + (1-\alpha) \cdot \text{norm}(S_{semantic})$$
    where $\alpha$ allows dynamic balancing between long-term preference and immediate intent.

## 📊 Evaluation Results (Offline)
Models were trained on the **MovieLens 1M** dataset using a strict stratified split (80% Train, 10% Val, 10% Test) to prevent data leakage.

| Model Variant | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| **LightGCN (This Project)** | **0.2445** | **0.1778** |
| Matrix Factorization (Baseline) | 0.1924 | 0.1450 |

*Analysis: LightGCN outperforms standard MF by ~27% on Recall, demonstrating the effectiveness of high-order graph propagation in capturing sparse signals.*

## 🚀 How to Run Locally

### Prerequisites
* Python 3.11+
* Node.js 20+ (for Next.js 16)
* CUDA 12+ (Optional, for GPU training)

### 1. Training (Offline)
```
# Install dependencies
pip install -r requirements.txt

# Download & Preprocess Data
python training/src/download.py
python training/src/preprocess.py

# Train LightGCN (Generates artifacts/embeddings/)
python training/src/train_lightgcn.py

# Generate NLP Embeddings
python training/src/embed_items.py
```
### 2. Backend API
```
# Start the inference server
python -m uvicorn backend.app.main:app --reload
```
### 3. Frontend UI
```
cd frontend
npm install
npm run dev
# Access at http://localhost:3000
```