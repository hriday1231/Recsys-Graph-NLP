from fastapi import APIRouter, HTTPException, Query
from app.recsys.engine import engine
from app.schemas.models import RecommendationResponse

router = APIRouter()

def ensure_loaded():
    if engine.user_emb is None:
        print("[INFO] Lazy loading model on first request...", flush=True)
        engine.load_artifacts()

@router.get("/health")
def health_check():
    """Health check that doesn't require model to be loaded"""
    return {
        "status": "ok", 
        "model_loaded": engine.user_emb is not None,
        "ready": True
    }

@router.get("/users")
def get_sample_users(limit: int = 50):
    ensure_loaded()
    
    if engine.user_emb is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [{"user_id": i} for i in range(min(limit, len(engine.user_emb)))]

@router.get("/recommend", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10):
    ensure_loaded()
    
    if user_id >= len(engine.user_emb):
        raise HTTPException(status_code=404, detail="User not found")
    
    items = engine.recommend(user_id, k=k)
    return {"user_id": user_id, "items": items}

@router.get("/recommend_query", response_model=RecommendationResponse)
def recommend_query(
    user_id: int, 
    q: str, 
    k: int = 10, 
    alpha: float = Query(0.5, ge=0.0, le=1.0)
):
    ensure_loaded()
    
    if user_id >= len(engine.user_emb):
        raise HTTPException(status_code=404, detail="User not found")

    items = engine.recommend_hybrid(user_id, query=q, alpha=alpha, k=k)
    return {"user_id": user_id, "items": items}