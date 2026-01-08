from pydantic import BaseModel
from typing import List, Optional

class ItemResponse(BaseModel):
    id: int
    title: str
    genres: str
    score: float
    explanation: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: int
    items: List[ItemResponse]