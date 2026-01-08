import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "RecSys Graph+NLP"
    VERSION: str = "1.0.0"
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    if os.path.exists("/app/artifacts"):
        ARTIFACTS_DIR = Path("/app/artifacts")
    else:
        ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
    
    EMBEDDING_PATH_USER = ARTIFACTS_DIR / "embeddings" / "user_embeddings.npy"
    EMBEDDING_PATH_ITEM = ARTIFACTS_DIR / "embeddings" / "item_embeddings.npy"
    EMBEDDING_PATH_TEXT = ARTIFACTS_DIR / "embeddings" / "item_text_embeddings.npy"
    METADATA_PATH = ARTIFACTS_DIR / "metadata" / "item_metadata.parquet"
    ID_MAP_PATH = ARTIFACTS_DIR / "metadata" / "id_maps.json"

settings = Settings()