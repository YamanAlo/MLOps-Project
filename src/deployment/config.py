import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML Model API"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    
    # Model Settings
    MODELS_DIR: str = "models"
    DEFAULT_MODEL: str = "best_model"
    
  
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True) 