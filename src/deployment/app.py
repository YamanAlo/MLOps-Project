import os
import sys
import logging
from typing import List, Dict, Optional
from datetime import datetime
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import joblib
import uvicorn
import tempfile

from .config import settings
from .utils import (
    validate_input_features,
    load_model_with_metadata,
    get_model_info,
    validate_predictions,
    sanitize_for_json
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for making predictions using trained ML models",
    version="1.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Pydantic models for request/response
class PredictionInput(BaseModel):
    features: List[List[float]] = Field(..., description="List of feature vectors for prediction")
    model_name: Optional[str] = Field(settings.DEFAULT_MODEL, description="Name of the model to use")

class PredictionResponse(BaseModel):
    predictions: List[float] = Field(..., description="Model predictions")
    model_used: str = Field(..., description="Name of the model used")
    prediction_time: str = Field(..., description="Timestamp of prediction")
    request_id: str = Field(..., description="Unique request identifier")
    model_info: Dict = Field(..., description="Information about the model used")

# Add a new Pydantic model for specific housing prediction input
class HousingPredictionInput(BaseModel):
    MedInc: float = Field(..., example=3.8716, description="Median income in block group")
    HouseAge: float = Field(..., example=21.0, description="Median house age in block group")
    AveRooms: float = Field(..., example=5.819, description="Average number of rooms per household")
    AveBedrms: float = Field(..., example=1.023, description="Average number of bedrooms per household")
    Population: float = Field(..., example=1384.0, description="Block group population")
    AveOccup: float = Field(..., example=2.523, description="Average number of household members")
    Latitude: float = Field(..., example=34.26, description="Block group latitude")
    Longitude: float = Field(..., example=-118.56, description="Block group longitude")

class HousingPredictionResponse(BaseModel):
    predicted_median_house_value: float
    model_name: str
    model_stage: str
    model_version: Optional[str] = None
    request_timestamp: str

# Global variables
MODELS: Dict[str, tuple] = {}  # Will store (model, metadata) tuples
# Globals for our specific housing model and preprocessors
HOUSING_MODEL = None
HOUSING_MODEL_VERSION = None
IMPUTER = None
SCALER = None
EXPECTED_FEATURE_ORDER_FOR_SCALER = None
PREPROCESS_ARTIFACTS_RUN_ID = "ef890bec678b42fc94f1042e06d3db65" # As used in performance_monitor

@app.on_event("startup")
async def startup_event():
    """Load all available models on startup"""
    logger.info("Loading models on startup...")
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # List all joblib files in models directory
        model_files = [f for f in os.listdir(settings.MODELS_DIR) if f.endswith('.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            try:
                model_path = os.path.join(settings.MODELS_DIR, model_file)
                model, metadata = load_model_with_metadata(model_path)
                MODELS[model_name] = (model, metadata)
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        if not MODELS:
            logger.warning("No models were loaded!")
        else:
            logger.info(f"Loaded {len(MODELS)} models: {list(MODELS.keys())}")

        # Load housing model and preprocessors
        try:
            global HOUSING_MODEL, HOUSING_MODEL_VERSION, IMPUTER, SCALER, EXPECTED_FEATURE_ORDER_FOR_SCALER
            # Initialize MLflow client
            client = mlflow.tracking.MlflowClient()
            # Load housing model from MLflow Model Registry
            housing_model_uri = f"models:/xgboost_model/Production"
            HOUSING_MODEL = mlflow.pyfunc.load_model(housing_model_uri)
            # Get model version
            version_details = client.get_latest_versions("xgboost_model", stages=["Production"])[0]
            HOUSING_MODEL_VERSION = version_details.version

            # Download preprocessing artifacts
            temp_dir = tempfile.mkdtemp()
            imputer_path = client.download_artifacts(PREPROCESS_ARTIFACTS_RUN_ID, "imputer.joblib", temp_dir)
            scaler_path = client.download_artifacts(PREPROCESS_ARTIFACTS_RUN_ID, "scaler.joblib", temp_dir)
            feature_names_path = client.download_artifacts(PREPROCESS_ARTIFACTS_RUN_ID, "feature_names.txt", temp_dir)

            # Load preprocessors and expected feature order
            IMPUTER = joblib.load(imputer_path)
            SCALER = joblib.load(scaler_path)
            with open(feature_names_path, 'r') as f:
                EXPECTED_FEATURE_ORDER_FOR_SCALER = [line.strip() for line in f.readlines()]

            logger.info("Successfully loaded housing model and preprocessors.")
        except Exception as e:
            logger.error(f"Failed to load housing model or preprocessors: {e}")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # raise # Commenting out raise to allow app to start even if some models fail

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API is running",
        "available_models": list(MODELS.keys()),
        "total_models": len(MODELS),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(MODELS) > 0
    }

@app.get("/models")
async def list_models():
    """List all available models with their metadata"""
    response = {
        model_name: {
            "info": get_model_info(model),
            "metadata": metadata
        }
        for model_name, (model, metadata) in MODELS.items()
    }
    return sanitize_for_json(response)



# Original feature columns as used in performance_monitor.py
ORIGINAL_HOUSING_FEATURE_COLUMNS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

@app.post("/predict_housing_value", response_model=HousingPredictionResponse)
async def predict_housing_value(input_data: HousingPredictionInput):
    if not HOUSING_MODEL or not IMPUTER or not SCALER or not EXPECTED_FEATURE_ORDER_FOR_SCALER:
        logger.error("Housing model or necessary preprocessors are not loaded. Cannot predict.")
        raise HTTPException(status_code=503, detail="Housing prediction service is not ready due to missing model or preprocessors.")

    try:
        
        raw_features_df = pd.DataFrame([input_data.dict()], columns=ORIGINAL_HOUSING_FEATURE_COLUMNS)
        
        
        imputed_features_array = IMPUTER.transform(raw_features_df)
        features_imputed_df = pd.DataFrame(imputed_features_array, columns=ORIGINAL_HOUSING_FEATURE_COLUMNS, index=raw_features_df.index)

        features_engineered_df = features_imputed_df.copy()
        features_engineered_df['BedroomRatio'] = features_engineered_df['AveBedrms'] / (features_engineered_df['AveRooms'] + 1e-6)
        features_engineered_df['PopulationDensity'] = features_engineered_df['Population'] / (features_engineered_df['AveOccup'] + 1e-6)

        
        try:
            features_for_scaling_df = features_engineered_df[EXPECTED_FEATURE_ORDER_FOR_SCALER]
        except KeyError as e:
            logger.error(f"Missing columns for scaling: {e}. Expected: {EXPECTED_FEATURE_ORDER_FOR_SCALER}. Available: {list(features_engineered_df.columns)}")
            raise HTTPException(status_code=400, detail=f"Input data processing error: Missing columns for scaling. {e}")

        scaled_features_array = SCALER.transform(features_for_scaling_df)
        
        final_features_df = pd.DataFrame(scaled_features_array, columns=EXPECTED_FEATURE_ORDER_FOR_SCALER, index=features_for_scaling_df.index)

        # Make Prediction
        prediction = HOUSING_MODEL.predict(final_features_df)
        
        # Prediction might be a numpy array or list with one element
        predicted_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)

        return HousingPredictionResponse(
            predicted_median_house_value=predicted_value,
            model_name="xgboost_model", 
            model_stage="Production",
            model_version=str(HOUSING_MODEL_VERSION),
            request_timestamp=datetime.now().isoformat()
        )

    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error during housing value prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    ) 