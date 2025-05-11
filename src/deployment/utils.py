import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import joblib
import mlflow
from datetime import datetime

from .config import settings

logger = logging.getLogger(__name__)

def sanitize_for_json(obj):

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def validate_input_features(features: List[List[float]], expected_shape: Tuple[int, ...] = None) -> np.ndarray:
   
    try:
        features_array = np.array(features)
        
        # Check if features are numeric
        if not np.issubdtype(features_array.dtype, np.number):
            raise ValueError("Features must be numeric")
            
        # Validate shape if provided
        if expected_shape and features_array.shape[1:] != expected_shape[1:]:
            raise ValueError(f"Expected feature shape {expected_shape}, got {features_array.shape}")
            
        return features_array
        
    except Exception as e:
        logger.error(f"Error validating input features: {e}")
        raise

def load_model_with_metadata(model_path: str) -> Tuple[Any, Dict[str, Any]]:
  
    try:
        # Load the model
        model = joblib.load(model_path)
        
        metadata = {}
        try:
            runs = mlflow.search_runs(filter_string=f"params.model_path = '{model_path}'")
            if not runs.empty:
                latest_run = runs.iloc[0]
                metadata = {
                    "run_id": latest_run.run_id,
                    "metrics": {k: v for k, v in latest_run.items() if k.startswith("metrics.")},
                    "params": {k: v for k, v in latest_run.items() if k.startswith("params.")},
                    "tags": {k: v for k, v in latest_run.items() if k.startswith("tags.")}
                }
        except Exception as e:
            logger.warning(f"Could not fetch MLflow metadata: {e}")
        
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def get_model_info(model: Any) -> Dict[str, Any]:
  
    try:
        return {
            "type": type(model).__name__,
            "parameters": getattr(model, "get_params", lambda: {})(),
            "features": getattr(model, "n_features_in_", None),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.warning(f"Could not get complete model info: {e}")
        return {
            "type": type(model).__name__,
            "timestamp": datetime.now().isoformat()
        }

def validate_predictions(predictions: np.ndarray) -> List[float]:
   
    try:
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
            
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            raise ValueError("Predictions contain invalid values (NaN or Inf)")
            
        return predictions.tolist()
        
    except Exception as e:
        logger.error(f"Error validating predictions: {e}")
        raise 