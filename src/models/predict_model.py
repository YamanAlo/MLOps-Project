import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
import mlflow
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/predict_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def load_model(model_path):

    logger.info(f"Loading model from {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def make_predictions(model, X, output_file=None):

    logger.info(f"Making predictions on {X.shape[0]} samples")
    
    try:
        # Make predictions
        predictions = model.predict(X)
        logger.info(f"Successfully generated {len(predictions)} predictions")
        
        # Save predictions if output file specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save based on file extension
            if output_file.endswith('.csv'):
                pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False)
            elif output_file.endswith('.npy'):
                np.save(output_file, predictions)
            elif output_file.endswith('.npz'):
                np.savez(output_file, predictions=predictions)
            else:
                # Default to numpy format
                np.save(output_file, predictions)
                
            logger.info(f"Predictions saved to {output_file}")
            
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def load_data(input_file):

    logger.info(f"Loading data from {input_file}")
    
    try:
        # Load based on file extension
        if input_file.endswith('.npz'):
            data = np.load(input_file)
            # Check if the npz file has 'X' key
            if 'X' in data:
                X = data['X']
            elif 'X_test' in data:
                X = data['X_test']
            else:
                # Use the first array in the npz file
                X = data[list(data.keys())[0]]
        elif input_file.endswith('.csv'):
            X = pd.read_csv(input_file).values
        elif input_file.endswith('.npy'):
            X = np.load(input_file)
        else:
            logger.error(f"Unsupported file format: {input_file}")
            raise ValueError(f"Unsupported file format: {input_file}")
            
        logger.info(f"Data loaded successfully with shape: {X.shape}")
        return X
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def log_prediction_run(model_path, input_file, output_file, num_predictions, metadata=None):

    logger.info("Logging prediction run to MLflow")
    
    # Set tracking URI to local mlruns directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("input_file", input_file)
            mlflow.log_param("output_file", output_file)
            mlflow.log_param("num_predictions", num_predictions)
            mlflow.log_param("prediction_time", datetime.now().isoformat())
            
            # Log additional metadata if provided
            if metadata:
                for key, value in metadata.items():
                    mlflow.log_param(key, value)
            
            # Log output file as artifact if it exists
            if output_file and os.path.exists(output_file):
                mlflow.log_artifact(output_file)
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Prediction run logged with ID: {run_id}")
            return run_id
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        return None

def main():
    """
    Main function to run predictions
    """
    # Check if input file is provided
    if len(sys.argv) < 2:
        logger.error("No input file specified. Usage: python predict_model.py <input_file> [output_file]")
        sys.exit(1)
        
    # Get input and output file paths
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Default output path
        output_dir = "data/predictions"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    try:
        # Try to load the best model
        model_path = "models/best_model.joblib"
        
        # If best model doesn't exist, look for alternative models
        if not os.path.exists(model_path):
            logger.warning(f"Best model not found at {model_path}")
            model_files = [f for f in os.listdir("models") if f.endswith("_model.joblib")]
            
            if model_files:
                model_path = os.path.join("models", model_files[0])
                logger.info(f"Using alternative model: {model_path}")
            else:
                logger.error("No trained models found in models directory")
                sys.exit(1)
        
        # Load model
        model = load_model(model_path)
        
        # Load data
        X = load_data(input_file)
        
        # Make predictions
        predictions = make_predictions(model, X, output_file)
        
        # Log prediction run to MLflow
        metadata = {
            "model_type": model.__class__.__name__,
            "input_shape": str(X.shape)
        }
        log_prediction_run(model_path, input_file, output_file, len(predictions), metadata)
        
        logger.info(f"Prediction process completed successfully. Made {len(predictions)} predictions.")
        
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 