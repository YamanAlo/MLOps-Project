import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import mlflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_california_housing_data(output_dir='data/raw'):
   
    logger.info("Downloading California Housing dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    rows, cols = df.shape
    mask = np.random.random(size=(rows, cols)) < 0.05
    df_with_missing = df.mask(mask)
    
    output_path = os.path.join(output_dir, 'california_housing.csv')
    df_with_missing.to_csv(output_path, index=False)
    
    logger.info(f"Dataset saved to {output_path}")
    
    with mlflow.start_run(run_name="download_data"):
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("rows", rows)
        mlflow.log_param("columns", cols)
        mlflow.log_param("missing_values_percentage", 5)
        mlflow.log_artifact(output_path)
    
    return output_path

if __name__ == "__main__":
    download_california_housing_data() 