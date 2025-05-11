import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import mlflow
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(input_path, output_dir='data/processed', test_size=0.15, val_size=0.15, random_state=42):

    logger.info(f"Preprocessing data from {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    
    # Drop rows with NaN in the target variable
    initial_rows = df.shape[0]
    df.dropna(subset=['MedHouseVal'], inplace=True)
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows due to NaN in target variable 'MedHouseVal'. New shape: {df.shape}")
    else:
        logger.info("No NaN values found in target variable.")

    # Define features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Start MLflow run
    with mlflow.start_run(run_name="preprocess_data"):
        mlflow.log_param("input_file", input_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("random_state", random_state)
        
        # Handle missing values with median imputation
        logger.info("Handling missing values")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Save the imputer for later use
        imputer_path = os.path.join(output_dir, 'imputer.joblib')
        joblib.dump(imputer, imputer_path)
        mlflow.log_artifact(imputer_path)
        
        logger.info("Performing feature engineering")
        X_df = pd.DataFrame(X_imputed, columns=X.columns)
        
        X_df['BedroomRatio'] = X_df['AveBedrms'] / X_df['AveRooms']
        
      
        X_df['PopulationDensity'] = X_df['Population'] / X_df['AveOccup']
        
        logger.info("Scaling features")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        test_fraction = test_size
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_fraction, random_state=random_state
        )
        
        val_fraction = val_size / (1 - test_fraction)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state
        )
        
        logger.info(f"Split data into train ({X_train.shape[0]} samples), validation ({X_val.shape[0]} samples), and test ({X_test.shape[0]} samples) sets")
        
        # Save processed datasets
        train_path = os.path.join(output_dir, 'train_data.npz')
        val_path = os.path.join(output_dir, 'val_data.npz')
        test_path = os.path.join(output_dir, 'test_data.npz')
        
        np.savez(train_path, X=X_train, y=y_train)
        np.savez(val_path, X=X_val, y=y_val)
        np.savez(test_path, X=X_test, y=y_test)
        
        # Save feature names for reference
        feature_names_path = os.path.join(output_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(X_df.columns))
        
        # Log metrics and artifacts
        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("val_samples", X_val.shape[0])
        mlflow.log_metric("test_samples", X_test.shape[0])
        mlflow.log_metric("num_features", X_train.shape[1])
        
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(val_path)
        mlflow.log_artifact(test_path)
        mlflow.log_artifact(feature_names_path)
        
        logger.info(f"Processed train data saved to {train_path}")
        logger.info(f"Processed validation data saved to {val_path}")
        logger.info(f"Processed test data saved to {test_path}")
        
        return train_path, val_path, test_path

if __name__ == "__main__":
    input_file = "data/raw/california_housing.csv"
    preprocess_data(input_file) 