import mlflow
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
REGISTERED_MODEL_NAME = "xgboost_model"
MODEL_STAGE = "Production"
# Run ID from the preprocess_data run which logged the imputer, scaler, and feature_names
PREPROCESS_ARTIFACTS_RUN_ID = "ef890bec678b42fc94f1042e06d3db65"
MONITORING_EXPERIMENT_NAME = f"PerformanceMonitoring_{REGISTERED_MODEL_NAME}"
RAW_DATA_PATH = "data/raw/california_housing.csv"
BATCH_SIZE = 10  # Number of rows to print for pred/actual, and also batch size for metrics
TARGET_COLUMN = "MedHouseVal"
MLFLOW_TRACKING_URI = "file:./mlruns" # Assuming default local tracking

# Define original feature columns (excluding target) based on California Housing dataset
ORIGINAL_FEATURE_COLUMNS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

def download_mlflow_artifact(run_id, artifact_path, dst_path):
    """Downloads an artifact from an MLflow run."""
    try:
        client = mlflow.tracking.MlflowClient()
        os.makedirs(dst_path, exist_ok=True)
        local_path = client.download_artifacts(run_id=run_id, path=artifact_path, dst_path=dst_path)
        logger.info(f"Artifact {artifact_path} from run {run_id} downloaded to {local_path}")
        
        return local_path # Use the direct output
    except Exception as e:
        logger.error(f"Failed to download artifact {artifact_path} from run {run_id}: {e}")
        raise

def main():
    logger.info("Starting performance monitoring script...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name(MONITORING_EXPERIMENT_NAME)
        if experiment is None:
            logger.info(f"Experiment '{MONITORING_EXPERIMENT_NAME}' not found. Creating new experiment.")
            experiment_id = mlflow.create_experiment(MONITORING_EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
        logger.info(f"Using experiment ID: {experiment_id}")

    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        return

    # --- Load Model ---
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model '{REGISTERED_MODEL_NAME}' version from stage '{MODEL_STAGE}'.")
    except Exception as e:
        logger.error(f"Failed to load model {model_uri}: {e}")
        return

    # --- Download Preprocessing Artifacts ---
    temp_artifact_dir = "temp_monitoring_artifacts"
    os.makedirs(temp_artifact_dir, exist_ok=True)

    try:
        imputer_artifact_path = "imputer.joblib"
        scaler_artifact_path = "scaler.joblib"
        feature_names_artifact_path = "feature_names.txt"

        downloaded_imputer_path = download_mlflow_artifact(PREPROCESS_ARTIFACTS_RUN_ID, imputer_artifact_path, temp_artifact_dir)
        downloaded_scaler_path = download_mlflow_artifact(PREPROCESS_ARTIFACTS_RUN_ID, scaler_artifact_path, temp_artifact_dir)
        downloaded_feature_names_path = download_mlflow_artifact(PREPROCESS_ARTIFACTS_RUN_ID, feature_names_artifact_path, temp_artifact_dir)
        
        imputer = joblib.load(downloaded_imputer_path)
        scaler = joblib.load(downloaded_scaler_path)
        with open(downloaded_feature_names_path, 'r') as f:
            # These are the feature names AFTER engineering, that the scaler expects
            expected_feature_order_for_scaler = [line.strip() for line in f.readlines()]
        
        logger.info("Successfully loaded preprocessing artifacts.")

    except Exception as e:
        logger.error(f"Failed to load preprocessing artifacts: {e}")
        # Clean up temp dir if created
        if os.path.exists(temp_artifact_dir):
            import shutil
            shutil.rmtree(temp_artifact_dir)
        return

    # --- Process Data in Batches ---
    try:
        logger.info(f"Simulating monitoring using batches of size {BATCH_SIZE} from {RAW_DATA_PATH}")
        batch_num = 0
        for batch_df_raw in pd.read_csv(RAW_DATA_PATH, chunksize=BATCH_SIZE):
            batch_num += 1
            logger.info(f"Processing batch {batch_num}...")

            if TARGET_COLUMN not in batch_df_raw.columns:
                logger.error(f"Target column '{TARGET_COLUMN}' not found in the raw data batch. Skipping.")
                continue
            
            actuals = batch_df_raw[TARGET_COLUMN]
            features_raw_df = batch_df_raw[ORIGINAL_FEATURE_COLUMNS].copy() # Ensure we only use original features for imputation

            # 1. Apply Imputer
            imputed_features_array = imputer.transform(features_raw_df)
            features_imputed_df = pd.DataFrame(imputed_features_array, columns=ORIGINAL_FEATURE_COLUMNS, index=features_raw_df.index)

            # 2. Perform Feature Engineering
            features_engineered_df = features_imputed_df.copy()
           
            features_engineered_df['BedroomRatio'] = features_engineered_df['AveBedrms'] / (features_engineered_df['AveRooms'] + 1e-6) 
            features_engineered_df['PopulationDensity'] = features_engineered_df['Population'] / (features_engineered_df['AveOccup'] + 1e-6)

            # 3. Ensure Column Order for Scaler

            try:
                features_for_scaling_df = features_engineered_df[expected_feature_order_for_scaler]
            except KeyError as e:
                logger.error(f"Missing columns for scaling in batch {batch_num}: {e}. Expected: {expected_feature_order_for_scaler}. Available: {list(features_engineered_df.columns)}")
                continue


            # 4. Apply Scaler
            scaled_features_array = scaler.transform(features_for_scaling_df)
            
            final_features_df = pd.DataFrame(scaled_features_array, columns=expected_feature_order_for_scaler, index=features_for_scaling_df.index)

            # Make Predictions
            predictions = model.predict(final_features_df)

            # Add debugging for NaNs
            logger.info(f"Batch {batch_num} - NaN check before metrics:")
            if actuals.isnull().any():
                logger.info(f"  Actuals (target values) contain {actuals.isnull().sum()} NaN(s). First few actuals: {actuals.head().tolist()}")
            else:
                logger.info("  Actuals (target values) contain NO NaNs.")
            
            predictions_series = pd.Series(predictions) 
            if predictions_series.isnull().any():
                logger.info(f"  Predictions contain {predictions_series.isnull().sum()} NaN(s). First few predictions: {predictions_series.head().tolist()}")
                logger.info(f"  Sample of final_features_df that led to NaN predictions:\n{final_features_df[predictions_series.isnull()].head()}")
            else:
                logger.info("  Predictions contain NO NaNs.")

            valid_indices = ~actuals.isnull()
            if not valid_indices.all():
                logger.info(f"  Filtering out {len(actuals) - valid_indices.sum()} row(s) from batch {batch_num} due to NaN in actuals.")
                actuals_filtered = actuals[valid_indices]
                predictions_filtered = predictions_series[valid_indices.to_numpy()].to_numpy() 
            else:
                actuals_filtered = actuals
                predictions_filtered = predictions 

            if len(actuals_filtered) > 0:
                batch_rmse = np.sqrt(mean_squared_error(actuals_filtered, predictions_filtered))
                batch_r2 = r2_score(actuals_filtered, predictions_filtered)

                logger.info(f"  Batch {batch_num} Results (on {len(actuals_filtered)} valid rows):")
                for i in range(min(len(predictions_filtered), BATCH_SIZE)):
                    logger.info(f"    Row {i+1}: pred={predictions_filtered[i]:.2f}, actual={actuals_filtered.iloc[i]:.2f}")
                
                logger.info(f"  Batch {batch_num} RMSE: {batch_rmse:.2f}")
                logger.info(f"  Batch {batch_num} R2 Score: {batch_r2:.2f}")

                # Log to MLflow
                with mlflow.start_run(experiment_id=experiment_id, run_name=f"Monitoring Batch {batch_num} - {time.strftime('%Y%m%d-%H%M%S')}"):
                    mlflow.log_param("registered_model_name", REGISTERED_MODEL_NAME)
                    mlflow.log_param("model_stage", MODEL_STAGE)
                    mlflow.log_param("preprocess_artifacts_run_id", PREPROCESS_ARTIFACTS_RUN_ID)
                    mlflow.log_param("batch_number", batch_num)
                    mlflow.log_param("batch_size", len(batch_df_raw))
                    mlflow.log_param("data_source", RAW_DATA_PATH)
                    
                    mlflow.log_metric("batch_rmse", batch_rmse)
                    mlflow.log_metric("batch_r2_score", batch_r2)
                    mlflow.log_metric("num_valid_rows_in_batch", len(actuals_filtered))
                    mlflow.log_metric("num_nan_rows_in_batch", len(actuals) - len(actuals_filtered))
                    
                    client = mlflow.tracking.MlflowClient()
                    try:
                        model_version_details = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])[0]
                        mlflow.log_param("monitored_model_version", model_version_details.version)
                        mlflow.log_param("monitored_model_run_id", model_version_details.run_id)
                    except IndexError:
                        logger.warning(f"Could not retrieve model version for {REGISTERED_MODEL_NAME}/{MODEL_STAGE}")
                    except Exception as e_mv:
                        logger.warning(f"Error getting model version details: {e_mv}")

                logger.info(f"Logged live RMSE: {batch_rmse:.2f}")
                logger.info(f"Logged live R2 Score: {batch_r2:.2f}")
            else:
                logger.warning(f"  Batch {batch_num} had no valid rows after filtering NaNs from actuals. Skipping metric calculation and logging for this batch.")

            logger.info(f"--- End of Batch {batch_num} ---")

           
            if batch_num >= 3:
                logger.info("Processed a few batches for demonstration. Stopping.")
                break
        
        logger.info("\nMonitoring simulation complete.")

    except FileNotFoundError:
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}. Please ensure it exists.")
    except Exception as e:
        logger.error(f"An error occurred during batch processing: {e}", exc_info=True)
    finally:
        if os.path.exists(temp_artifact_dir):
            import shutil
            shutil.rmtree(temp_artifact_dir)
            logger.info(f"Cleaned up temporary artifact directory: {temp_artifact_dir}")

if __name__ == "__main__":
    main() 