import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import logging
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.evaluation import ModelEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from evidently import Report
from evidently.presets import RegressionPreset
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluate_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

def load_test_data(data_path="data/processed/test_data.npz"):

    logger.info(f"Loading test data from {data_path}")
    try:
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            # Handle both the case where data is stored as X/y and X_test/y_test
            X_test = data['X_test'] if 'X_test' in data else data['X']
            y_test = data['y_test'] if 'y_test' in data else data['y']
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            X_test = df.iloc[:, :-1].values
            y_test = df.iloc[:, -1].values
        else:
            logger.error(f"Unsupported file format: {data_path}")
            raise ValueError(f"Unsupported file format: {data_path}")
            
        logger.info(f"Test data loaded successfully. X shape: {X_test.shape}, y shape: {y_test.shape}")
        return X_test, y_test
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def load_feature_names(feature_path="data/processed/feature_names.txt"):

    try:
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(feature_names)} feature names")
        return feature_names
    except Exception as e:
        logger.warning(f"Could not load feature names: {e}")
        return None

def evaluate_regression_model(model, X_test, y_test, model_name, feature_names=None):
 
    logger.info(f"Evaluating regression model: {model_name}")
    
    artifacts = []
    
    # Convert data to pandas DataFrame for better visualization
    X_test_df = pd.DataFrame(X_test, columns=feature_names if feature_names else [f"feature_{i}" for i in range(X_test.shape[1])])
    y_test_series = pd.Series(y_test, name="target")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "test_mean_squared_error": mean_squared_error(y_test, y_pred),
        "test_root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "test_r2_score": r2_score(y_test, y_pred)
    }
    
    # Create residual plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred - y_test)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {model_name}")
    residual_plot_path = f"reports/figures/{model_name}_residual_plot.png"
    plt.savefig(residual_plot_path)
    plt.close()
    artifacts.append(residual_plot_path)
    
    # Create actual vs predicted plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted - {model_name}")
    prediction_plot_path = f"reports/figures/{model_name}_prediction_plot.png"
    plt.savefig(prediction_plot_path)
    plt.close()
    artifacts.append(prediction_plot_path)
    
    return metrics, artifacts

def evaluate_with_mlflow(model_uri, test_data, evaluators_config):
  
    X_test, y_test = test_data
    
    # Define evaluation dataset
    eval_data = pd.DataFrame(X_test)
    eval_data["target"] = y_test
    
    # Configure evaluator
    evaluator = ModelEvaluator()
    
    # Run evaluation
    eval_results = evaluator.evaluate(
        model=model_uri,
        data=eval_data,
        targets="target",
        evaluators=evaluators_config
    )
    
    return eval_results

def log_evaluation_to_mlflow(model_path, metrics, artifacts=None, custom_tags=None, model_name=None):
    """
    Log evaluation metrics and artifacts to MLflow
    """
    logger.info("Logging evaluation results to MLflow")
    run_name = f"{model_name}_evaluation" if model_name else "model_evaluation"
    with mlflow.start_run(run_name=run_name):
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        # Log artifacts
        if artifacts:
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
        # Log custom tags
        if custom_tags:
            mlflow.set_tags(custom_tags)
        # Manual validation status
        if metrics.get("test_mean_squared_error", float('inf')) < 1.0 and metrics.get("test_r2_score", 0) > 0.7:
            mlflow.set_tag("validation_status", "pass")
        else:
            mlflow.set_tag("validation_status", "fail")

def main():
    """Main function for model evaluation"""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Use default test data path or override from command line
    test_data_path = "data/processed/test_data.npz"
    if len(sys.argv) > 1:
        test_data_path = sys.argv[1]
    
    # Load test data
    X_test, y_test = load_test_data(test_data_path)
    
    # Load feature names
    feature_names = load_feature_names()
    
    # Find all model files
    model_dir = "models"
    model_files = glob.glob(os.path.join(model_dir, "*_model.joblib"))
    if not model_files:
        logger.error("No model files found in the models directory.")
        sys.exit(1)
    
    all_metrics = {}  # Collect metrics for all models
    for model_path in model_files:
        try:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            logger.info(f"Evaluating model: {model_name}")
            model = joblib.load(model_path)
            
            # Evaluate on test dataset
            metrics, artifacts = evaluate_regression_model(model, X_test, y_test, model_name, feature_names)
            
            # Log evaluation results to MLflow
            log_evaluation_to_mlflow(model_path, metrics, artifacts, model_name=model_name)
            logger.info(f"Evaluation completed for {model_name}")
            
            # Save metrics in the dict, using a clean model name (remove _model suffix)
            clean_name = model_name.replace('_model', '')
            all_metrics[clean_name] = metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_path}: {e}")
    
    # Save all metrics to JSON file
    os.makedirs("reports", exist_ok=True)
    metrics_path = os.path.join("reports", "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"All evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 