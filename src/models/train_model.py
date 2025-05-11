import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from datetime import datetime
import optuna
from xgboost import XGBRegressor

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_train_data(data_path="data/processed/train_data.npz"):
 
    logger.info(f"Loading training data from {data_path}")
    
    try:
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        logger.info(f"Loaded training data with shape X: {X.shape}, y: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def load_val_data(data_path="data/processed/val_data.npz"):

    logger.info(f"Loading validation data from {data_path}")
    
    try:
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        logger.info(f"Loaded validation data with shape X: {X.shape}, y: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
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

def objective(trial, X_train, y_train, X_val, y_val, model_type):
    
    if model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = RandomForestRegressor(**params, random_state=42)
    
    elif model_type == "gradient_boosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        }
        model = GradientBoostingRegressor(**params, random_state=42)
    
    elif model_type == "elastic_net":
        params = {
            "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
        }
        model = ElasticNet(**params, random_state=42)
    
    elif model_type == "lasso":
        params = {
            "alpha": trial.suggest_float("alpha", 0.0001, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
            "tol": trial.suggest_float("tol", 0.0001, 0.01),
            
        }
        model = Lasso(**params, random_state=42)
    
    elif model_type == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 0.0001, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
            "tol": trial.suggest_float("tol", 0.0001, 0.01),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
        }
        model = Ridge(**params, random_state=42)
    
    elif model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        
    elif model_type == "linear_regression":
        params = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "n_jobs": trial.suggest_int("n_jobs", 1, 15)
        }
        model = LinearRegression(**params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

def train_model(X_train, y_train, model_type="random_forest", params=None):
 
    logger.info(f"Training {model_type} model")
    
    # Set default parameters if none provided
    if params is None:
        params = {}
    
    # Initialize the specified model type
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", 1),
            random_state=42
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            subsample=params.get("subsample", 1.0),
            random_state=42
        )
    elif model_type == "elastic_net":
        model = ElasticNet(
            alpha=params.get("alpha", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=params.get("max_iter", 1000),
            tol=params.get("tol", 0.0001),
            random_state=42
        )
    elif model_type == "lasso":
        model = Lasso(
            alpha=params.get("alpha", 1.0),
            max_iter=params.get("max_iter", 1000),
            tol=params.get("tol", 0.0001),
            
            random_state=42
        )
    elif model_type == "ridge":
        model = Ridge(
            alpha=params.get("alpha", 1.0),
            max_iter=params.get("max_iter", 1000),
            tol=params.get("tol", 0.0001),
            solver=params.get("solver", "auto"),
            random_state=42
        )
    elif model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            random_state=42,
            verbosity=0
        )
    elif model_type == "linear_regression":
        model = LinearRegression(
            fit_intercept=params.get("fit_intercept", True),
            n_jobs=params.get("n_jobs", None)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type, n_trials=100):
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, model_type),
                  n_trials=n_trials)
    
    return study.best_params

def evaluate_model(model, X, y, dataset_name="train"):

    y_pred = model.predict(X)
    
    metrics = {
        f"{dataset_name}_mean_squared_error": mean_squared_error(y, y_pred),
        f"{dataset_name}_root_mean_squared_error": np.sqrt(mean_squared_error(y, y_pred)),
        f"{dataset_name}_mean_absolute_error": mean_absolute_error(y, y_pred),
        f"{dataset_name}_r2_score": r2_score(y, y_pred)
    }
    
    return metrics

def plot_feature_importance(model, feature_names, output_path):
    
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature importances")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] if feature_names else f"feature_{i}" for i in indices],
               rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def save_model(model, output_path="models/model.joblib"):
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")

def log_to_mlflow(model, model_type, params, metrics, X_train, y_train, feature_names=None, artifacts=None):
    # Set the experiment
    mlflow.set_experiment("regression_models")
    
    # Start a new run
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Create model signature
        signature = infer_signature(X_train, y_train)
        
        # Create input example
        input_example = X_train[:5]
        
        # Log the model with signature and input example
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
            registered_model_name=f"{model_type}_model"
        )
        
        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                if path and os.path.exists(path):
                    mlflow.log_artifact(path)
        
        # Set model version stage
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(f"{model_type}_model", stages=["None"])[0]
        client.transition_model_version_stage(
            name=f"{model_type}_model",
            version=latest_version.version,
            stage="Staging"
        )

def main():
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Load pre-split data
    X_train, y_train = load_train_data()
    X_val, y_val = load_val_data()
    feature_names = load_feature_names()
    
    # Model types to train
    model_types = ["random_forest", "gradient_boosting", "elastic_net", "lasso", "ridge", "xgboost", "linear_regression"]
    
    for model_type in model_types:
        logger.info(f"\nTraining {model_type} model")
        
        # Optimize hyperparameters using validation data
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type)
        logger.info(f"Best parameters for {model_type}: {best_params}")
        
        # Train model with best parameters
        model = train_model(X_train, y_train, model_type, best_params)
        
        # Evaluate model on both training and validation data
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        val_metrics = evaluate_model(model, X_val, y_val, "val")
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        logger.info(f"Training metrics for {model_type}: {train_metrics}")
        logger.info(f"Validation metrics for {model_type}: {val_metrics}")
        
        # Generate artifacts
        artifacts = {}
        if hasattr(model, "feature_importances_"):
            importance_plot_path = f"reports/figures/{model_type}_feature_importance.png"
            plot_feature_importance(model, feature_names, importance_plot_path)
            artifacts["feature_importance_plot"] = importance_plot_path
        
        # Save model
        model_path = f"models/{model_type}_model.joblib"
        save_model(model, model_path)
        artifacts["model_path"] = model_path
        
        # Log to MLflow
        log_to_mlflow(
            model=model,
            model_type=model_type,
            params=best_params,
            metrics=metrics,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            artifacts=artifacts
        )
        
        logger.info(f"Completed training and logging for {model_type}")

if __name__ == "__main__":
    main() 