# California Housing Price Prediction MLOps Project

This project implements a comprehensive machine learning operations (MLOps) workflow for predicting housing prices using the California Housing dataset.

## Project Structure
- `data/`: Houses all data-related files.
  - `raw/`: The original, immutable data files.
  - `processed/`: Cleaned, transformed, and feature-engineered data ready for model training.
  - `predictions/`: Stores predictions made by the models.
- `logs/`: Contains log files generated during various processes (e.g., training, deployment).
- `mlruns/`: Directory used by MLflow to store experiment tracking data, including parameters, metrics, artifacts, and model metadata.
- `models/`: Stores serialized trained models. Each subdirectory typically corresponds to a model type or version (e.g., `elastic_net_model/`, `random_forest_model/`).
- `reports/`: Contains generated reports, analyses, and visualizations.
  - `figures/`: Stores plots, charts, and images used in reports.
- `results/`: General-purpose directory for storing outputs from scripts, such as evaluation metrics or intermediate results.
- `src/`: Contains the source code for the project.
  - `data/`: Python scripts for data loading, preprocessing, and transformation.
  - `deployment/`: Code related to model deployment.
  - `models/`: Python scripts for model definitions, training pipelines, and evaluation.
- `.gitignore`: Specifies files and directories that Git should ignore.
- `Project_Report.md`: The main report document for the project.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training pipeline: `python src/models/train_model.py`
4. Evaluate the models: `python src/models/evaluate_model.py`

## Dataset
The California Housing dataset is used for this project. The goal is to predict median house values in Californian districts, based on various features.

## MLOps Components
- **Experiment Tracking**: MLflow is used to track experiments, parameters, metrics, and artifacts
- **Model Training and Tuning**: Hyperparameter optimization with cross-validation for regression models.
- **Model Registry**: Versioning and stage transitions (development, staging, production)
- **Model Deployment**: REST API for real-time housing price predictions
- **Performance Monitoring**: Drift detection and model retraining triggers 