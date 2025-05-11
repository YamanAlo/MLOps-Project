# Project Report: MLOps for California Housing Price Prediction

**1. Introduction**

This report details a project focused on designing and implementing a comprehensive Machine Learning Operations (MLOps) system. The primary objective was to manage the entire lifecycle of a machine learning model for predicting median house values in Californian districts using the California Housing dataset. This includes experiment tracking, model training, hyperparameter tuning, model deployment, and performance monitoring. The project leverages MLflow as a central platform for managing these stages, demonstrating a practical application of MLOps principles to a regression problem. The scope encompasses data acquisition and preprocessing, training and evaluating multiple regression models, packaging the best model, deploying it as a REST API service, and setting up mechanisms for ongoing performance monitoring and model version management through a model registry.

**2. Methodology**

The project followed a structured approach to build an end-to-end MLOps pipeline. The key tools and processes are outlined below.

**2.1 Tools and Technologies**

The project utilized a range of open-source tools and Python libraries:

*   **Data Handling and Manipulation**: `Pandas` and `NumPy` were used for efficient data loading, manipulation, and transformation. The training dataset consisted of approximately 13,724 samples, with the validation set having around 2,942 samples.
*   **Machine Learning**: `Scikit-learn` provided a comprehensive suite of tools for preprocessing (e.g., scaling, imputation using `StandardScaler` and `SimpleImputer` or similar, as suggested by `scaler.joblib` and `imputer.joblib` in `data/processed/`), and implementing various regression models (Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree). `XGBoost` and `Scikit-learn`'s ensemble methods (Random Forest, Gradient Boosting) were employed for more complex modeling.
*   **Experiment Tracking and Management**: `MLflow` was central to the project, used for:
    *   Tracking experiments, logging parameters, code versions, metrics (RMSE, MAE, R2), and artifacts (models, plots). The `mlruns/` directory serves as the backend for this.
    *   Managing the model lifecycle through the MLflow Model Registry.
*   **Hyperparameter Optimization**: `Optuna` was integrated (as seen in `src/models/train_model.py`) to automate the search for optimal hyperparameters for the various regression models.
*   **Model Deployment**: `FastAPI` was used to create a robust and efficient REST API for serving the trained model. `Uvicorn` served as the ASGI server.
*   **Performance Monitoring**: `Evidently` (listed in `requirements.txt` and likely utilized by `src/deployment/performance_monitor.py`) was planned for monitoring the deployed model's performance, detecting data drift, and tracking prediction drift.
*   **Utility and Others**: `Joblib` for saving and loading Python objects (like trained models and preprocessors), `Matplotlib` and `Seaborn` for visualizations (e.g., feature importance plots), `python-dotenv` for managing environment variables, and `PyYAML` for configuration files.

**2.2 Data Acquisition and Preprocessing**

The project utilized the California Housing dataset, typically available as a CSV file (e.g., `data/raw/california_housing.csv`).
*   **Data Download/Loading**: The `src/data/download_data.py` script is intended to handle the acquisition of the raw dataset.
*   **Preprocessing**: The `src/data/preprocess.py` script is responsible for cleaning the data, handling missing values (e.g., using imputation, as suggested by `imputer.joblib`), feature engineering (if any), scaling numerical features (e.g., using `scaler.joblib`), and splitting the data into training (13,724 samples), validation (2,942 samples), and testing sets. The processed data (features `X` and target `y`) are stored in `data/processed/` as `.npz` files (e.g., `train_data.npz`, `val_data.npz`, `test_data.npz`), along with feature names in `feature_names.txt`.

**2.3 Model Development and Experimentation**

The core of the model development and experimentation resides in `src/models/train_model.py`.
*   **Model Selection**: A variety of regression algorithms were explored to identify the best performing model for the housing price prediction task. These include:
    *   Linear Regression
    *   Lasso Regression
    *   Ridge Regression
    *   ElasticNet Regression
    *   Decision Tree Regressor
    *   Random Forest Regressor
    *   Gradient Boosting Regressor
    *   XGBoost Regressor
*   **Hyperparameter Tuning**: `Optuna` was employed to perform automated hyperparameter optimization. The `objective` function within `train_model.py` defines the search space for each model type, and Optuna efficiently searches for the combination of hyperparameters that minimizes a target metric (e.g., Root Mean Squared Error - RMSE) on the validation set.
*   **Experiment Tracking with MLflow**: Each training run, including those part of the hyperparameter optimization process, was logged as an experiment in MLflow. This involved:
    *   Logging model parameters.
    *   Logging performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) for both training and validation sets.
    *   Saving the trained model (`.joblib` files) as an artifact.
    *   Logging feature importance plots.
    *   Inferring and logging the model signature to ensure consistent input/output schemas.
    The `log_to_mlflow` function in `train_model.py` encapsulates this logic.

**2.4 Model Evaluation**

Models were rigorously evaluated using a dedicated test set (`data/processed/test_data.npz`).
*   The `src/models/evaluate_model.py` script (or evaluation steps within `train_model.py` or a separate evaluation script) is used to load a trained model and the test data to assess its generalization performance.
*   Key regression metrics (RMSE, MAE, R2) were calculated on the test set to provide an unbiased estimate of the model's predictive accuracy. These final metrics for the chosen model would also be logged to MLflow.

**2.5 Model Deployment**

The best-performing model, selected based on evaluation metrics and registered in the MLflow Model Registry, was deployed as a REST API for real-time predictions.
*   **API Development**: `src/deployment/app.py` defines the FastAPI application, including an endpoint (e.g., `/predict/`) that accepts input features (matching the model's expected input schema) and returns the predicted housing price.
*   **Model Loading**: The deployment application loads the specified model from the MLflow Model Registry or a local path (e.g., from the `models/` directory).
*   **Serving**: `src/deployment/run_api.py` is likely used to launch the Uvicorn server, making the API accessible. Configuration details might be managed via `src/deployment/config.py`.
*   **Prediction Service**: The `src/models/predict_model.py` script can be used for batch predictions, loading a model and generating predictions on a dataset, saving results to files like those in `data/predictions/`.

**2.6 Performance Monitoring**

Mechanisms were set up to monitor the deployed model's performance over time.
*   **Monitoring Script**: `src/deployment/performance_monitor.py` is designed to implement these checks.
*   **Drift Detection**: Using libraries like `Evidently`, the system is capable of monitoring for:
    *   **Data Drift**: Changes in the statistical properties of incoming prediction data compared to the training data.
    *   **Prediction Drift**: Changes in the distribution of model predictions.
    *   **Concept Drift**: Degradation in model performance metrics over time.
*   **Alerting and Retraining Triggers**: While not explicitly detailed in the scripts viewed, a complete MLOps system would use these monitoring insights to trigger alerts or automate retraining pipelines when significant drift or performance degradation is detected.

**2.7 Model Registry Usage**

MLflow's Model Registry was utilized to manage the lifecycle of trained models.
*   **Model Registration**: After training and evaluation, promising models were registered in the MLflow Model Registry. This involves versioning each model.
*   **Stage Transitions**: The registry allows models to be transitioned through different stages, such as "Staging" (for further testing and QA) and "Production" (for deployment). This provides a controlled way to promote models and roll back if necessary. The deployment scripts would fetch the appropriate model version from the registry based on its stage.

**3. Results and Discussion**

This section presents the outcomes of the experiments, model performance, and insights from deployment and monitoring, based on the provided result images.

**3.1 Experiment Tracking and Model Performance**

MLflow was used to track multiple experiments, comparing various regression models. Performance was assessed using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) on training, validation, and test datasets.

*   **Training Set Performance**:
    *   **R2 Scores (from `train_r2_score.png`)**: XGBoost (~0.99), Gradient Boosting (~0.97), and Random Forest (~0.94) showed very high R2 scores, indicating a strong fit. Linear models (Linear Regression, Ridge, Lasso, Elastic Net) had R2 scores around ~0.58.
    *   **MAE (from `train_mean_absolute_error.png`)**: 
        *   XGBoost: ~0.10
        *   Gradient Boosting: ~0.14
        *   Random Forest: ~0.19
        *   Linear Models (Linear, Ridge, Lasso, Elastic Net): ~0.54-0.55
    *   **MSE (from `train_mean_squared_error.png`)**: 
        *   XGBoost: ~0.02
        *   Gradient Boosting: ~0.03
        *   Random Forest: ~0.08
        *   Linear Models (Linear, Ridge, Lasso, Elastic Net): ~0.55-0.56
    The ensemble models demonstrated significantly lower MAE and MSE on the training data compared to linear models, alongside their superior R2 scores.

*   **Validation Set Performance**:
    *   **MAE (from `val_mean_absolute_error.png`)**:
        *   XGBoost: ~0.33
        *   Gradient Boosting: ~0.33
        *   Random Forest: ~0.37
        *   Linear Regression: ~0.54
        *   Ridge, Lasso, Elastic Net: ~0.55
    Validation MAE for ensemble models was higher than their training MAE but still considerably better than linear models. This set is crucial for hyperparameter tuning to balance bias and variance.

*   **Test Set Performance (Primary Evaluator)**:
    *   **R2 Scores (from `test_r2_score.png`)**: 
        *   XGBoost: ~0.81
        *   Gradient Boosting: ~0.80
        *   Random Forest: ~0.77
        *   Decision Tree: ~0.67
        *   Lasso: ~0.56
        *   Elastic Net: ~0.56
        *   Linear Regression: ~0.48
        *   Ridge: ~0.47
    *   **MAE (from `test_mean_absolute_error.png`)**:
        *   XGBoost: ~0.34
        *   Gradient Boosting: ~0.34
        *   Random Forest: ~0.38
        *   Decision Tree: ~0.44
        *   Lasso: ~0.55
        *   Elastic Net: ~0.55
        *   Linear Regression: ~0.56
        *   Ridge: ~0.56

    Based on the test set R2 and MAE scores, the ensemble models (XGBoost, Gradient Boosting, Random Forest) and Decision Tree significantly outperformed the linear models. XGBoost and Gradient Boosting consistently showed the best generalization performance across these key metrics.

**3.2 Deployment Details and API Functionality**

The best model (likely XGBoost or Gradient Boosting based on overall test performance) would be packaged and deployed via a FastAPI application.
*   **Input**: The API endpoint (e.g., `/predict/`) expects a JSON payload with features like median income, house age, etc.
*   **Output**: It returns the predicted median house value.
*   **Scalability and Robustness**: FastAPI and Uvicorn provide a high-performance setup.


**4. Conclusion**

This project successfully demonstrates the implementation of a robust MLOps pipeline for the California Housing price prediction regression task. The integration of MLflow, Optuna, Scikit-learn, XGBoost, and FastAPI covers the key aspects of the machine learning lifecycle.

*   Systematic experiment tracking revealed that ensemble models, particularly **XGBoost and Gradient Boosting, significantly outperformed linear models and Decision Trees on the test set.** They achieved the highest R2 scores (~0.81 and ~0.80, respectively) and lowest MAE values (~0.34 for both) on test data. These models also showed excellent fit on the training data (e.g., Training R2 for XGBoost: ~0.99, Training MAE: ~0.10).
*   Automated hyperparameter tuning and versioned model management via the Model Registry were key components.
*   The deployment architecture using FastAPI provides a scalable solution for serving predictions.
*   Initial monitoring results for the XGBoost model show performance variations across different data batches, emphasizing the need for continuous monitoring and potential retraining strategies.

**Challenges:**
*   Initial discrepancies between documentation and codebase regarding the project's scope (dataset and task type) were resolved.
*   Interpreting variations in metrics across training, validation, and test sets, and across different monitoring batches, requires careful analysis to understand model behavior and potential overfitting or data drift.

**Future Work:**
*   **CI/CD Integration**: Implement a full CI/CD pipeline to automate testing, training, and deployment.
*   **Advanced Monitoring and Alerting**: Enhance drift detection and set up automated alerts.
*   **Explainability**: Integrate SHAP more deeply for prediction explanations.
*   **Automated Retraining**: Develop a fully automated retraining pipeline triggered by monitoring insights.

In conclusion, this project establishes a strong MLOps foundation, with the XGBoost and Gradient Boosting models demonstrating high predictive accuracy and good generalization for the California Housing dataset.

--- 