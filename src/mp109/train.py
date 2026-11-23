import hashlib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
PARAMS_PATH = ROOT_DIR / "params.yaml"

# Load configuration
with open(PARAMS_PATH, "r", encoding="utf-8-sig") as file:
    CONFIG = yaml.safe_load(file)

# Load and clean training data
df = pd.read_csv(TRAIN_PATH)

from src.mp109.cleaning import clean_data

df = clean_data(df, CONFIG["cleaning"])

# Define input features and target
NUMERIC_FEATURES = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents"
]

CATEGORICAL_FEATURES = [
    "brand_tier",
    "city_zone"
]

TARGET = "resale_thousand_rs"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=CONFIG["training"]["test_size"],
    random_state=CONFIG["training"]["random_state"]
)

# Preprocessing for numerical features
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Preprocessing for categorical features
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ]
)

# Calculate SHA-256 of the original training dataset
with open(TRAIN_PATH, "rb") as file:
    DATA_SHA256 = hashlib.sha256(file.read()).hexdigest()

    # Configure MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("MP109_Used_Two_Wheeler_v1")

# Store experiment results
results = []

for experiment_config in CONFIG["training"]["experiments"]:

    with mlflow.start_run():

        # Create model pipeline
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=experiment_config["n_estimators"],
                    max_depth=experiment_config["max_depth"],
                    random_state=CONFIG["training"]["random_state"]
                ))
            ]
        )

        # Train model
        model.fit(X_train, y_train)

        # Validate model
        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", experiment_config["n_estimators"])
        mlflow.log_param("max_depth", experiment_config["max_depth"])
        mlflow.log_param("data_sha256", DATA_SHA256)

        mlflow.log_metric("mae_validation", mae)

        # Store result
        results.append({
            "model": model,
            "mae": mae,
            "parameters": experiment_config
        })

        print(
            f"Parameters: {experiment_config} | "
            f"Validation MAE: {mae:.4f}"
        )

        # Select the model with the lowest validation MAE
best_result = min(results, key=lambda result: result["mae"])

best_model = best_result["model"]
best_mae = best_result["mae"]

print("\nBest model parameters:", best_result["parameters"])
print(f"Best validation MAE: {best_mae:.4f}")

# Register the best model in MLflow
with mlflow.start_run(run_name="best_v1_model"):

    mlflow.log_metric("best_validation_mae", best_mae)

    mlflow.log_param("data_sha256", DATA_SHA256)

    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="MP109_Used_Two_Wheeler"
    )

    model_version = model_info.registered_model_version

    print("Registered model version:", model_version)