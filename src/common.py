from pathlib import Path
import json
import joblib

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
SPLIT_DIR = BASE_DIR / "splits"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_OPTUNA_TRIALS = 50


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "uci": DATA_DIR / "uci_phishing.csv",
    "web_page": DATA_DIR / "web_page_phishing.csv",
    "phiusil": DATA_DIR / "phiusil.csv",
    "zenodo": DATA_DIR / "zenodo_phishing.csv",
}


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

EXPERIMENTS = {
    "E1": {
        "classifier": "RandomForest",
        "optimizer": "Default",
    },
    "E2": {
        "classifier": "RandomForest",
        "optimizer": "GridSearchCV",
    },
    "E3": {
        "classifier": "RandomForest",
        "optimizer": "Optuna",
    },
    "E4": {
        "classifier": "XGBoost",
        "optimizer": "Default",
    },
    "E5": {
        "classifier": "XGBoost",
        "optimizer": "GridSearchCV",
    },
    "E6": {
        "classifier": "XGBoost",
        "optimizer": "Optuna",
    },
}


# ============================================================
# RANDOM FOREST GRIDSEARCH SPACE
# ============================================================

RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


# ============================================================
# XGBOOST GRIDSEARCH SPACE
# ============================================================

XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.10, 0.20],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ============================================================
# RANDOM FOREST OPTUNA SPACE
# ============================================================

RF_OPTUNA = {
    "n_estimators": (100, 300),
    "max_depth": (10, 30),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 4),
    "max_features": ["sqrt", "log2"],
}


# ============================================================
# XGBOOST OPTUNA SPACE
# ============================================================

XGB_OPTUNA = {
    "n_estimators": (100, 300),
    "learning_rate": (0.01, 0.20),
    "max_depth": (3, 7),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
}


# ============================================================
# EVALUATION SETTINGS
# ============================================================

CLASSIFICATION_THRESHOLD = 0.50

LATENCY_WARMUP = 10
LATENCY_SAMPLES = 1000


# ============================================================
# FILE UTILITIES
# ============================================================

def ensure_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def save_split(split_data, dataset_name):
    ensure_directories()

    path = SPLIT_DIR / f"{dataset_name}_split.joblib"
    joblib.dump(split_data, path)

    return path


def load_split(dataset_name):
    path = SPLIT_DIR / f"{dataset_name}_split.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"Canonical split not found for '{dataset_name}'. "
            f"Run train_baseline.py first."
        )

    return joblib.load(path)